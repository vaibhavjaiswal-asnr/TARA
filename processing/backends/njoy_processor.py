"""
Execution sandbox for the NJOY processing pipeline.

Provides both transient context-managed execution for in-memory consumers and
persistent tape extraction for production workflows.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
from typing import Iterator, Mapping, Optional
import uuid

from endf_parserpy import EndfParserPy

from common.validation import require_existing_file
from core.data.input_pdos import PhononSpectra
from core.data.material_scattering import MaterialScatteringModel
from input_output.leapr.core import LeaprFile
from input_output.leapr.workflows import serialize_leapr_spectrum, serialize_leapr_material
from processing.deck_builders.leapr_builder import build_analysis_deck


@dataclass(frozen=True)
class NjoyExecution:
    """Transient state produced by one isolated NJOY execution.

    Parameters
    ----------
    run_dir : pathlib.Path
        Active NJOY sandbox.
    stdout_text : str
        Combined standard output and error from the process.
    """

    run_dir: Path
    stdout_text: str

    def tape_path(self, tape_name: str) -> Path:
        """Return the path to a tape inside the active NJOY sandbox.

        Parameters
        ----------
        tape_name : str
            NJOY logical tape name, such as tape24.

        Returns
        -------
        pathlib.Path
            Path of the tape in the sandbox.
        """
        return self.run_dir / tape_name

    def has_tape(self, tape_name: str) -> bool:
        """Return whether a tape exists inside the active NJOY sandbox.

        Parameters
        ----------
        tape_name : str
            NJOY logical tape name, such as tape24.

        Returns
        -------
        bool
            True if the tape exists as a file.
        """
        return self.tape_path(tape_name).is_file()


class NjoyProcessor:
    """
    Execute NJOY decks inside isolated Processing-owned sandboxes.

    Persistent callers may use the processor only as an execution backend.
    Analyzer callers may additionally configure a base ENDF evaluation and a
    LEAPR template; the processor then prepares a sample-specific LEAPR module
    through the Input/Output serializer before composing and executing NJOY.
    """

    def __init__(
        self,
        njoy_exec: str = None,
        temp_dir: str = None,
        base_endf_file: str = None,
        base_leapr_file: str = None,
    ):
        """Initialize the NJOY backend and optional transient-analysis template.

        Parameters
        ----------
        njoy_exec : str
            NJOY executable path or name.
        temp_dir : str
            Optional scratch directory for backend execution.
        base_endf_file : str
            Optional reference neutron ENDF tape.
        base_leapr_file : str
            Optional LEAPR material template file.
        """
        self.njoy_exec = self._resolve_njoy_path(njoy_exec)
        self.temp_dir = Path(temp_dir) if temp_dir else Path("/tmp")

        self.base_endf_file: Optional[str] = None
        self.base_leapr_file: Optional[str] = None
        self.base_mat: Optional[int] = None
        self._leapr_template: Optional[LeaprFile] = None

        self._post_init(base_endf_file, base_leapr_file)

    def _post_init(
        self,
        base_endf_file: Optional[str],
        base_leapr_file: Optional[str],
    ) -> None:
        """Validate and load the optional transient analysis inputs.

        Parameters
        ----------
        base_endf_file : str or None
            Base neutron ENDF evaluation.
        base_leapr_file : str or None
            LEAPR template paired with the base evaluation.

        Raises
        ------
        ValueError
            If only one of the two input files is supplied.
        """
        if (base_endf_file is None) != (base_leapr_file is None):
            raise ValueError(
                "base_endf_file and base_leapr_file must be provided together "
                "for transient NJOY analysis."
            )

        if base_endf_file is not None and base_leapr_file is not None:
            self._initialize_analysis_template(
                base_endf_file=base_endf_file,
                base_leapr_file=base_leapr_file,
            )

    def _resolve_njoy_path(self, provided_path: str = None) -> str:
        """Resolve and validate the NJOY executable path.

        Parameters
        ----------
        provided_path : str
            Configured NJOY executable path or name.

        Returns
        -------
        str
            Validated NJOY executable path.
        """
        if provided_path:
            resolved_path = provided_path
        elif os.environ.get("TARA_NJOY_EXEC"):
            resolved_path = os.environ.get("TARA_NJOY_EXEC")
        else:
            resolved_path = shutil.which("njoy")

        if not resolved_path or not Path(resolved_path).is_file():
            raise FileNotFoundError(
                "NJOY executable not found. Please ensure 'njoy' is in your PATH, "
                "set the TARA_NJOY_EXEC environment variable, or pass the path explicitly."
            )

        return str(Path(resolved_path).resolve())

    def _initialize_analysis_template(
        self,
        *,
        base_endf_file: str,
        base_leapr_file: str,
    ) -> None:
        """Load reusable Analyzer inputs and derive the base ENDF MAT once.

        Parameters
        ----------
        base_endf_file : str
            Optional reference neutron ENDF tape.
        base_leapr_file : str
            Optional LEAPR material template file.
        """
        endf_path = Path(base_endf_file).resolve()
        leapr_path = Path(base_leapr_file).resolve()

        if not endf_path.is_file():
            raise FileNotFoundError(
                f"Base ENDF evaluation not found at: {endf_path}"
            )
        if not leapr_path.is_file():
            raise FileNotFoundError(
                f"Base LEAPR template not found at: {leapr_path}"
            )

        self.base_endf_file = str(endf_path)
        self.base_leapr_file = str(leapr_path)
        self.base_mat = self._read_base_mat(self.base_endf_file)
        self._leapr_template = LeaprFile.from_file(self.base_leapr_file)

    @staticmethod
    def _read_base_mat(base_endf_file: str) -> int:
        """Read MAT from the standard ENDF MF=1/MT=451 header.

        Parameters
        ----------
        base_endf_file : str
            Path to the base neutron evaluation.

        Returns
        -------
        int
            MAT identifier from MF1/MT451.

        Raises
        ------
        ValueError
            If the header has no valid MAT identifier.
        """
        tape_data = EndfParserPy().parsefile(base_endf_file)
        try:
            return int(tape_data[1][451]["MAT"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Base ENDF evaluation is missing a valid MAT value in "
                f"MF=1/MT=451: {base_endf_file}."
            ) from exc

    @staticmethod
    def _validate_tape_name(tape_name: str) -> None:
        """Reject paths and malformed logical-tape names.

        Parameters
        ----------
        tape_name : str
            Requested filename within the sandbox.

        Raises
        ------
        ValueError
            If the name does not match the tapeNN convention.
        """
        if (
            not tape_name.startswith("tape")
            or tape_name != Path(tape_name).name
            or not tape_name[4:].isdigit()
        ):
            raise ValueError(
                f"Invalid NJOY logical tape name: {tape_name!r}."
            )

    @classmethod
    def _stage_input_tapes(
        cls,
        run_dir: Path,
        input_tapes: Optional[Mapping[str, str]],
    ) -> None:
        """Copy external inputs into the Processing-owned NJOY sandbox.

        Parameters
        ----------
        run_dir : Path
            Temporary NJOY execution directory.
        input_tapes : Optional[Mapping[str, str]]
            External input tape names and source paths.
        """
        if not input_tapes:
            return

        for tape_name, source_path in input_tapes.items():
            cls._validate_tape_name(tape_name)
            source = Path(source_path).resolve()
            require_existing_file(
                source, message=f"NJOY input for {tape_name} not found at: {source}",
            )
            shutil.copy(str(source), str(run_dir / tape_name))

    @contextmanager
    def execute(
        self,
        deck: str,
        input_tapes: Optional[Mapping[str, str]] = None,
    ) -> Iterator[NjoyExecution]:
        """Execute an NJOY deck and expose its transient sandbox.

        Parameters
        ----------
        deck : str
            NJOY command deck text.
        input_tapes : Optional[Mapping[str, str]]
            External input tape names and source paths.

        Yields
        ------
        NjoyExecution
            Transient execution and its sandbox.
        """
        # NJOY uses fixed tape names, so give each execution its own directory.
        run_id = uuid.uuid4().hex[:8]
        run_dir = self.temp_dir / f"njoy_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            self._stage_input_tapes(run_dir, input_tapes)

            stdout_text = self._run_deck(run_dir, deck)

            yield NjoyExecution(
                run_dir=run_dir,
                stdout_text=stdout_text,
            )
        finally:
            shutil.rmtree(str(run_dir), ignore_errors=True)

    def _run_deck(self, run_dir, deck):
        """Send the deck to NJOY and validate its process result.

        Parameters
        ----------
        run_dir : pathlib.Path
            Isolated directory with staged NJOY inputs.
        deck : str
            NJOY deck text passed on standard input.

        Returns
        -------
        str
            The captured NJOY output.
        """
        process = subprocess.Popen(
            [str(self.njoy_exec)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(run_dir),
            text=True,
        )
        stdout_text, _ = process.communicate(input=deck)
        stdout_text = stdout_text or ""

        # Some NJOY errors appear only in the transcript.
        if process.returncode != 0 or "error" in stdout_text.lower():
            raise RuntimeError(
                f"NJOY execution failed.\nOutput:\n{stdout_text}"
            )

        return stdout_text

    @contextmanager
    def execute_analysis(
        self,
        spectrum: PhononSpectra,
        *,
        temperature: float,
        thermr_emax: float = 5.0,
        thermr_tol: float = 0.05,
        thermr_nbin: int = 16,
        export_full_spectra: bool = False,
    ) -> Iterator[NjoyExecution]:
        """Prepare and execute one Analyzer spectrum through NJOY transiently.

        Parameters
        ----------
        spectrum : PhononSpectra
            Phonon spectrum to process.
        temperature : float
            Material or processing temperature in kelvin.
        thermr_emax : float
            THERMR maximum outgoing energy in electronvolts.
        thermr_tol : float
            THERMR numerical tolerance.
        thermr_nbin : int
            Number of THERMR angular bins.
        export_full_spectra : bool, optional
            Export continuous ND=0 spectra instead of native sampled modes.

        Yields
        ------
        Iterator[NjoyExecution]
            Transient NJOY execution context.
        """
        self._require_analysis_template()
        leapr_module_text = serialize_leapr_spectrum(
            self._leapr_template,
            spectrum,
            temperature=temperature,
            export_full_spectra=export_full_spectra,
        )
        with self._execute_analysis_deck(
            leapr_module_text, temperature, thermr_emax, thermr_tol, thermr_nbin,
        ) as execution:
            yield execution

    @contextmanager
    def execute_material_analysis(
        self, material: MaterialScatteringModel, *, temperature: float,
        thermr_emax: float = 5.0, thermr_tol: float = 0.05,
        thermr_nbin: int = 16, export_full_spectra: bool = False,
    ) -> Iterator[NjoyExecution]:
        """Execute LEAPR and THERMR with a complete sampled liquid material.

        Parameters
        ----------
        material : MaterialScatteringModel
            Realization with vibrational data and sampled translation settings.
        temperature : float
            Configured processing temperature in kelvin.
        thermr_emax : float, optional
            THERMR upper energy in eV.
        thermr_tol : float, optional
            THERMR interpolation tolerance.
        thermr_nbin : int, optional
            THERMR angular bins.
        export_full_spectra : bool, optional
            Write the full vibrational spectrum as a continuum.

        Yields
        ------
        NjoyExecution
            Transient output tapes, valid within this context.
        """
        self._require_analysis_template()
        # Material serialization preserves the sampled Card 13 coefficients.
        module = serialize_leapr_material(
            self._leapr_template, material, temperature=temperature,
            export_full_spectra=export_full_spectra,
        )
        with self._execute_analysis_deck(
            module, temperature, thermr_emax, thermr_tol, thermr_nbin,
        ) as execution:
            yield execution

    def _require_analysis_template(self):
        """Require both input files before preparing an analysis deck.

        Raises
        ------
        RuntimeError
            If the base evaluation or LEAPR template was omitted.
        """
        if (
            self._leapr_template is None
            or self.base_endf_file is None
            or self.base_mat is None
        ):
            raise RuntimeError(
                "Transient NJOY analysis requires base_endf_file and "
                "base_leapr_file when constructing NjoyProcessor."
            )

    @contextmanager
    def _execute_analysis_deck(
        self, leapr_module_text, temperature, thermr_emax, thermr_tol, thermr_nbin,
    ):
        """Assemble and execute the common LEAPR and THERMR sequence.

        Parameters
        ----------
        leapr_module_text : str
            Prepared sample-specific LEAPR input.
        temperature, thermr_emax : float
            Processing temperature in kelvin and upper THERMR energy in eV.
        thermr_tol : float
            THERMR interpolation tolerance.
        thermr_nbin : int
            Number of THERMR angular bins.

        Yields
        ------
        NjoyExecution
            Execution with staged input and output tapes.
        """
        leapr_mat = int(self._leapr_template.material_info["mat"])
        thermr_natom = int(self._leapr_template.principal_scatterer["npr"])
        thermr_icoh = (
            1 if int(self._leapr_template.principal_scatterer.get("iel", 0)) != 0 else 0
        )

        deck = build_analysis_deck(
            base_mat=self.base_mat,
            leapr_mat=leapr_mat,
            temperature=float(temperature),
            leapr_module_text=leapr_module_text,
            thermr_emax=thermr_emax,
            thermr_tol=thermr_tol,
            thermr_nbin=thermr_nbin,
            thermr_icoh=thermr_icoh,
            thermr_natom=thermr_natom,
        )

        with self.execute(
            deck,
            input_tapes={"tape20": self.base_endf_file},
        ) as execution:
            yield execution

    def execute_leapr(self, leapr_deck: str, output_endf_path: str) -> None:
        """Execute a LEAPR deck and persist the generated ``tape24``.

        Parameters
        ----------
        leapr_deck : str
            LEAPR deck that writes tape24.
        output_endf_path : str
            Destination ENDF path.
        """
        self._run_njoy_sandbox(
            leapr_deck,
            target_tape="tape24",
            output_path=output_endf_path,
        )

    def execute_ace(
        self,
        ace_deck: str,
        output_ace_path: str,
        *,
        base_endf_path: str,
        tsl_endf_path: str,
        save_xsdir: bool = True,
    ) -> None:
        """Persist ACER's ACE tape40 and optionally its XSDIR tape41.

        Parameters
        ----------
        ace_deck : str
            NJOY deck which emits ACE on tape40 and XSDIR on tape41.
        output_ace_path : str
            Destination for the true ACE library.
        base_endf_path : str
            Base neutron ENDF input.
        tsl_endf_path : str
            Thermal scattering ENDF input.
        save_xsdir : bool, optional
            Save ``tape41`` to ``<ace_dir>/xsdir/<sample>.xsdir``.
            Defaults to ``True``.

        Raises
        ------
        FileNotFoundError
            If NJOY does not produce the expected output tape(s).
        ValueError
            If the selected ACE output looks like an XSDIR entry.
        """
        ace_target = Path(output_ace_path)
        xsdir_target = ace_target.parent / "xsdir" / f"{ace_target.stem}.xsdir"
        ace_target.unlink(missing_ok=True)
        # A disabled XSDIR option must not leave an old file masquerading as new.
        xsdir_target.unlink(missing_ok=True)
        try:
            with self.execute(
                ace_deck,
                input_tapes={"tape20": base_endf_path, "tape70": tsl_endf_path},
            ) as execution:
                ace_tape, index_tape = self._validate_acer_tapes(execution, save_xsdir)
                ace_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ace_tape, ace_target)
                if save_xsdir:
                    xsdir_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(index_tape, xsdir_target)
        except Exception:
            ace_target.unlink(missing_ok=True)
            xsdir_target.unlink(missing_ok=True)
            raise

    @staticmethod
    def _validate_acer_tapes(execution, save_xsdir):
        """Validate ACER's ACE library and optional index tape.

        Parameters
        ----------
        execution : NjoyExecution
            Completed NJOY execution with access to its sandbox.
        save_xsdir : bool
            Whether the XSDIR index is required.

        Returns
        -------
        tuple of pathlib.Path and pathlib.Path or None
            ACE tape path and optional XSDIR tape path.
        """
        ace_tape = execution.tape_path("tape40")
        if not ace_tape.is_file() or ace_tape.stat().st_size == 0:
            raise FileNotFoundError(
                f"NJOY did not generate a nonempty ACE tape40. "
                f"Output:\n{execution.stdout_text}"
            )
        # Guard against accidentally extracting a short XSDIR index
        # entry as an ACE library again.
        with ace_tape.open("rb") as source:
            if b"filename route" in source.read(1024).lower():
                raise ValueError("NJOY ACE tape40 contains an XSDIR entry, not ACE data.")
        index_tape = None
        if save_xsdir:
            index_tape = execution.tape_path("tape41")
            if not index_tape.is_file() or index_tape.stat().st_size == 0:
                raise FileNotFoundError(
                    f"NJOY did not generate a nonempty XSDIR tape41. "
                    f"Output:\n{execution.stdout_text}"
                )
        return ace_tape, index_tape

    def _run_njoy_sandbox(
        self,
        deck: str,
        target_tape: str,
        output_path: str,
        input_tapes: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Execute NJOY transiently and persist one selected tape.

        Parameters
        ----------
        deck : str
            NJOY command deck text.
        target_tape : str
            Output tape name to retain.
        output_path : str
            File path of generated output.
        input_tapes : Optional[Mapping[str, str]]
            External input tape names and source paths.
        """
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with self.execute(deck, input_tapes=input_tapes) as execution:
            tape_path = execution.tape_path(target_tape)

            if not tape_path.is_file():
                raise FileNotFoundError(
                    f"NJOY failed to generate {target_tape}. "
                    f"Output:\n{execution.stdout_text}"
                )

            shutil.copy(str(tape_path), str(target_path))
