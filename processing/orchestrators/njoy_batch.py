"""
Batch orchestrator for high-throughput NJOY processing.
Manages the two-phase pipeline: Spectra -> ENDF (MF7), and ENDF -> ACE.
"""
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Any
from parallel_execution import run_parallel

from common.validation import require_existing_file
from processing.backends.njoy_processor import NjoyProcessor
from processing.validators.endf_validator import EndfValidator
from processing.deck_builders.leapr_builder import (
    build_leapr_deck,
    build_leapr_deck_from_module,
    format_njoy_array,
)
from processing.deck_builders.ace_builder import build_ace_deck
from core.data.input_pdos import PhononSpectra
from core.data.spectra_ensemble import SpectraEnsemble
from core.data.material_ensemble import MaterialEnsemble
from core.data.material_scattering import MaterialScatteringModel
from input_output.leapr.core import LeaprFile
from input_output.leapr.workflows import serialize_leapr_spectrum, serialize_leapr_material
from input_output.leapr.temperature import resolve_leapr_temperature

logger = logging.getLogger(__name__)


class NjoyBatchOrchestrator:
    """
    Manages the parallel conversion of a TARA ensemble into ENDF and ACE libraries.
    """

    def __init__(self, njoy_exec: str, n_jobs: int = -1):
        """Initialize the NJOY batch runner and its ENDF validator.

        Parameters
        ----------
        njoy_exec : str
            Path or name of the NJOY executable.
        n_jobs : int, default=-1
            Requested worker count.
        """
        self.njoy_exec = njoy_exec
        self.n_jobs = n_jobs
        self.validator = EndfValidator()


    # --- PHASE 1: GENERATE ENDF WITH S(alpha, beta) ---

    def generate_ensemble_sab_tapes(
        self,
        ensemble: SpectraEnsemble | MaterialEnsemble,
        output_dir: str,
        base_leapr_file: str,
        temperature: float | None = None,
        material_state: str = "solid",
        export_full_spectra: bool = False,
    ) -> Dict[str, str]:
        """Generate validated LEAPR ENDF tapes for a TARA ensemble.

        ``sample_000`` is the separate unperturbed ``ensemble.reference``;
        stochastic ``ensemble[i]`` is written as ``sample_{i+1:03d}``.
        The entire LEAPR temperature list establishes the allowable range, but
        each sample is serialized with exactly one requested temperature.

        Each job loads its own template and runs NJOY in the processor's
        isolated sandbox, permitting parallel execution via ``self.n_jobs``.
        Unlike the legacy ``process_leapr_ensemble``, this method raises if
        *any* tape fails to generate or validate.

        Parameters
        ----------
        ensemble : SpectraEnsemble
            Stochastic spectra and separate reference spectrum.
        output_dir : str
            Directory for sample-specific TSL ENDF outputs.
        base_leapr_file : str
            Original multi-temperature LEAPR template.
        temperature : float or None, optional
            Requested single output temperature in kelvin. If omitted, use
            the original first positive-temperature block. Solids retain
            the legacy five-kelvin margin; liquids require an explicitly
            tabulated temperature and use all of its physical parameters.
        material_state : str, optional
            Optional liquid preflight, preserving the original solid default.
        export_full_spectra : bool, optional
            Export continuous ND=0 spectra instead of native sampled modes.

        Returns
        -------
        dict of str to str
            Ordered sample identifiers mapped to their generated ENDF paths.

        Warns
        -----
        UserWarning
            When the supplied spectrum is reused at another temperature.

        Raises
        ------
        ValueError
            If a liquid temperature is not tabulated, or a solid request
            exceeds the permitted range.
        RuntimeError
            If any NJOY sample fails to generate a valid ENDF tape.
        """
        if ensemble.reference is None:
            raise ValueError("The ensemble has no reference spectrum (sample_000).")
        if isinstance(ensemble, MaterialEnsemble):
            if material_state != 'liquid':
                raise ValueError("MaterialEnsemble processing requires liquid state.")
            if temperature is not None and abs(ensemble.reference.temperature_k - temperature) > 1e-8:
                raise ValueError("Material ensemble reference temperature differs from requested temperature.")

        template_path = Path(base_leapr_file).resolve()
        require_existing_file(template_path, message=f"LEAPR template not found: {template_path}")

        template = LeaprFile.from_file(str(template_path))
        output_temperature = self._resolve_batch_temperature(template, temperature, material_state, ensemble)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        results = run_parallel(
            self._generate_tape_from_template,
            self._leapr_generation_calls(
                ensemble, template_path, out_dir, output_temperature,
                material_state=material_state, export_full_spectra=export_full_spectra,
            ),
            n_jobs=self.n_jobs,
        )
        self._raise_invalid_leapr_batch(results, len(ensemble) + 1)

        logger.info("Generated %d validated NJOY ENDF tapes.", len(results))
        return {sid: path for sid, path, _ in results}

    def _resolve_batch_temperature(self, template, temperature, material_state, ensemble):
        """Resolve the single output temperature before starting any NJOY workers.

        Parameters
        ----------
        template : LeaprFile
            Parsed LEAPR template.
        temperature : float or None
            Requested evaluation temperature in kelvin, if specified.
        material_state : str
            Solid or liquid source model.
        ensemble : SpectraEnsemble or MaterialEnsemble
            Ensemble with its original reference spectrum or material.

        Returns
        -------
        float
            Validated output temperature in kelvin.
        """
        # Validate and warn once, before launching any NJOY worker.
        if material_state == 'liquid':
            # Match one authoritative temperature-dependent liquid block.
            from input_output.leapr.material_adapter import material_model_from_leapr
            material = material_model_from_leapr(
                template, state='liquid', temperature=temperature,
            )
            output_temperature = material.temperature_k
            if isinstance(ensemble, MaterialEnsemble):
                self._validate_liquid_reference(ensemble.reference, material)
        elif material_state == 'solid':
            output_temperature = resolve_leapr_temperature(template, temperature)
        else:
            raise ValueError('material_state must be solid or liquid.')

        return output_temperature

    @staticmethod
    def _validate_liquid_reference(reference, material):
        """Verify the sampled reference against the authoritative LEAPR liquid block.

        Parameters
        ----------
        reference, material : MaterialScatteringModel
            Ensemble reference and source material at the selected temperature.
        """
        if abs(reference.temperature_k - material.temperature_k) > 1e-8:
            raise ValueError('Material ensemble reference temperature does not match LEAPR source.')
        # The unperturbed reference carries the source translation parameters.
        original = material.dynamics
        actual = reference.dynamics
        if (actual.translation_model != original.translation_model
            or abs(actual.translation_weight - original.translation_weight) > 1e-10
            or abs((actual.translation_parameter or 0.0) - (original.translation_parameter or 0.0)) > 1e-10):
            raise ValueError('Material ensemble reference translation differs from LEAPR source.')

    def _leapr_generation_calls(
        self, ensemble: SpectraEnsemble | MaterialEnsemble, template_path: Path, output_dir: Path,
        temperature: float,
        material_state: str = 'solid',
        export_full_spectra: bool = False,
    ):
        """Stream the reference and stochastic LEAPR jobs in stable order.

        Parameters
        ----------
        ensemble : SpectraEnsemble
            Input spectra and separate reference.
        template_path : pathlib.Path
            LEAPR input template.
        output_dir : pathlib.Path
            ENDF destination.
        temperature : float
            Validated output temperature in kelvin.
        material_state : str, default="solid"
            Source model used when selecting the LEAPR temperature block.
        export_full_spectra : bool, default=False
            Write a full continuum spectrum with ND=0.

        Yields
        ------
        tuple
            Per-sample worker positional and keyword arguments.
        """
        yield self._leapr_call(
            "sample_000", ensemble.reference, template_path, output_dir, temperature,
            material_state=material_state, export_full_spectra=export_full_spectra,
        )
        for index in range(len(ensemble)):
            sample_id = f"sample_{index + 1:03d}"
            yield self._leapr_call(
                sample_id, ensemble[index], template_path, output_dir, temperature,
                material_state=material_state, export_full_spectra=export_full_spectra,
            )

    def _leapr_call(
        self, sample_id: str, spectrum: PhononSpectra | MaterialScatteringModel,
        template_path: Path, output_dir: Path, temperature: float,
        material_state: str = 'solid',
        export_full_spectra: bool = False,
    ):
        """Construct one independent LEAPR worker invocation.

        Parameters
        ----------
        sample_id : str
            Unique sample identifier.
        spectrum : PhononSpectra
            Stochastic or reference spectrum.
        template_path : pathlib.Path
            Source LEAPR template.
        output_dir : pathlib.Path
            ENDF destination.
        temperature : float
            Validated output temperature in kelvin.
        material_state : str, default="solid"
            Source model used when selecting the LEAPR temperature block.
        export_full_spectra : bool, default=False
            Write a full continuum spectrum with ND=0.

        Returns
        -------
        tuple
            Worker arguments and keyword options.
        """
        parameters = {
            "sample_id": sample_id,
            "spectrum": spectrum,
            "base_leapr_file": str(template_path),
            "output_path": str(output_dir / f"{sample_id}_sab.endf"),
            "njoy_exec": self.njoy_exec,
            "temperature": temperature,
        }
        if material_state == 'liquid':
            parameters['material_state'] = 'liquid'
        if export_full_spectra:
            parameters['export_full_spectra'] = True
        return ((), parameters)

    @staticmethod
    def _raise_invalid_leapr_batch(results, total: int) -> None:
        """Report every failed LEAPR output instead of silently omitting it.

        Parameters
        ----------
        results : iterable
            Worker triples of sample ID, output path, and optional error.
        total : int
            Total number of reference and stochastic tapes requested.

        Raises
        ------
        RuntimeError
            If one or more ENDF jobs failed.
        """
        failures = [(sid, error) for sid, path, error in results if path is None]
        if not failures:
            return
        details = "; ".join(f"{sid}: {error}" for sid, error in failures[:10])
        remainder = len(failures) - min(10, len(failures))
        if remainder:
            details += f"; ... and {remainder} additional failures"
        raise RuntimeError(
            f"NJOY LEAPR batch failed for {len(failures)} of {total} "
            f"spectra: {details}"
        )

    @staticmethod
    def _generate_tape_from_template(
        sample_id: str,
        spectrum: PhononSpectra | MaterialScatteringModel,
        base_leapr_file: str,
        output_path: str,
        njoy_exec: str,
        temperature: float | None = None,
        material_state: str = 'solid',
        export_full_spectra: bool = False,
    ):
        """Prepare one single-temperature tape in an isolated worker.

        Parameters
        ----------
        sample_id : str
            Stable sample identifier.
        spectrum : PhononSpectra
            Phonon spectrum to process.
        base_leapr_file : str
            Optional LEAPR material template file.
        output_path : str
            File path of generated output.
        njoy_exec : str
            NJOY executable path or name.
        temperature : float | None
            Material or processing temperature in kelvin.
        material_state : str
            Solid or liquid material processing state.
        export_full_spectra : bool, optional
            Export continuous ND=0 spectra instead of native sampled modes.

        Returns
        -------
        object
            Worker result tuple with sample ID, output path, and status.
        """
        target = Path(output_path)
        try:
            # Never accept a valid-looking output left by an earlier run.
            target.unlink(missing_ok=True)

            template = LeaprFile.from_file(base_leapr_file)
            if isinstance(spectrum, MaterialScatteringModel):
                module_text = serialize_leapr_material(
                    leapr_file=template, model=spectrum, temperature=temperature,
                    export_full_spectra=export_full_spectra,
                )
            else:
                module_text = serialize_leapr_spectrum(
                    leapr_file=template, spectrum=spectrum, temperature=temperature,
                    warn_on_temperature_reuse=False,
                    preserve_temperature_parameters=(material_state == 'liquid'),
                    export_full_spectra=export_full_spectra,
                )
            deck = build_leapr_deck_from_module(leapr_module_text=module_text)
            processor = NjoyProcessor(njoy_exec=njoy_exec)
            processor.execute_leapr(
                leapr_deck=deck,
                output_endf_path=str(target),
            )

            if not target.is_file():
                raise RuntimeError("NJOY did not produce an ENDF tape.")
            if not EndfValidator().validate_sab_tape(file_path=str(target)):
                raise ValueError("Generated ENDF failed MF7/MT4 validation.")
            return sample_id, str(target), None
        except Exception as exc:
            target.unlink(missing_ok=True)
            logger.error("Sample %s NJOY LEAPR failed: %s", sample_id, exc)
            return sample_id, None, f"{type(exc).__name__}: {exc}"

    def process_leapr_ensemble(
        self,
        spectra_ensemble: Dict[str, PhononSpectra],
        output_dir: str,
        config: Dict[str, Any]
    ) -> Dict[str, str]:
        """Phase 1: Converts Python PhononSpectra objects into ENDF MF7 tapes.

        Parameters
        ----------
        spectra_ensemble : Dict[str, PhononSpectra]
            Sample identifiers mapped to sampled spectra.
        output_dir : str
            Destination directory for generated outputs.
        config : Dict[str, Any]
            Validated workflow or backend configuration.

        Returns
        -------
        Dict[str, str]
            Successful sample-to-ENDF-path mapping.
        """
        out_dir_path = Path(output_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Phase 1: Starting LEAPR processing for {len(spectra_ensemble)} samples...")

        results = run_parallel(
            self._run_single_leapr,
            (
                ((sample_id, spectra, config, str(out_dir_path / f"{sample_id}_sab.endf")), {})
                for sample_id, spectra in spectra_ensemble.items()
            ),
            n_jobs=self.n_jobs,
        )

        successful_tapes = {sid: path for sid, path in results if path is not None}
        logger.info(f"Phase 1 Complete: {len(successful_tapes)} valid ENDF tapes generated.")
        return successful_tapes

    def _run_single_leapr(self, sample_id: str, spectra: PhononSpectra, config: dict, out_path: str):
        """Generate and validate one LEAPR tape from sampled data.

        Parameters
        ----------
        sample_id : str
            Stable sample identifier.
        spectra : PhononSpectra
            Sampled phonon spectrum.
        config : dict
            LEAPR deck settings.
        out_path : str
            Output path for the ENDF tape.

        Returns
        -------
        tuple
            Sample identifier and tape path, or ``None`` on failure.
        """
        deck = build_leapr_deck(
            mat_id=config['mat_id'],
            lat=config.get('lat', 0),
            mass_amu=config['mass_amu'],
            temperature=config['temperature'],
            nalpha=config['nalpha'],
            nbeta=config['nbeta'],
            alpha_grid_str=config['alpha_grid_str'],
            beta_grid_str=config['beta_grid_str'],
            egrid_str=format_njoy_array(spectra.energy_grid),
            rho_str=format_njoy_array(spectra.density_values)
        )

        processor = NjoyProcessor(njoy_exec=self.njoy_exec)
        try:
            processor.execute_leapr(deck, out_path)
            if self.validator.validate_sab_tape(out_path):
                return sample_id, out_path
            logger.error(f"Sample {sample_id} failed ENDF MF7 validation.")
        except Exception as e:
            logger.error(f"Sample {sample_id} LEAPR execution failed: {str(e)}")
        return sample_id, None

    # --- PHASE 2: GENERATE ACE LIBRARIES ---

    def process_ace_ensemble(
        self,
        sab_endf_files: Dict[str, str],
        output_dir: str,
        config: Dict[str, Any],
        save_xsdir: bool = True,
        *,
        save_njoy_input: bool = False,
    ) -> Dict[str, str]:
        """Convert ENDF MF7 tapes to ACE libraries and optional XSDIR entries.

        Parameters
        ----------
        sab_endf_files : dict[str, str]
            TSL ENDF paths indexed by sample identifier.
        output_dir : str
            Destination for sample-specific ``.ace`` files.
        config : dict[str, Any]
            NJOY material metadata and numerical parameters.
        save_xsdir : bool, optional
            Persist ``.xsdir`` entries under ``output_dir/xsdir``; default True.
        save_njoy_input : bool, optional
            Write or replace ``output_dir/njoy_ace.inp`` before running NJOY.
            Reuse the same deck for all jobs, staging a different ``tape70``
            for each sample.

        Returns
        -------
        dict[str, str]
            Successfully converted sample identifiers mapped to ACE paths.
        """
        if not sab_endf_files:
            return {}

        # The reference and stochastic samples share one LEAPR temperature
        # policy. Check the reference once *before* dispatching any NJOY jobs.
        representative_id = (
            "sample_000" if "sample_000" in sab_endf_files
            else next(iter(sab_endf_files))
        )
        self.validator.validate_tsl_temperature(
            sab_endf_files[representative_id],
            config["temperature"],
            sample_id=representative_id,
        )

        out_dir_path = Path(output_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Phase 2: Starting ACE generation for {len(sab_endf_files)} ENDF tapes...")

        worker_options = {"save_xsdir": save_xsdir}
        if save_njoy_input:
            deck = self._build_sample_ace_deck(config, sab_endf_files[representative_id])
            self._save_ace_input(deck, out_dir_path)
            worker_options["deck"] = deck

        results = run_parallel(
            self._run_single_ace,
            (
                ((sample_id, sab_path, config, str(out_dir_path / f"{sample_id}.ace")),
                 worker_options.copy())
                for sample_id, sab_path in sab_endf_files.items()
            ),
            n_jobs=self.n_jobs,
        )

        successful_aces = {sid: path for sid, path in results if path is not None}
        logger.info(f"Phase 2 Complete: {len(successful_aces)} ACE libraries generated.")
        return successful_aces

    @staticmethod
    def _save_ace_input(deck: str, directory: Path) -> None:
        """Publish the shared input deck before dispatching NJOY jobs.

        Parameters
        ----------
        deck : str
            Complete NJOY command input.
        directory : pathlib.Path
            Existing ACE output directory.
        """
        with TemporaryDirectory(prefix=".njoy_input_", dir=directory) as temporary:
            staged = Path(temporary) / "njoy_ace.inp"
            staged.write_bytes(deck.encode("utf-8"))
            staged.replace(directory / staged.name)

    @staticmethod
    def _build_sample_ace_deck(config: Dict[str, Any], sab_path: str) -> str:
        """Build one ACE deck from shared metadata and a sample TSL tape.

        Parameters
        ----------
        config : dict[str, Any]
            NJOY ACE material and processing options.
        sab_path : str
            Sample-specific TSL ENDF path.

        Returns
        -------
        str
            Fully constructed NJOY ACER deck.
        """
        return build_ace_deck(
            base_endf_path=config['base_endf_path'],
            tsl_endf_path=sab_path,
            base_za=config['base_za'],
            base_mat=config['base_mat'],
            tsl_mat=config['tsl_mat'],
            temp=config['temperature'],
            icoh=config['icoh'],
            natom=config['natom'],
            emax=config['emax'],
            suffix=config['suffix'],
            za1=config['za1'],
            za2=config['za2'],
            za3=config['za3'],
            elas_mat=config['elas_mat'],
            ielas=config['ielas'],
            nbin=config.get('nbin', 16),
            tol=config.get('tol', 0.001),
            mti=config.get('mti', 237),
            mte=config.get('mte', config['elas_mat']),
        )


    def _run_single_ace(
        self, sample_id: str, sab_path: str, config: dict, out_path: str,
        *, save_xsdir: bool = True, deck: str | None = None,
    ):
        """Execute one ACER job in its own NJOY sandbox.

        Parameters
        ----------
        sample_id : str
            Identifier of the sample being converted.
        sab_path : str
            Source MF7 ENDF tape.
        config : dict
            NJOY deck parameters.
        out_path : str
            Target ACE library file.
        save_xsdir : bool, optional
            Whether to retain the XSDIR output, default True.
        deck : str or None, optional
            Shared NJOY input, if already built by the batch.

        Returns
        -------
        tuple[str, str or None]
            Sample identifier and ACE path, or None on failure.
        """
        if deck is None:
            deck = self._build_sample_ace_deck(config, sab_path)

        processor = NjoyProcessor(njoy_exec=self.njoy_exec)
        try:
            processor.execute_ace(
                deck,
                out_path,
                base_endf_path=config['base_endf_path'],
                tsl_endf_path=sab_path,
                save_xsdir=save_xsdir,
            )
            return sample_id, out_path
        except Exception as e:
            logger.error(f"Sample {sample_id} ACE execution failed: {str(e)}")
        return sample_id, None
