"""
Core data models for the processing domain.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import math

from common.validation import require_finite_positive


@dataclass
class AceGenerationConfig:
    """Configuration for the NJOY ACE generation sequence.

    Parameters
    ----------
    base_endf_path : str
        Path to the base neutron ENDF evaluation.
    base_za : int
        ZA identifier of the base evaluation.
    base_mat : int
        MAT identifier of the base evaluation.
    tsl_mat : int
        MAT identifier of the thermal-scattering evaluation.
    temperature : float
        Processing temperature in kelvin.
    icoh : int
        Flag indicating whether an elastic TSL section is present.
    natom : int
        Number of principal scattering atoms.
    emax : float
        Upper energy limit for thermal processing.
    suffix : str
        ACE table identifier suffix.
    za1, za2, za3 : int
        Moderator ZA identifiers for the NJOY thermal processing deck.
    elas_mat : int
        Legacy name for the thermal elastic reaction MT used by ACER.
        Must agree with ``mte`` when ``mte`` is specified.
    ielas : int
        Elastic treatment option for NJOY.
    nbin : int, optional
        THERMR angular bin count. Default is 16; unrelated to ACER ``nbint``.
    tol : float, optional
        NJOY processing tolerance. Default is 0.001.
    mti : int, optional
        Inelastic thermal reaction MT used by THERMR and ACER.
    mte : int or None, optional
        Elastic thermal reaction MT; ``None`` preserves the legacy
        ``elas_mat`` value and zero denotes no ACE elastic reaction.
    """
    # Source evaluation and material identifiers.
    base_endf_path: str
    base_za: int
    base_mat: int
    tsl_mat: int

    # Thermal scattering controls.
    temperature: float
    icoh: int
    natom: int
    emax: float

    # ACE table and isotope identifiers.
    suffix: str
    za1: int
    za2: int
    za3: int

    # Elastic treatment and reaction options.
    elas_mat: int
    ielas: int
    nbin: int = 16
    tol: float = 0.001
    mti: int = 237
    mte: int | None = None

    def __post_init__(self):
        """Validate reaction identifiers and require the base ENDF input.

        Raises
        ------
        FileNotFoundError
            If the base evaluation file does not exist.
        ValueError
            If the reaction identifiers are invalid or conflicting.
        """
        self._validate_reaction_ids()
        if self.mte is None:
            # Keep the legacy constructor's elastic MT when no override is supplied.
            self.mte = self.elas_mat
        if not Path(self.base_endf_path).is_file():
            raise FileNotFoundError(
                f"Base ENDF evaluation not found at: {self.base_endf_path}"
            )

    def _validate_reaction_ids(self) -> None:
        """Check the compatible THERMR and ACER reaction identifiers.

        Raises
        ------
        ValueError
            If an MT identifier is invalid or the two elastic values disagree.
        """
        if type(self.mti) is not int or not 221 <= self.mti <= 250:
            raise ValueError("THERMR/ACER inelastic reaction mti must be an integer from 221 to 250.")
        if type(self.elas_mat) is not int or self.elas_mat < 0:
            raise ValueError("ACER elastic reaction elas_mat must be a nonnegative integer.")
        if self.mte is None:
            return
        if type(self.mte) is not int or self.mte < 0:
            raise ValueError("ACER elastic reaction mte must be a nonnegative integer.")
        if self.mte != self.elas_mat:
            raise ValueError("Conflicting ACER elastic reactions: mte and elas_mat must agree.")

    @staticmethod
    def _positive_endf_identifier(raw, label: str) -> int:
        """Convert and validate a positive integer-valued ENDF field.

        Parameters
        ----------
        raw : object
            Raw value returned by the ENDF parser.
        label : str
            Field description used in error messages.

        Returns
        -------
        int
            Validated positive integer.

        Raises
        ------
        ValueError
            If the value is nonfinite, nonpositive, or not integer-valued.
        """
        try:
            number = float(raw)
            value = int(round(number))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid ENDF {label}: {raw!r}.") from exc
        if not math.isfinite(number) or value <= 0 or abs(number - value) > 1e-5:
            raise ValueError(f"Invalid ENDF {label}: {raw!r}.")
        return value

    @staticmethod
    def _parse_endf(parser, path: Path, description: str) -> Mapping:
        """Parse a nonempty ENDF file with contextual error reporting.

        Parameters
        ----------
        parser : EndfParserPy
            ENDF parser instance.
        path : pathlib.Path
            Path to the ENDF file.
        description : str
            Human-readable file description for diagnostics.

        Returns
        -------
        collections.abc.Mapping
            Parsed ENDF sections.

        Raises
        ------
        FileNotFoundError
            If the file is missing or empty.
        ValueError
            If parsing fails or the result is not a mapping.
        """
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty {description}: {path}")
        try:
            data = parser.parsefile(str(path))
        except Exception as exc:
            raise ValueError(f"Could not parse {description} ({path}): {exc}") from exc
        if not isinstance(data, Mapping):
            raise ValueError(f"Unexpected ENDF parser output for {description}: {path}")
        return data

    @classmethod
    def _read_base_metadata(cls, parser, base_path: Path) -> tuple[int, int]:
        """Extract the base evaluation identifiers from MF1/MT451.

        Parameters
        ----------
        parser : EndfParserPy
            ENDF parser instance.
        base_path : pathlib.Path
            Path to the base neutron evaluation.

        Returns
        -------
        tuple of int
            Base ZA and MAT identifiers, respectively.

        Raises
        ------
        FileNotFoundError
            If the base evaluation is missing or empty.
        ValueError
            If required metadata is missing or invalid.
        """
        data = cls._parse_endf(parser, base_path, "base neutron ENDF evaluation")
        try:
            header = data[1][451]
            return (
                cls._positive_endf_identifier(header["ZA"], "base ZA"),
                cls._positive_endf_identifier(header["MAT"], "base MAT"),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("Base ENDF is missing MF1/MT451 ZA/MAT metadata.") from exc

    @classmethod
    def _read_tsl_metadata(
        cls, parser, sample_id: str, filename: str,
        *, allow_additional_scatterers: bool = False,
    ) -> tuple[int, int, int, int]:
        """Extract NJOY metadata from one thermal-scattering ENDF tape.

        Parameters
        ----------
        parser : EndfParserPy
            ENDF parser instance.
        sample_id : str
            Identifier of the thermal-scattering sample.
        filename : str
            Path to the thermal-scattering ENDF tape.
        allow_additional_scatterers : bool, optional
            Permit additional scatterers in an explicitly selected ACE model.

        Returns
        -------
        tuple of int
            TSL MAT, principal multiplicity, LTHR and secondary NS count.

        Raises
        ------
        FileNotFoundError
            If the TSL file is missing or empty.
        ValueError
            If required metadata is missing, invalid, or describes
            unsupported additional moderator species.
        """
        tape = cls._parse_endf(parser, Path(filename), f"TSL ENDF for {sample_id}")
        try:
            inelastic = tape[7][4]
            tsl_mat = cls._positive_endf_identifier(inelastic["MAT"], f"{sample_id} TSL MAT")
        except (KeyError, TypeError) as exc:
            raise ValueError(f"TSL ENDF for {sample_id} is missing MF7/MT4 MAT metadata.") from exc

        try:
            natom = cls._positive_endf_identifier(inelastic["B"][6], f"{sample_id} MF7/MT4 B(6)")
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                f"TSL ENDF for {sample_id} has no MF7/MT4 B(6) principal-atom "
                "multiplicity. Use an explicit AceGenerationConfig if the file cannot supply it."
            ) from exc

        try:
            ns = int(inelastic["NS"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"TSL ENDF for {sample_id} has no valid MF7/MT4 NS atom-type "
                "count; cannot determine the moderator composition."
            ) from exc
        if ns < 0:
            raise ValueError(f"Invalid negative MF7/MT4 NS for {sample_id}: {ns}.")
        if ns != 0 and not allow_additional_scatterers:
            raise ValueError(
                f"TSL ENDF for {sample_id} declares NS={ns} additional atom "
                "types. Their moderator ZA entries require explicit AceGenerationConfig settings."
            )

        lthr = cls._read_elastic_flag(tape, sample_id)
        return tsl_mat, natom, lthr, ns

    @staticmethod
    def _read_elastic_flag(tape, sample_id):
        """Read and validate the optional MF7/MT2 LTHR flag.

        Parameters
        ----------
        tape : Mapping
            Parsed TSL ENDF evaluation.
        sample_id : str
            Sample identifier used in diagnostics.

        Returns
        -------
        int
            LTHR code, or zero when the elastic section is absent.
        """
        elastic = tape.get(7, {}).get(2)
        try:
            lthr = int(elastic["LTHR"]) if elastic is not None else 0
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid MF7/MT2 elastic metadata for {sample_id}.") from exc
        if lthr not in (0, 1, 2, 3):
            raise ValueError(f"Unsupported MF7/MT2 LTHR={lthr} for {sample_id}.")
        return lthr

    @classmethod
    def _read_shared_tsl_metadata(
        cls, parser, tsl_endf_files: Mapping[str, str],
        *, allow_additional_scatterers: bool = False,
    ) -> tuple[int, int, int]:
        """Check that all TSL samples share one ACE configuration.

        Parameters
        ----------
        parser : EndfParserPy
            ENDF parser instance.
        tsl_endf_files : Mapping[str, str]
            Mapping of sample identifiers to TSL ENDF file paths.
        allow_additional_scatterers : bool, optional
            Accept secondary species only with explicit handling.

        Returns
        -------
        tuple of int
            Shared TSL MAT, principal atom multiplicity, and LTHR flag.

        Raises
        ------
        ValueError
            If no files are supplied or their metadata is incompatible.
        """
        shared = None
        for sample_id, filename in tsl_endf_files.items():
            metadata = cls._read_tsl_metadata(
                parser, sample_id, filename,
                allow_additional_scatterers=allow_additional_scatterers,
            )
            if shared is None:
                shared = metadata
            elif metadata != shared:
                raise ValueError(
                    f"Inconsistent TSL MAT/NATOM/LTHR in {sample_id}: {metadata}; "
                    f"expected {shared}. All files must use one ACE configuration."
                )
        if shared is None:
            raise ValueError("No thermal-scattering ENDF files were supplied.")
        # NS must agree across samples even though it is not an ACE card.
        return shared[:3]

    @staticmethod
    def _validate_ace_options(temperature: float, emax: float, nbin: int, tol: float) -> None:
        """Validate numerical NJOY processing options.

        Parameters
        ----------
        temperature : float
            Processing temperature in kelvin.
        emax : float
            Upper thermal energy limit.
        nbin : int
            Number of angular bins.
        tol : float
            NJOY processing tolerance.

        Raises
        ------
        ValueError
            If an option is outside its supported range.
        """
        require_finite_positive(temperature, message="ACE temperature must be a finite positive number in kelvin.")
        require_finite_positive(emax, message="ACE thermal upper energy must be finite and positive.")
        if not isinstance(nbin, int) or nbin < 1:
            raise ValueError("ACE angular bin count must be a positive integer.")
        require_finite_positive(tol, message="ACE tolerance must be finite and positive.")

    @staticmethod
    def _resolve_inferred_reactions(lthr, mti, mte, allow_additional_scatterers):
        """Resolve NJOY reaction identifiers from inferred elastic metadata.

        Parameters
        ----------
        lthr : int
            ENDF MF7/MT2 elastic treatment flag.
        mti, mte : int or None
            Optional inelastic and elastic MT overrides.
        allow_additional_scatterers : bool
            Whether the caller enabled support for additional scatterers.

        Returns
        -------
        tuple of int
            Resolved inelastic and elastic MT identifiers.
        """
        if allow_additional_scatterers and mti is None:
            raise ValueError("Explicit ACER mti required for additional scatterer models.")
        resolved_mti = 237 if mti is None else mti
        resolved_mte = (
            (resolved_mti + 1 if lthr else 0)
            if mte is None else mte
        )
        if type(resolved_mte) is not int or resolved_mte < 0:
            raise ValueError("ACER elastic reaction mte must be a nonnegative integer.")
        if resolved_mte != 0 and not lthr:
            raise ValueError("A nonzero ACER mte requires an MF7/MT2 elastic section.")
        if lthr and resolved_mte != 0 and resolved_mte != resolved_mti + 1:
            raise ValueError("ACER elastic mte must equal THERMR mti + 1 for this model.")
        return resolved_mti, resolved_mte

    @classmethod
    def from_endf(
        cls,
        *,
        base_endf_path: str,
        tsl_endf_files: Mapping[str, str],
        temperature: float,
        emax: float = 5.0,
        suffix: str = ".01",
        nbin: int = 16,
        tol: float = 0.001,
        mti: int | None = None,
        mte: int | None = None,
        allow_additional_scatterers: bool = False,
    ) -> "AceGenerationConfig":
        """Build an ACE configuration from neutron and TSL ENDF evaluations.

        Automatic inference supports one principal scattering species
        (MF7/MT4 ``NS=0``). Additional moderator species require explicit
        validation with ``allow_additional_scatterers=True`` and ``mti``.

        Parameters
        ----------
        base_endf_path : str
            Path to the base neutron ENDF evaluation.
        tsl_endf_files : Mapping[str, str]
            Mapping of sample identifiers to thermal-scattering ENDF paths.
            All samples must have compatible MAT, NATOM, and LTHR metadata.
        temperature : float
            Processing temperature in kelvin.
        emax : float, optional
            Upper thermal energy limit. Default is 5.0.
        suffix : str, optional
            ACE table suffix. Default is ``'.01'``.
        nbin : int, optional
            Number of angular bins. Default is 16.
        tol : float, optional
            NJOY processing tolerance. Default is 0.001.
        mti : int or None, optional
            Explicit THERMR/ACER inelastic MT. Defaults to legacy 237.
        mte : int or None, optional
            Explicit ACER elastic MT; use zero when absent.
        allow_additional_scatterers : bool, optional
            Permit secondary species in a verified principal-only ACE model.

        Returns
        -------
        AceGenerationConfig
            Configuration inferred from the ENDF metadata.

        Raises
        ------
        FileNotFoundError
            If the base evaluation or a TSL file is missing or empty.
        ValueError
            If the processing options or ENDF metadata are invalid, or
            the TSL samples cannot share one configuration.
        """
        from endf_parserpy import EndfParserPy

        cls._validate_ace_options(temperature, emax, nbin, tol)
        if not tsl_endf_files:
            raise ValueError("No thermal-scattering ENDF files were supplied.")

        base_path = Path(base_endf_path)
        if not base_path.is_file() or base_path.stat().st_size == 0:
            raise FileNotFoundError(f"Base ENDF evaluation is missing or empty: {base_path}")

        parser = EndfParserPy()
        base_za, base_mat = cls._read_base_metadata(parser, base_path)
        tsl_mat, natom, lthr = cls._read_shared_tsl_metadata(
            parser, tsl_endf_files,
            allow_additional_scatterers=allow_additional_scatterers,
        )
        resolved_mti, resolved_mte = cls._resolve_inferred_reactions(
            lthr, mti, mte, allow_additional_scatterers,
        )
        return cls(
            base_endf_path=str(base_path),
            base_za=base_za,
            base_mat=base_mat,
            tsl_mat=tsl_mat,
            temperature=temperature,
            icoh=int(lthr != 0),
            natom=natom,
            emax=emax,
            suffix=suffix,
            za1=base_za,
            za2=0,
            za3=0,
            elas_mat=resolved_mte,
            ielas=lthr - 1 if lthr else 0,
            nbin=nbin,
            tol=tol,
            mti=resolved_mti,
            mte=resolved_mte,
        )
