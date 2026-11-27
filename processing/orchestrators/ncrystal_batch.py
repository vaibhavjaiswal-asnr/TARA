"""
Batch orchestrator for high-throughput NCrystal ENDF generation.
"""
import logging
import csv
from fractions import Fraction
from math import lcm
import json
import warnings

import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Dict, Any
from parallel_execution import run_parallel

from common.validation import require_existing_file
from processing.backends.ncrystal_processor import NcrystalProcessor
from input_output.ncrystal.liquids import (load_liquid_ncmat, parse_custom_liquids, replace_vdos)
from input_output.ncrystal.material_adapter import fold_solidlike_vdos
from processing.validators.endf_validator import EndfValidator
from input_output.ncrystal.writer import NCrystalWriter  # Assuming this maps to your I/O layer
from core.data.input_pdos import PhononSpectra

logger = logging.getLogger(__name__)


class NcrystalBatchOrchestrator:
    """
    Manages the parallel conversion of a TARA ensemble into ENDF tapes via ncmat2endf.
    """

    def __init__(self, n_jobs: int = -1):
        """Initialize the NCrystal batch runner and its ENDF validator.

        Parameters
        ----------
        n_jobs : int, default=-1
            Requested worker count.
        """
        self.n_jobs = n_jobs
        self.validator = EndfValidator()

    def generate_ensemble_sab_tapes(
        self,
        ensemble,
        output_dir: str,
        base_ncmat_file: str,
        temperature: float,
        atom_label: str,
        elastic_mode: str = "mixed",
        *,
        include_reference: bool = True,
        liquid_options: dict | None = None,
    ) -> Dict[str, str]:
        """Convert a TARA ensemble to validated ENDF thermal-scattering tapes.

        When requested, ``ensemble.reference`` is saved as ``sample_000``;
        stochastic ``ensemble[i]`` is saved as ``sample_{i+1:03d}``. Each
        task owns its own NCrystal processor and temporary NCMAT, so tasks
        may execute in independent joblib worker processes.

        Use ``NcrystalBatchOrchestrator(n_jobs=1)`` for sequential execution,
        ``n_jobs=2`` for two workers, or ``n_jobs=-1`` for the
        HPC-aware reserved default. Requests above the allocation are capped. Failed tapes are never silently omitted: the batch
        raises ``RuntimeError`` with the affected sample IDs.

        Parameters
        ----------
        ensemble : SpectraEnsemble
            Generated stochastic spectra with a separate reference spectrum.
        output_dir : str
            Destination for ``sample_NNN_sab.endf`` files.
        base_ncmat_file : str
            Template NCMAT material into which each spectrum is injected.
        temperature : float
            NCrystal evaluation temperature in kelvin.
        atom_label : str
            Element label to inject (e.g., ``"C"`` for graphite).
        elastic_mode : str, optional
            Forwarded to NCrystal's ENDF generator. Defaults to ``"mixed"``.
        include_reference : bool, optional
            Schedule the unperturbed reference as ``sample_000``. Defaults to
            True. When False, stochastic numbering still starts at one.
        liquid_options : dict, optional
            Optional liquid export overrides. TARA normally builds the capture
            adapter in a temporary directory and infers ``natom`` from NCMAT.
            Overrides include ``capture_library``, ``natom``, ``vdoslux``,
            ``epsilon_e_over_kt``, ``emax_eV``, ``effective_temperature_K``,
            ``timeout``, and ``force``.

        Returns
        -------
        dict of str to str
            Sample-ID-to-ENDF-path mapping for all scheduled samples.
        """
        if include_reference and ensemble.reference is None:
            raise ValueError("The ensemble has no reference spectrum (sample_000).")
        require_existing_file(base_ncmat_file, message=f"NCrystal template not found: {base_ncmat_file}")
        if not atom_label:
            raise ValueError("atom_label must identify the NCrystal target element.")
        if temperature <= 0:
            raise ValueError("temperature must be positive (kelvin).")
        if "@CUSTOM_LIQUIDS" in Path(base_ncmat_file).read_text(encoding="utf-8"):
            return self._generate_liquid_ensemble_sab_tapes(
                ensemble=ensemble,
                output_dir=output_dir,
                base_ncmat_file=base_ncmat_file,
                temperature=temperature,
                atom_label=atom_label,
                include_reference=include_reference,
                liquid_options=liquid_options or {},
            )


        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        results = run_parallel(
            self._generate_tape_from_template,
            self._template_generation_calls(
                ensemble=ensemble,
                output_dir=out_dir,
                base_ncmat_file=base_ncmat_file,
                temperature=temperature,
                atom_label=atom_label,
                elastic_mode=elastic_mode,
                include_reference=include_reference,
            ),
            n_jobs=self.n_jobs,
        )

        self._raise_invalid_solid_batch(results, ensemble, include_reference)

        return {sid: path for sid, path, _ in results}

    @staticmethod
    def _raise_invalid_solid_batch(results, ensemble, include_reference):
        """Report the count and identities of failed solid ENDF jobs.

        Parameters
        ----------
        results : sequence of tuple
            Sample identifiers, output paths, and failure messages.
        ensemble : SpectraEnsemble
            Original ensemble used to determine the expected sample count.
        include_reference : bool
            Whether the reference was scheduled.
        """
        failures = [(sid, error) for sid, path, error in results if path is None]
        if failures:
            details = "; ".join(f"{sid}: {error}" for sid, error in failures[:10])
            remainder = len(failures) - min(10, len(failures))
            if remainder:
                details += f"; ... and {remainder} additional failures"
            n_expected = len(ensemble) + int(include_reference)
            raise RuntimeError(
                f"NCrystal batch failed for {len(failures)} of {n_expected} "
                f"spectra: {details}"
            )

    def _generate_liquid_ensemble_sab_tapes(
        self,
        *,
        ensemble,
        output_dir: str,
        base_ncmat_file: str,
        temperature: float,
        atom_label: str,
        include_reference: bool,
        liquid_options: Dict[str, Any],
    ) -> Dict[str, str]:
        """Dispatch parallel liquid kernel captures and MF7 ENDF exports.

        Parameters
        ----------
        ensemble : SpectraEnsemble
            Generated stochastic spectra and optional reference spectrum.
        output_dir : str
            Target directory for ``sample_NNN_sab.endf`` files.
        base_ncmat_file : str
            Liquid NCMAT template containing ``@CUSTOM_LIQUIDS``.
        temperature : float
            Evaluation temperature in kelvin.
        atom_label : str
            Target liquid species label (e.g., ``"H"``).
        include_reference : bool
            Whether to include ``sample_000``.
        liquid_options : dict of str to Any
            Optional overrides for capture and MF7 export.

        Returns
        -------
        dict of str to str
            Mapping of sample identifiers to validated ENDF file paths.

        Raises
        ------
        ValueError
            If the options or base material composition are invalid.
        RuntimeError
            If any sample fails capture or ENDF verification.
        """
        from processing.backends.ncrystal_liquid_endf import generate_liquid_tape

        from processing.backends.liquid_capture.build import build_capture_library

        options = self._prepare_liquid_options(liquid_options, base_ncmat_file, atom_label)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base_calls = self._template_generation_calls(
            ensemble=ensemble,
            output_dir=out_dir,
            base_ncmat_file=base_ncmat_file,
            temperature=temperature,
            atom_label=atom_label,
            elastic_mode="none",
            include_reference=include_reference,
        )

        if options.get("capture_library") is not None:
            results = run_parallel(generate_liquid_tape, self._liquid_capture_calls(base_calls, options), n_jobs=self.n_jobs)
        else:
            with tempfile.TemporaryDirectory(prefix="tara_liquid_capture_") as build_dir:
                options["capture_library"] = str(build_capture_library(build_dir))
                results = run_parallel(generate_liquid_tape, self._liquid_capture_calls(base_calls, options), n_jobs=self.n_jobs)
        failures = [(sid, error) for sid, path, error in results if path is None]
        if failures:
            raise RuntimeError(f"Liquid MF7 generation failed: {failures}")
        return {sid: path for sid, path, _ in results}


    @staticmethod
    def _liquid_capture_calls(base_calls, options):
        """Apply liquid capture options to each template job.

        Parameters
        ----------
        base_calls : iterable of tuple
            Positional and keyword arguments for template jobs.
        options : dict
            Shared options for liquid kernel capture and export.

        Yields
        ------
        tuple
            Positional arguments and the updated liquid export keywords.
        """
        for args, kwargs in base_calls:
            kwargs.pop("elastic_mode", None)
            kwargs.update(options)
            yield args, kwargs

    def _prepare_liquid_options(self, liquid_options, base_ncmat_file, atom_label):
        """Validate caller options and infer the principal scatterer count if needed.

        Parameters
        ----------
        liquid_options : dict
            Optional overrides for capture and export.
        base_ncmat_file : str
            Liquid NCMAT template.
        atom_label : str
            Principal species label.

        Returns
        -------
        dict
            Job options containing a principal scatterer count.
        """
        options = dict(liquid_options)
        allowed = {"capture_library", "epsilon_e_over_kt", "emax_eV",
                   "effective_temperature_K", "natom", "vdoslux", "timeout", "force"}
        if options.keys() - allowed:
            raise ValueError(f"Unknown liquid options: {sorted(options.keys() - allowed)}")
        if "natom" not in options:
            options["natom"] = self._liquid_principal_atom_count(base_ncmat_file, atom_label)
        return options

    @staticmethod
    def _liquid_principal_atom_count(base_ncmat_file: str, atom_label: str) -> int:
        """Derive the target count in the smallest integer composition.

        Parameters
        ----------
        base_ncmat_file : str
            Liquid NCMAT template containing the original DYNINFO fractions.
        atom_label : str
            Target species, for example H in water.

        Returns
        -------
        int
            Target count in the reduced formula unit.

        Raises
        ------
        ValueError
            If these fractions do not describe a small integer composition.
        """
        from input_output.ncrystal.core import NCrystalFile

        fractions = {}
        for block in NCrystalFile.from_file(str(base_ncmat_file)).dyninfo:
            label = block["element"]
            if label in fractions:
                raise ValueError(f"Repeated NCMAT species label: {label!r}")
            value = Fraction(block["fraction"])
            ratio = value.limit_denominator(1000)
            if ratio <= 0 or abs(value - ratio) > Fraction(1, 1_000_000):
                raise ValueError("Cannot infer natom from NCMAT fractions; supply natom explicitly.")
            fractions[label] = ratio
        if atom_label not in fractions or sum(fractions.values()) != 1:
            raise ValueError("Cannot infer natom from NCMAT fractions; supply natom explicitly.")
        denominator = lcm(*(ratio.denominator for ratio in fractions.values()))
        if denominator > 1000:
            raise ValueError("Cannot infer a small formula unit; supply natom explicitly.")
        return int(fractions[atom_label] * denominator)

    @staticmethod
    def _template_generation_calls(
        *, ensemble, output_dir: Path, base_ncmat_file: str,
        temperature: float, atom_label: str, elastic_mode: str,
        include_reference: bool = True,
    ):
        """Yield ordered, independent NCrystal sample call specifications.

        Parameters
        ----------
        ensemble : SpectraEnsemble
            Source spectra and separate reference.
        output_dir : pathlib.Path
            Target ENDF directory.
        base_ncmat_file : str
            Template material path.
        temperature : float
            Processing temperature in kelvin.
        atom_label : str
            NCrystal target atom label.
        elastic_mode : str
            Elastic scattering mode.
        include_reference : bool, optional
            Yield a reference task before the stochastic tasks.

        Yields
        ------
        tuple
            Worker positional arguments and keyword arguments.
        """
        if include_reference:
            yield NcrystalBatchOrchestrator._template_call(
                "sample_000", 0, ensemble.reference, output_dir,
                base_ncmat_file, temperature, atom_label, elastic_mode,
            )
        for index in range(len(ensemble)):
            yield NcrystalBatchOrchestrator._template_call(
                f"sample_{index + 1:03d}", index + 1, ensemble[index],
                output_dir, base_ncmat_file, temperature, atom_label, elastic_mode,
            )

    @staticmethod
    def _template_call(
        sample_id: str, sample_index: int, spectrum: PhononSpectra,
        output_dir: Path, base_ncmat_file: str, temperature: float,
        atom_label: str, elastic_mode: str,
    ):
        """Construct one isolated NCrystal worker invocation.

        Parameters
        ----------
        sample_id : str
            Stable sample name.
        sample_index : int
            One-based generated output number, including the reference at zero.
        spectrum : PhononSpectra
            Spectrum to process.
        output_dir : pathlib.Path
            ENDF output directory.
        base_ncmat_file : str
            NCMAT template path.
        temperature : float
            Processing temperature.
        atom_label : str
            Target atom label.
        elastic_mode : str
            Elastic scattering mode.

        Returns
        -------
        tuple
            Argument pair for :func:`parallel_execution.run_parallel`.
        """
        return ((), {
            "sample_id": sample_id, "sample_index": sample_index,
            "spectrum": spectrum,
            "output_path": str(output_dir / f"{sample_id}_sab.endf"),
            "base_ncmat_file": str(base_ncmat_file),
            "temperature": temperature, "atom_label": atom_label,
            "elastic_mode": elastic_mode,
        })

    @staticmethod
    def _generate_tape_from_template(
        sample_id: str,
        sample_index: int,
        spectrum: PhononSpectra,
        output_path: str,
        base_ncmat_file: str,
        temperature: float,
        atom_label: str,
        elastic_mode: str,
    ):
        """Generate one tape, isolated from the other parallel workers.

        Parameters
        ----------
        sample_id : str
            Stable sample identifier.
        sample_index : int
            Numeric index of the sample.
        spectrum : PhononSpectra
            Phonon spectrum to process.
        output_path : str
            File path of generated output.
        base_ncmat_file : str
            Input NCMAT material template.
        temperature : float
            Material or processing temperature in kelvin.
        atom_label : str
            Target NCrystal element or isotope label.
        elastic_mode : str
            Configured elastic scattering mode.

        Returns
        -------
        object
            Worker result tuple with sample ID, output path, and status.
        """
        processor = None
        state = None
        try:
            processor = NcrystalProcessor(
                base_ncmat_file=base_ncmat_file,
                temperature=temperature,
            )
            state = processor.create_analysis_state(
                sample_index=sample_index,
                spectrum=spectrum,
                atom_label=atom_label,
            )
            processor.generate_sab_tape(
                ncmat_filepath=state.filepath,
                temperature=temperature,
                material_name=f"tara_{sample_id}",
                output_endf_path=output_path,
                elastic_mode=elastic_mode,
            )
            # A new validator per worker avoids sharing parser state.
            if not EndfValidator().validate_sab_tape(file_path=output_path):
                raise ValueError(f"Generated ENDF is missing valid MF7/MT4: {output_path}")
            return sample_id, output_path, None
        except Exception as exc:
            # Do not leave a stale/partial tape masquerading as a valid result.
            Path(output_path).unlink(missing_ok=True)
            return sample_id, None, f"{type(exc).__name__}: {exc}"
        finally:
            if state is not None:
                processor.cleanup_analysis_file(filepath=state.filepath)

    def process_ensemble(
        self,
        spectra_ensemble: Dict[str, PhononSpectra],
        output_dir: str,
        config: Dict[str, Any]
    ) -> Dict[str, str]:
        """Converts Python PhononSpectra objects into ENDF MF7 tapes.

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

        logger.info(f"Starting NCrystal processing for {len(spectra_ensemble)} samples...")

        results = run_parallel(
            self._run_single_ncrystal,
            (
                ((sample_id, spectra, config, str(out_dir_path / f"{sample_id}_sab.endf")), {})
                for sample_id, spectra in spectra_ensemble.items()
            ),
            n_jobs=self.n_jobs,
        )

        successful_tapes = {sid: path for sid, path in results if path is not None}
        logger.info(
            f"NCrystal Processing Complete: {len(successful_tapes)}/{len(spectra_ensemble)} "
            f"valid ENDF tapes generated."
        )
        return successful_tapes

    def _run_single_ncrystal(
        self,
        sample_id: str,
        spectra: PhononSpectra,
        config: dict,
        out_path: str
    ):
        """Generate and validate one legacy NCrystal ENDF tape.

        Parameters
        ----------
        sample_id : str
            Stable sample identifier.
        spectra : PhononSpectra
            Sampled phonon spectrum.
        config : dict
            NCrystal writer and execution settings.
        out_path : str
            Output path for the ENDF tape.

        Returns
        -------
        tuple
            Sample identifier and tape path, or ``None`` on failure.
        """
        processor = NcrystalProcessor()

        # 1. Create a temporary path for the NCMAT file
        with tempfile.NamedTemporaryFile(suffix=".ncmat", delete=False) as tmp_ncmat:
            tmp_ncmat_path = tmp_ncmat.name

        try:
            # 2. Write the NCMAT file to disk using the correct NCrystalWriter API
            writer = NCrystalWriter(spectra)
            writer.write(tmp_ncmat_path)

            # 3. Execute the ncmat2endf processor
            material_name = f"{config.get('material_prefix', 'Material')}-{sample_id}"

            processor.generate_sab_tape(
                ncmat_filepath=tmp_ncmat_path,
                temperature=config['temperature'],
                material_name=material_name,
                elastic_mode=config.get('elastic_mode', 'mixed'),
                output_endf_path=out_path
            )

            # 4. Validate the resulting ENDF tape for MF7 compliance
            if self.validator.validate_sab_tape(out_path):
                return sample_id, out_path
            else:
                logger.error(f"Sample {sample_id} failed ENDF MF7 validation.")

        except Exception as e:
            logger.error(f"Sample {sample_id} NCrystal execution failed: {str(e)}")

        finally:
            # 5. Clean up the temporary NCMAT file
            if os.path.exists(tmp_ncmat_path):
                os.remove(tmp_ncmat_path)

        return sample_id, None

    def generate_native_ensemble(
        self, ensemble, output_dir: str, base_ncmat_file: str,
        temperature: float, atom_label: str, incident_energies_ev, *,
        sample_energy_ev: float | None = None, n_events: int = 0,
        verify_plugin: bool = True, physical_model=None,
        vdos_moment_policy: str = "warn", max_moment_change: float = 0.10,
        max_shape_l1_change: float = 0.10,
    ) -> Dict[str, Dict[str, str]]:
        """Produce NCrystal-native liquid outputs for a complete ensemble.

        Preserve the reference as ``sample_000`` and keep liquid translation
        parameters fixed in the template. The physical model, when supplied,
        adds fixed oscillator modes to each sampled continuous spectrum.

        Parameters
        ----------
        ensemble : SpectraEnsemble
            Reference spectrum and stochastic solid-like spectra.
        output_dir : str
            Destination for NCMAT, CSV, event, and manifest files.
        base_ncmat_file : str
            Liquid NCMAT template containing ``@CUSTOM_LIQUIDS``.
        temperature : float
            NCrystal evaluation temperature in kelvin.
        atom_label : str
            Target species whose VDOS is replaced.
        incident_energies_ev : array_like
            Incident energies at which to calculate scattering cross sections.
        sample_energy_ev : float, optional
            Incident energy to use for native final-state sampling.
        n_events : int, default=0
            Number of native final states per ensemble member.
        verify_plugin : bool, default=True
            Whether to verify registration of the compiled liquid plugin.
        physical_model : object, optional
            Physical liquid model supplying fixed oscillator and mode weights.
        vdos_moment_policy : {'warn', 'raise'}, default='warn'
            Reaction to excessive displacement-moment distortion.
        max_moment_change : float, default=0.10
            Maximum allowed relative displacement-moment change.
        max_shape_l1_change : float, default=0.10
            Maximum allowed spectral-shape distortion.

        Returns
        -------
        dict of str to dict of str to str
            Per-sample NCMAT, cross-section and optional event file paths.
        """
        energies = self._validate_native_options(
            ensemble, atom_label, incident_energies_ev, sample_energy_ev,
            n_events, vdos_moment_policy, max_moment_change, max_shape_l1_change,
        )
        source = load_liquid_ncmat(base_ncmat_file)
        self._validate_native_material(source, physical_model, atom_label, temperature)
        materials, diagnostics = self._native_materials(
            ensemble, source, atom_label, physical_model, temperature,
            vdos_moment_policy, max_moment_change, max_shape_l1_change,
        )
        if verify_plugin:
            from processing.backends.ncrystal_processor import verify_liquids_plugin
            verify_liquids_plugin()
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        outputs = {
            sid: self._write_native_sample(
                target, sid, material, temperature, energies,
                sample_energy_ev, n_events,
            )
            for sid, material in materials.items()
        }
        self._write_native_manifest(
            target, outputs, diagnostics, temperature, atom_label,
            verify_plugin, vdos_moment_policy, max_moment_change,
            max_shape_l1_change,
        )
        return outputs

    @staticmethod
    def _validate_native_options(
        ensemble, atom_label, incident_energies_ev, sample_energy_ev,
        n_events, vdos_moment_policy, max_moment_change, max_shape_l1_change,
    ) -> np.ndarray:
        """Validate the native liquid invocation before touching output files.

        Parameters
        ----------
        ensemble : SpectraEnsemble
            Ensemble expected to contain a reference spectrum.
        atom_label : str
            Nonempty target atom label.
        incident_energies_ev : array_like
            Requested incident energy grid.
        sample_energy_ev : float or None
            Incident energy for event generation, if enabled.
        n_events : int
            Requested number of events.
        vdos_moment_policy : str
            Displacement-moment validation policy.
        max_moment_change, max_shape_l1_change : float
            Maximum permitted VDOS representation changes.

        Returns
        -------
        numpy.ndarray
            Validated incident energy grid in eV.
        """
        if vdos_moment_policy not in ("warn", "raise"):
            raise ValueError("vdos_moment_policy must be 'warn' or 'raise'.")
        if not np.isfinite(max_moment_change) or not 0 <= max_moment_change < 1:
            raise ValueError("max_moment_change must be a finite fraction in [0,1).")
        if not np.isfinite(max_shape_l1_change) or not 0 <= max_shape_l1_change < 1:
            raise ValueError("max_shape_l1_change must be a finite fraction in [0,1).")
        if getattr(ensemble, "reference", None) is None:
            raise ValueError("Native NCrystal requires sample_000 reference.")
        if not isinstance(atom_label, str) or not atom_label.strip():
            raise ValueError("A nonempty atom_label is required.")
        if type(n_events) is not int or n_events < 0:
            raise ValueError("n_events must be a nonnegative integer.")
        if n_events and sample_energy_ev is None:
            raise ValueError("sample_energy_ev is required when n_events > 0.")
        energies = np.asarray(incident_energies_ev, dtype=float)
        if (energies.ndim != 1 or energies.size < 2 or not np.isfinite(energies).all()
                or np.any(energies <= 0)):
            raise ValueError("Provide at least two positive finite incident energies.")
        return energies

    @staticmethod
    def _validate_native_material(source, physical_model, atom_label, temperature) -> None:
        """Verify template mode weights against the optional physical model.

        Parameters
        ----------
        source : str
            Validated liquid NCMAT text.
        physical_model : object or None
            Optional liquid model with fixed translation and oscillator weights.
        atom_label : str
            Principal species label.
        temperature : float
            Target evaluation temperature in kelvin.
        """
        if physical_model is None:
            return
        if physical_model.principal.label != atom_label:
            raise ValueError("physical_model principal species must match atom_label.")
        if not np.isclose(physical_model.temperature_k, temperature, atol=1e-6):
            raise ValueError("physical_model temperature must match requested temperature.")
        settings = {entry.atom: entry for entry in parse_custom_liquids(source)}
        if atom_label not in settings:
            raise ValueError("Selected principal species has no @CUSTOM_LIQUIDS entry.")
        expected_translation = physical_model.dynamics.translation_weight
        expected_solidlike = (
            physical_model.dynamics.continuous_weight
            + sum(osc.weight for osc in physical_model.dynamics.oscillators)
        )
        if (not np.isclose(settings[atom_label].wt, expected_translation, atol=2e-5)
                or not np.isclose(settings[atom_label].ws, expected_solidlike, atol=2e-5)):
            raise ValueError("NCMAT liquid weights do not match the supplied physical model.")

    @staticmethod
    def _native_spectra(ensemble):
        """Return reference-first spectra with stable sample identifiers.

        Parameters
        ----------
        ensemble : SpectraEnsemble
            Reference spectrum and stochastic samples.

        Returns
        -------
        list of tuple
            ``(sample_id, spectrum)`` pairs.
        """
        spectra = [("sample_000", ensemble.reference)]
        spectra.extend((f"sample_{i + 1:03d}", ensemble[i]) for i in range(len(ensemble)))
        return spectra

    @staticmethod
    def _validate_native_moments(
        sid, detail, vdos_moment_policy, max_moment_change, max_shape_l1_change,
    ) -> None:
        """Apply configured representation-distortion limits to one sample.

        Parameters
        ----------
        sid : str
            Identifier of the sample being checked.
        detail : object
            Diagnostics returned by the VDOS material adapter.
        vdos_moment_policy : str
            Reaction to a displacement-moment discrepancy.
        max_moment_change, max_shape_l1_change : float
            Allowed relative VDOS representation changes.
        """
        if detail.shape_l1_change > max_shape_l1_change:
            raise ValueError(
                f"{sid}: regular VDOS representation changes the spectral "
                f"shape by {detail.shape_l1_change:.2%} "
                f"(limit {max_shape_l1_change:.2%}). "
                "Refuse excessive model distortion."
            )
        delta = abs(detail.displacement_moment_relative_change)
        if delta > max_moment_change:
            msg = (
                f"{sid}: NCrystal regular VDOS representation changes the "
                f"low-energy displacement moment by {delta:.1%} "
                f"(limit {max_moment_change:.1%}). Native scattering may "
                "run, but the liquid model is not quantitatively validated."
            )
            if vdos_moment_policy == "raise":
                raise ValueError(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=5)

    @classmethod
    def _native_vdos(
        cls, sid, spectrum, physical_model, temperature,
        vdos_moment_policy, max_moment_change, max_shape_l1_change,
    ):
        """Represent one sampled spectrum as a NCrystal-compatible VDOS.

        Parameters
        ----------
        sid : str
            Sample identifier for error messages.
        spectrum : PhononSpectra
            Continuous sampled spectrum.
        physical_model : object or None
            Optional liquid mode decomposition.
        temperature : float
            Processing temperature in kelvin.
        vdos_moment_policy : str
            Warning/error policy for moment changes.
        max_moment_change, max_shape_l1_change : float
            Representation tolerances.

        Returns
        -------
        tuple
            Energy grid, density values, and optional diagnostic dictionary.
        """
        if physical_model is None:
            grid = np.asarray(spectrum.energy_grid, dtype=float)
            density = np.asarray(spectrum.density_values, dtype=float)
            # Preserve input values; no implicit filtering or normalization.
            if (len(grid) < 3 or not np.isclose(np.trapezoid(density, grid), 1,
                                                atol=1e-5)):
                raise ValueError("Solid-like VDOS must be unit-area and >=3 points.")
            return grid, density, None
        grid, density, detail = fold_solidlike_vdos(
            physical_model, spectrum, return_diagnostics=True,
            temperature_k=temperature,
        )
        cls._validate_native_moments(
            sid, detail, vdos_moment_policy, max_moment_change, max_shape_l1_change,
        )
        return grid, density, detail.as_dict()

    @classmethod
    def _native_materials(
        cls, ensemble, source, atom_label, physical_model, temperature,
        vdos_moment_policy, max_moment_change, max_shape_l1_change,
    ):
        """Prepare all liquid NCMAT documents before creating output folders.

        Parameters
        ----------
        ensemble : SpectraEnsemble
            Reference-first input ensemble.
        source : str
            Liquid NCMAT template text.
        atom_label : str
            Target atom label.
        physical_model : object or None
            Fixed liquid physics model.
        temperature : float
            Scattering temperature in kelvin.
        vdos_moment_policy : str
            Displacement-moment policy.
        max_moment_change, max_shape_l1_change : float
            Maximum VDOS representation changes.

        Returns
        -------
        tuple of dict
            Material texts and per-sample representation diagnostics.
        """
        materials = {}
        diagnostics = {}
        for sid, spectrum in cls._native_spectra(ensemble):
            grid, density, detail = cls._native_vdos(
                sid, spectrum, physical_model, temperature,
                vdos_moment_policy, max_moment_change, max_shape_l1_change,
            )
            if detail is not None:
                diagnostics[sid] = detail
            # Validate NCMAT before launching the plugin or creating outputs.
            materials[sid] = replace_vdos(source, atom_label, grid, density)
        return materials, diagnostics

    @staticmethod
    def _write_native_sample(
        target, sid, material, temperature, energies, sample_energy_ev, n_events,
    ) -> Dict[str, str]:
        """Generate files for one native liquid sample, cleaning partial outputs.

        Parameters
        ----------
        target : pathlib.Path
            Output directory.
        sid : str
            Sample identifier.
        material : str
            Fully constructed NCMAT text.
        temperature : float
            Native NCrystal evaluation temperature in kelvin.
        energies : numpy.ndarray
            Incident energies for cross-section evaluation.
        sample_energy_ev : float or None
            Incident energy for optional final-state events.
        n_events : int
            Number of requested final states.

        Returns
        -------
        dict of str to str
            Paths of material, cross-section, and optional event files.
        """
        ncmat_path = target / f"{sid}.ncmat"
        xs_path = target / f"{sid}_cross_sections.csv"
        event_path = target / f"{sid}_events.npz"
        try:
            ncmat_path.write_text(material, encoding="utf-8")
            processor = NcrystalProcessor()
            state = processor.create_native_scattering_state(
                str(ncmat_path), temperature, verify_plugin=False,
            )
            xs = processor.native_cross_sections(state, energies)
            with xs_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("energy_ev", "scattering_xs_barn"))
                writer.writerows(zip(energies, xs))
            paths = {"ncmat": str(ncmat_path), "cross_sections": str(xs_path)}
            if n_events:
                eout, mu = processor.native_final_states(
                    state, sample_energy_ev, n_events,
                )
                np.savez_compressed(
                    event_path, ein_ev=float(sample_energy_ev), eout_ev=eout, mu=mu,
                )
                paths["events"] = str(event_path)
            return paths
        except Exception:
            for path in (ncmat_path, xs_path, event_path):
                path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _write_native_manifest(
        target, outputs, diagnostics, temperature, atom_label, verify_plugin,
        vdos_moment_policy, max_moment_change, max_shape_l1_change,
    ) -> None:
        """Save native liquid run provenance without changing its schema.

        Parameters
        ----------
        target : pathlib.Path
            Directory containing the sample files.
        outputs : dict
            Paths generated for each sample.
        diagnostics : dict
            Representation diagnostics by sample identifier.
        temperature : float
            Evaluation temperature in kelvin.
        atom_label : str
            Target NCrystal atom label.
        verify_plugin : bool
            Whether plugin registration was checked.
        vdos_moment_policy : str
            Moment-check failure policy.
        max_moment_change, max_shape_l1_change : float
            Reported representation tolerances.
        """
        (target / "manifest.json").write_text(json.dumps({
            "backend": "ncrystal", "material_state": "liquid", "output": "native",
            "temperature_k": temperature, "atom_label": atom_label,
            "plugin_verified": verify_plugin, "ace_export": "unsupported",
            "sample_ids": list(outputs), "files": outputs,
            "vdos_representation_diagnostics": diagnostics,
            "vdos_moment_policy": vdos_moment_policy,
            "max_moment_change": max_moment_change,
            "max_shape_l1_change": max_shape_l1_change,
            "quantitative_liquid_validation": False,
        }, indent=2) + "\n", encoding="utf-8")
