"""
Reusable execution backend for NCrystal.

The processor owns NCrystal-specific material preparation, temporary NCMAT
lifecycle management, and persistent ENDF generation. Downstream domains such
as ``analyzer`` may compose these low-level capabilities without duplicating
NCrystal API or file-management logic.
"""

from dataclasses import dataclass
import os
import math
import subprocess
from pathlib import Path
import shutil
import tempfile
from typing import Any, Collection, Dict, Optional, Tuple, Union
import uuid

import numpy as np
import NCrystal as NC
from NCrystal import ncmat2endf
from NCrystal._ncmat2endf_impl import NuclearData
from scipy.constants import physical_constants

from common.validation import (
    require_finite_array,
    require_finite_positive,
    require_ndim,
    require_nonnegative_array,
    require_same_shape,
    require_strictly_increasing,
)
from core.data.input_pdos import PhononSpectra
from input_output.ncrystal.liquids import load_liquid_ncmat
from processing.backends.ncmat_isolation import isolate_ncmat_dyninfo, normalize_active_species


MIN_NCRYSTAL_VDOS_ENERGY_EV = 1.0e-5


@dataclass
class NCrystalAnalysisState:
    """Loaded transient NCrystal state for downstream analysis.

    Parameters
    ----------
    filepath : str
        Temporary NCMAT file backing the loaded NCrystal state.
    target_lbl : str
        Atom label selected as the primary analysis target.
    mat : Any
        Loaded NCrystal material object.
    nuc_data : Any
        Loaded NCrystal ``NuclearData`` object.
    mass_amu : float
        Target atomic mass in atomic mass units.
    energy_grid : numpy.ndarray
        Original full Core VDOS energy grid in eV.
    pdos : numpy.ndarray
        Original Core VDOS density values.
    backend_energy_grid, backend_pdos : numpy.ndarray or None
        VDOS arrays passed to the NCMAT composer.
    scatter_inelastic, scatter_coherent_elastic, scatter_incoherent_elastic : object or None
        Prepared scattering handles when component loading is requested.
    """

    # Temporary source and loaded analysis target.
    filepath: str
    target_lbl: str
    mat: Any
    nuc_data: Any
    mass_amu: float

    # Core vibrational spectrum.
    energy_grid: np.ndarray
    pdos: np.ndarray

    # Native scattering components.
    scatter_inelastic: Optional[Any] = None
    scatter_coherent_elastic: Optional[Any] = None
    scatter_incoherent_elastic: Optional[Any] = None

    # Spectrum passed to the NCMAT composer.
    backend_energy_grid: Optional[np.ndarray] = None
    backend_pdos: Optional[np.ndarray] = None


def prepare_ncrystal_vdos(
    spectrum: PhononSpectra,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a TARA PDOS into the regular VDOS grid required by NCrystal.

    This is an NCrystal Processing concern, not a notebook or format-conversion
    concern. A TARA PhononSpectra can contain a dense, synthetic quadratic tail
    (2,000 inserted points); stochastic ensembles can retain that tail even
    though ``original_starts_at_zero`` is True, because their spectra are
    reconstructed from a zero-start sample grid. Spatial perturbations can
    also change the spacing of the remaining *physical* energy samples.

    Remove only an identifiable synthetic low-energy prefix, and interpolate
    only if the remaining grid is genuinely nonuniform. Already-valid
    reference grids (including the LEAPR-derived NCMAT grid) are unchanged.
    The returned density is rescaled to preserve its integral over the
    retained physical-energy interval after interpolation.

    Parameters
    ----------
    spectrum : PhononSpectra
        TARA energy grid in eV and its sampled vibrational density.

    Returns
    -------
    energy_grid, density : ndarray
        Finite, nonnegative VDOS on a strictly increasing regular grid with a
        positive origin. The arrays are safe to pass to set_dyninfo_vdos().
    """
    energy_grid = np.asarray(spectrum.energy_grid, dtype=np.float64)
    density = np.asarray(spectrum.density_values, dtype=np.float64)

    _validate_vdos_arrays(energy_grid, density)
    energy_grid, density = _remove_synthetic_debye_prefix(
        energy_grid, density, spectrum.original_starts_at_zero,
    )
    energy_grid, density = _retain_physical_vdos_points(energy_grid, density)
    return _regularize_vdos_grid(energy_grid, density)


def _validate_vdos_arrays(energy_grid, density):
    """Validate the sampled energy and density arrays.

    Parameters
    ----------
    energy_grid, density : numpy.ndarray
        Paired one-dimensional VDOS arrays on an energy grid in eV.
    """
    shape_error = "NCrystal VDOS: energy and density must be paired 1D arrays"
    require_ndim(energy_grid, 1, message=shape_error)
    require_same_shape(density, energy_grid, message=shape_error)
    if energy_grid.size < 3:
        raise ValueError("NCrystal VDOS: at least three energy points are required")
    finite_error = "NCrystal VDOS: nonfinite energy or density values"
    require_finite_array(energy_grid, message=finite_error)
    require_finite_array(density, message=finite_error)
    grid_error = "NCrystal VDOS: energies must be nonnegative and increasing"
    if energy_grid[0] < 0:
        raise ValueError(grid_error)
    require_strictly_increasing(energy_grid, message=grid_error)
    require_nonnegative_array(density, message="NCrystal VDOS: density must be nonnegative")


def _remove_synthetic_debye_prefix(energy_grid, density, original_starts_at_zero):
    """Identify and drop the synthetic low-energy prefix.

    Parameters
    ----------
    energy_grid, density : numpy.ndarray
        Paired one-dimensional VDOS arrays on an energy grid in eV.
    original_starts_at_zero : bool
        Whether the source spectrum began at zero.

    Returns
    -------
    tuple of numpy.ndarray
        Energy grid in eV and the corresponding density.
    """
    # PhononSpectra prepends exactly one origin point and 2,000 quadratic
    # interior points to a positive-origin input. Check the stored flag and
    # the tail's values before discarding this known synthetic region.
    if not original_starts_at_zero:
        n_prefix = 2001
        if energy_grid.size < n_prefix + 3 or energy_grid[0] != 0.0:
            raise ValueError("NCrystal VDOS: unexpected reconstructed Debye-tail layout")
        e_first = energy_grid[n_prefix]
        expected_tail = density[n_prefix] * (energy_grid[:n_prefix] / e_first) ** 2
        if not np.allclose(density[:n_prefix], expected_tail,
                           rtol=1e-8, atol=1e-12):
            raise ValueError("NCrystal VDOS: reconstructed Debye-tail check failed")
        energy_grid = energy_grid[n_prefix:]
        density = density[n_prefix:]
    elif energy_grid[0] == 0.0 and energy_grid.size > 600:
        # TARA ensemble reconstruction often sets original_starts_at_zero=True
        # even if the ensemble was generated from an NCrystal reference with a
        # synthetic tail. Identify the tail by its *spacing discontinuity*,
        # not by its density (amplitude/spatial perturbations modify density).
        diffs = np.diff(energy_grid)
        fine_step = float(np.median(diffs[1:501]))
        coarse_step = float(np.median(diffs[-min(64, diffs.size):]))
        if fine_step > 0 and coarse_step > 30.0 * fine_step:
            # A physical NCrystal grid has much coarser bins than the 2,000
            # synthetic integration points. Locate its first bin after >=500
            # fine intervals. Guard against discarding unrelated input data.
            threshold = max(20.0 * fine_step, 0.20 * coarse_step)
            candidates = np.flatnonzero(diffs[500:] >= threshold) + 500
            if candidates.size == 0:
                raise ValueError(
                    "NCrystal VDOS: dense low-energy prefix detected but its "
                    "boundary could not be identified safely"
                )
            first_native_index = int(candidates[0])
            if first_native_index >= energy_grid.size - 3:
                raise ValueError("NCrystal VDOS: insufficient points beyond Debye prefix")
            energy_grid = energy_grid[first_native_index:]
            density = density[first_native_index:]

    return energy_grid, density


def _retain_physical_vdos_points(energy_grid, density):
    """Keep the energy points accepted by the NCrystal VDOS composer.

    Parameters
    ----------
    energy_grid, density : numpy.ndarray
        Paired one-dimensional VDOS arrays on an energy grid in eV.

    Returns
    -------
    tuple of numpy.ndarray
        Energy grid in eV and the corresponding density.
    """
    # Omit the exact zero point and truly subthreshold physical points. NCrystal
    # extrapolates the Debye region itself from the first retained point.
    retained = energy_grid >= MIN_NCRYSTAL_VDOS_ENERGY_EV
    energy_grid = energy_grid[retained]
    density = density[retained]
    if energy_grid.size < 3:
        raise ValueError("NCrystal VDOS: fewer than three physical points >=1e-5 eV")

    return energy_grid, density


def _regularize_vdos_grid(energy_grid, density):
    """Interpolate a nonuniform grid while preserving its retained integral.

    Parameters
    ----------
    energy_grid, density : numpy.ndarray
        Paired one-dimensional VDOS arrays on an energy grid in eV.

    Returns
    -------
    tuple of numpy.ndarray
        Energy grid in eV and the corresponding density.
    """
    spacing = float((energy_grid[-1] - energy_grid[0]) / (energy_grid.size - 1))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("NCrystal VDOS: invalid energy-grid spacing")

    # TARA's spatial energy perturbations can produce physically meaningful
    # *nonuniform* coordinates. NCMAT VDOS needs a regular grid, so interpolate
    # those spectra here, not in each notebook. Preserve the physical extent
    # and number of input points, and compensate for interpolation quadrature.
    diffs = np.diff(energy_grid)
    if not np.allclose(diffs, spacing, rtol=1e-6, atol=1e-12):
        regular_grid = np.linspace(
            float(energy_grid[0]), float(energy_grid[-1]),
            energy_grid.size, dtype=np.float64,
        )
        regular_density = np.interp(regular_grid, energy_grid, density)
        original_area = float(np.trapezoid(density, energy_grid))
        regular_area = float(np.trapezoid(regular_density, regular_grid))
        if not (np.isfinite(original_area) and np.isfinite(regular_area)
                and original_area > 0.0 and regular_area > 0.0):
            raise ValueError("NCrystal VDOS: cannot normalize regridded PDOS")
        regular_density *= original_area / regular_area
        energy_grid, density = regular_grid, regular_density

    # Do not impose integer E_min/delta_E; NCrystal accepts a positive-origin
    # regular grid with nonintegral E_min/delta_E (the original CH2 NCMAT has
    # precisely such a grid).
    return energy_grid, density


def reconstruct_debye_tail(
    energy_grid: np.ndarray,
    density_values: np.ndarray,
    n_points: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct the quadratic Debye tail below the first NCrystal point.

    Parameters
    ----------
    energy_grid : numpy.ndarray
        NCrystal-compatible VDOS energy grid in eV.
    density_values : numpy.ndarray
        Density values corresponding to ``energy_grid``.
    n_points : int, default=2000
        Number of points added from zero up to the first retained energy,
        excluding that energy.

    Returns
    -------
    full_energy_grid : numpy.ndarray
        Energy grid extended down to zero in eV.
    full_density_values : numpy.ndarray
        Density values including the reconstructed Debye tail.

    Raises
    ------
    ValueError
        If the arrays are not one-dimensional or have different lengths,
        the energy grid is empty or starts at a nonpositive energy, or
        fewer than two tail points are requested.

    Notes
    -----
    The continuation ``rho(E) = rho_1 * (E / E_1)**2`` joins continuously
    to the first retained point ``(E_1, rho_1)``.
    """
    energy_grid = np.asarray(energy_grid, dtype=np.float64)
    density_values = np.asarray(density_values, dtype=np.float64)
    _validate_debye_tail_inputs(energy_grid, density_values, n_points)

    first_energy = float(energy_grid[0])
    first_density = float(density_values[0])

    # The original grid already contains the joining point.
    tail_energy = np.linspace(
        0.0, first_energy, n_points + 1, dtype=np.float64
    )[:-1]
    tail_density = first_density * (tail_energy / first_energy) ** 2

    return (
        np.concatenate((tail_energy, energy_grid)),
        np.concatenate((tail_density, density_values)),
    )


def _validate_debye_tail_inputs(
    energy_grid: np.ndarray,
    density_values: np.ndarray,
    n_points: int,
) -> None:
    """Validate the paired input arrays and requested tail length.

    Parameters
    ----------
    energy_grid : numpy.ndarray
        Source energy grid in eV.
    density_values : numpy.ndarray
        Density values paired with the source energies.
    n_points : int
        Number of points to add below the first source energy.

    Raises
    ------
    ValueError
        If the array dimensions or lengths disagree, the energy grid is
        empty or starts at a nonpositive energy, or ``n_points`` is below two.
    """
    if energy_grid.ndim != 1 or density_values.ndim != 1:
        raise ValueError("energy_grid and density_values must be one-dimensional.")
    if energy_grid.size != density_values.size:
        raise ValueError("energy_grid and density_values must have the same length.")
    if energy_grid.size == 0 or energy_grid[0] <= 0.0:
        raise ValueError("The first NCrystal VDOS energy must be strictly positive.")
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")


class UnsupportedLiquidAceExport(NotImplementedError):
    """Raised if native liquid ACE export is requested without proven fidelity."""


@dataclass
class NCrystalNativeState:
    """Native NCrystal scattering state, with no ENDF intermediate.

    Parameters
    ----------
    cfgstr : str
        Material configuration used for native scattering.
    scatter : object
        Loaded NCrystal scattering handle.
    """

    cfgstr: str
    scatter: Any


def verify_liquids_plugin(*, command: str = "nctool") -> str:
    """Confirm compiled Liquids plugin registration using nctool --plugins.

    Parameters
    ----------
    command : str
        Executable name or path for the NCrystal diagnostic tool.

    Returns
    -------
    str
        Text of the plugin registration report.
    """
    try:
        result = subprocess.run([command, "--plugins"], capture_output=True,
                                text=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Cannot verify plugin; run 'nctool --plugins'.") from exc
    output = result.stdout + result.stderr
    if not any("Liquids" in line for line in output.splitlines()):
        raise RuntimeError("NCrystal Liquids plugin not active; check nctool --plugins.")
    return output


def sampled_kinematics(
    incident_energy_ev: float,
    outgoing_energy_ev,
    cosine_scatter,
    *,
    temperature_k: float,
    target_mass_ratio: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compute momentum and energy transfer for sampled scattering events.

    Parameters
    ----------
    incident_energy_ev : float
        Incident neutron energy in eV.
    outgoing_energy_ev : array_like
        One-dimensional outgoing neutron energies in eV.
    cosine_scatter : array_like
        Scattering-angle cosines with the same shape as the outgoing energies.
    temperature_k : float
        Material temperature in kelvin.
    target_mass_ratio : float, default=1.0
        Ratio of target mass to neutron mass. Choose the target species
        explicitly for a multi-species material.

    Returns
    -------
    dict of str to numpy.ndarray
        Event-wise momentum transfer in inverse angstroms, energy transfer
        in eV, and dimensionless alpha and beta.

    Raises
    ------
    ValueError
        If the scalar conditions or event arrays are invalid.

    Notes
    -----
    Positive energy transfer corresponds to neutron energy loss. These
    event diagnostics can be histogrammed to visualize sampled events.
    """
    _validate_kinematic_conditions(
        incident_energy_ev, temperature_k, target_mass_ratio,
    )
    outgoing = np.asarray(outgoing_energy_ev, dtype=float)
    cosine = np.asarray(cosine_scatter, dtype=float)
    _validate_scattering_events(outgoing, cosine)

    # Clip small cosine overshoots and guard against negative roundoff at Q=0.
    q_energy = np.maximum(
        incident_energy_ev + outgoing
        - 2 * np.sqrt(incident_energy_ev * outgoing) * np.clip(cosine, -1, 1),
        0,
    )
    energy_transfer = incident_energy_ev - outgoing
    kbt = 8.617333262145e-5 * temperature_k

    return {
        "q_inv_angstrom": np.sqrt(q_energy / 0.002072124855),
        "energy_transfer_ev": energy_transfer,
        "alpha": q_energy / (target_mass_ratio * kbt),
        "beta": energy_transfer / kbt,
    }


def _validate_kinematic_conditions(
    incident_energy_ev: float,
    temperature_k: float,
    target_mass_ratio: float,
) -> None:
    """Validate the scalar conditions shared by all sampled events.

    Parameters
    ----------
    incident_energy_ev : float
        Incident neutron energy in eV.
    temperature_k : float
        Material temperature in kelvin.
    target_mass_ratio : float
        Ratio of target mass to neutron mass.

    Raises
    ------
    ValueError
        If any condition is nonfinite or nonpositive.
    """
    conditions = (incident_energy_ev, temperature_k, target_mass_ratio)
    if not all(math.isfinite(value) and value > 0 for value in conditions):
        raise ValueError("Incident energy, temperature, mass ratio must be positive.")


def _validate_scattering_events(
    outgoing: np.ndarray,
    cosine: np.ndarray,
) -> None:
    """Validate the paired arrays of outgoing energies and angle cosines.

    Parameters
    ----------
    outgoing : numpy.ndarray
        One-dimensional outgoing neutron energies in eV.
    cosine : numpy.ndarray
        Scattering-angle cosines paired with the outgoing energies.

    Raises
    ------
    ValueError
        If the arrays have incompatible shapes, contain nonfinite values,
        include negative energies, or contain cosines outside the tolerance.
    """
    if (
        outgoing.ndim != 1
        or cosine.shape != outgoing.shape
        or not np.isfinite(outgoing).all()
        or not np.isfinite(cosine).all()
        or np.any(outgoing < 0)
        or np.any(np.abs(cosine) > 1.000001)
    ):
        raise ValueError("Invalid outgoing energies or scattering angles.")


class NcrystalProcessor:
    """Own reusable NCrystal execution and temporary-file infrastructure.

    The processor supports two consumers without mixing their orchestration:

    * Processing uses :meth:`generate_sab_tape` for persistent ENDF output.
    * Analyzer uses :meth:`create_analysis_state` for transient in-memory
      evaluation and delegates cleanup back through
      :meth:`cleanup_analysis_file`.

    Parameters
    ----------
    base_ncmat_file : str, optional
        Base NCMAT structural template. Required only for transient Analyzer
        evaluations.
    temperature : float, optional
        Material temperature in kelvin. Required with ``base_ncmat_file`` for
        transient Analyzer evaluations.
    temp_dir : str, optional
        Directory for transient NCMAT files. Defaults to the system temporary
        directory.
    active_species : collection of str, optional
        Keep only these NCMAT element labels scattering-active for transient
        analysis. None preserves the full material. Not used for ENDF export.
    """

    def __init__(
        self,
        base_ncmat_file: Optional[str] = None,
        temperature: Optional[float] = None,
        temp_dir: Optional[str] = None,
        active_species: Optional[Collection[str]] = None,
    ) -> None:
        """Configure NCrystal material templates and temporary-resource ownership.

        Parameters
        ----------
        base_ncmat_file : Optional[str], optional
            Path to the NCMAT material template.
        temperature : Optional[float], optional
            Material temperature in kelvin.
        temp_dir : Optional[str], optional
            Directory for temporary execution resources.
        active_species : collection of str, optional
            NCMAT elements retained for transient scattering calculations.

        Raises
        ------
        ValueError
            If the input or required state violates the operation contract.
        """
        self.base_ncmat_file = base_ncmat_file
        self.temperature = temperature
        self.active_species = (None if active_species is None
                               else normalize_active_species(active_species))
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self._base_composer = None

        if (base_ncmat_file is None) != (temperature is None):
            raise ValueError(
                "base_ncmat_file and temperature must be provided together "
                "for transient NCrystal analysis."
            )

        if base_ncmat_file is not None:
            # The legacy ENDF/Analyzer composer is not validated for Liquids
            # plugin convolution. Never silently discard a custom liquid block.
            if Path(base_ncmat_file).is_file() and "@CUSTOM_LIQUIDS" in Path(base_ncmat_file).read_text(encoding="utf-8"):
                raise UnsupportedLiquidAceExport(
                    "Native liquid NCMATs are not supported by solid ENDF composition."
                )
            self._initialize_analysis_template()

    def _initialize_analysis_template(self) -> None:
        """Load and refine the reusable base NCMAT composer.
        """
        reading = f"{self.base_ncmat_file};temp={self.temperature:.1f}K"
        self._base_composer = NC.NCMATComposer.from_cfgstr(reading)
        self._base_composer.refine_crystal_structure(quiet=True)

    def create_analysis_state(
        self,
        sample_index: int,
        spectrum: Union[PhononSpectra, Dict[str, PhononSpectra]],
        atom_label: Optional[str] = None,
        *,
        load_scattering_components: bool = False,
    ) -> NCrystalAnalysisState:
        """Inject spectra and load a transient NCrystal analysis state.

        Parameters
        ----------
        sample_index : int
            Analyzer sample identifier used only for the temporary filename.
        spectrum : PhononSpectra or dict of str to PhononSpectra
            Spectrum or element-labelled spectra to inject into the base NCMAT
            material.
        atom_label : str, optional
            Target element. Required for monoatomic inputs. For dictionaries,
            defaults to the first supplied label.
        load_scattering_components : bool, optional
            Load the three scattering handles required by the Analyzer.

        Returns
        -------
        NCrystalAnalysisState
            Fully loaded in-memory NCrystal objects plus the temporary NCMAT
            path. The caller must later call :meth:`cleanup_analysis_file`.
        """
        if self._base_composer is None:
            raise RuntimeError(
                "Transient NCrystal analysis requires base_ncmat_file and "
                "temperature when constructing NcrystalProcessor."
            )

        unique_id = uuid.uuid4().hex[:8]
        temp_filename = self.temp_dir / f"pdos_{sample_index}_{unique_id}.ncmat"
        composer = self._base_composer.clone()

        try:
            target_lbl, target_energy_grid, target_density = self._inject_spectra(
                composer,
                spectrum,
                atom_label,
            )
            self._write_analysis_material(composer, temp_filename, target_lbl)

            reading = f"{temp_filename};temp={self.temperature:.1f}K"
            mat = NC.load(reading)
            nuc_data = NuclearData(
                ncmat_cfg=reading,
                temperatures=(self.temperature,),
                elastic_mode="mixed",
                requested_emax=5.0,
                verbosity=0,
            )

            neutron_mass_amu = physical_constants["neutron mass in u"][0]
            mass_amu = nuc_data.elements[target_lbl]._awr * neutron_mass_amu

            scatter_inelastic, scatter_coherent_elastic, scatter_incoherent_elastic = (
                self._load_scattering_components(reading) if load_scattering_components
                else (None, None, None)
            )

            target_spectrum = spectrum[target_lbl] if isinstance(spectrum, dict) else spectrum
            return NCrystalAnalysisState(
                filepath=str(temp_filename),
                target_lbl=target_lbl,
                mat=mat,
                nuc_data=nuc_data,
                mass_amu=mass_amu,
                energy_grid=target_spectrum.energy_grid,
                pdos=target_spectrum.density_values,
                backend_energy_grid=target_energy_grid,
                backend_pdos=target_density,
                scatter_inelastic=scatter_inelastic,
                scatter_coherent_elastic=scatter_coherent_elastic,
                scatter_incoherent_elastic=scatter_incoherent_elastic,
            )
        except Exception:
            self.cleanup_analysis_file(str(temp_filename))
            raise

    def _write_analysis_material(self, composer, temp_filename, target_lbl):
        """Write the temporary material and apply optional species isolation.

        Parameters
        ----------
        composer : NCrystal.NCMATComposer
            Material with the requested spectra injected.
        temp_filename : pathlib.Path
            Destination for the isolated material.
        target_lbl : str
            Target NCMAT species label.
        """
        if self.active_species is not None and target_lbl not in self.active_species:
            raise ValueError(
                f"Analysis target {target_lbl!r} must belong to active_species "
                f"{sorted(self.active_species)!r}."
            )
        composer.write(str(temp_filename))
        if self.active_species is not None:
            # Isolation is NCrystal execution configuration, not LEAPR / NCMAT
            # input conversion. Change only the freshly written temp file.
            ncmat_text = temp_filename.read_text(encoding="utf-8")
            isolated = isolate_ncmat_dyninfo(ncmat_text, self.active_species)
            temp_filename.write_text(isolated, encoding="utf-8")
        NC.clearCaches()

    @staticmethod
    def _load_scattering_components(reading):
        """Load the three independent native scattering handles.

        Parameters
        ----------
        reading : str
            NCrystal material configuration including temperature.

        Returns
        -------
        tuple
            Inelastic, coherent elastic, and incoherent elastic handles.
        """
        from NCrystal.misc import MaterialSource

        source = MaterialSource(reading)
        return tuple(
            source.load(f"comp={component}", doInfo=False, doAbsorption=False).scatter
            for component in ("inelas", "coh_elas", "incoh_elas")
        )

    @staticmethod
    def _inject_spectra(
        composer: Any,
        spectrum: Union[PhononSpectra, Dict[str, PhononSpectra]],
        atom_label: Optional[str],
    ) -> Tuple[str, np.ndarray, np.ndarray]:
        """Inject one or more spectra into a cloned NCMAT composer.

        Parameters
        ----------
        composer : Any
            NCMAT composer used for temporary material construction.
        spectrum : Union[PhononSpectra, Dict[str, PhononSpectra]]
            Source phonon spectrum to evaluate or perturb.
        atom_label : Optional[str]
            Material-specific atomic label to replace or select.

        Returns
        -------
        Tuple[str, np.ndarray, np.ndarray]
            Calculated arrays or records in the order described by the operation.

        Raises
        ------
        ValueError
            If the input or required state violates the operation contract.
        """
        if isinstance(spectrum, dict):
            if not spectrum:
                raise ValueError("The supplied spectrum dictionary is empty.")

            target_lbl = atom_label if atom_label else next(iter(spectrum))
            if target_lbl not in spectrum:
                raise ValueError(
                    f"Requested atom_label {target_lbl!r} is not present in "
                    "the supplied spectrum dictionary."
                )

            target_energy_grid = None
            target_density = None

            for label, current_spectrum in spectrum.items():
                energy_grid, density = prepare_ncrystal_vdos(current_spectrum)
                composer.set_dyninfo_vdos(
                    composer.find_label(label),
                    vdos_egrid=energy_grid,
                    vdos=density,
                )
                if label == target_lbl:
                    target_energy_grid = energy_grid
                    target_density = density

            return target_lbl, target_energy_grid, target_density

        if not atom_label:
            raise ValueError(
                "The 'atom_label' kwarg is required for monoatomic injection."
            )

        energy_grid, density = prepare_ncrystal_vdos(spectrum)
        composer.set_dyninfo_vdos(
            composer.find_label(atom_label),
            vdos_egrid=energy_grid,
            vdos=density,
        )
        return atom_label, energy_grid, density

    @staticmethod
    def cleanup_analysis_file(filepath: Optional[str]) -> None:
        """Delete a transient NCMAT file created for Analyzer evaluation.

        Parameters
        ----------
        filepath : Optional[str]
            Path identifying the input or temporary resource.
        """
        if not filepath:
            return
        try:
            os.remove(filepath)
        except FileNotFoundError:
            pass
        except OSError:
            pass

    def generate_sab_tape(
        self,
        ncmat_filepath: str,
        temperature: float,
        material_name: str,
        output_endf_path: str,
        elastic_mode: str = "mixed",
    ) -> None:
        """Generate a persistent ENDF MF7 tape with ``ncmat2endf``.

        This public Processing API is intentionally preserved so existing
        orchestrators and pipelines continue to work unchanged.

        Parameters
        ----------
        ncmat_filepath : str
            Path to the source NCMAT file.
        temperature : float
            Evaluation temperature in kelvin.
        material_name : str
            Internal material identifier passed to ``ncmat2endf``.
        output_endf_path : str
            Permanent output ENDF path.
        elastic_mode : str, default="mixed"
            NCrystal elastic-mode setting.
        """
        ncmat_path = Path(ncmat_filepath)
        # Keep the legacy solid/mock API intact: test doubles may supply a
        # virtual filename. Only inspect the actual file when it exists.
        if ncmat_path.is_file() and "@CUSTOM_LIQUIDS" in ncmat_path.read_text(encoding="utf-8"):
            raise UnsupportedLiquidAceExport(
                "Plugin liquid scattering cannot be exported by the solid ncmat2endf path."
            )
        target_path = Path(output_endf_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        ncmat_cfg = f"{ncmat_filepath};temp={temperature:.1f}K"

        # Scope backend scratch data to a workspace that will be removed after execution.
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                # Delegate thermal-scattering ENDF production to the supported NCrystal
                # conversion interface.
                ncmat2endf.ncmat2endf(
                    ncmat_cfg=ncmat_cfg,
                    material_name=material_name,
                    elastic_mode=elastic_mode,
                    force=True,
                    outdir=tmp_dir,
                    verbosity=0,
                )

                generated_file = Path(tmp_dir) / f"{material_name}.endf"
                if not generated_file.exists():
                    endf_files = list(Path(tmp_dir).glob("*.endf"))
                    if not endf_files:
                        raise FileNotFoundError(
                            "ncmat2endf executed but failed to write the ENDF tape."
                        )
                    generated_file = endf_files[0]

                shutil.copy(str(generated_file), str(target_path))
            except Exception as exc:
                raise RuntimeError(
                    f"NCrystal ENDF generation failed: {str(exc)}"
                ) from exc

    def create_native_scattering_state(
        self, ncmat_filepath: str, temperature: float, *,
        verify_plugin: bool = True, ncrystal: Any = None,
    ) -> "NCrystalNativeState":
        """Load a native plugin material without conversion through ENDF.

        Unlike transient solid ENDF preparation, this method loads the original
        NCMAT directly, preserving its @CUSTOM_LIQUIDS convolution settings.
        The compiled plugin must be active in NCrystal for genuine liquid physics.

        Parameters
        ----------
        ncmat_filepath : str
            NCMAT material file to load.
        temperature : float
            Material or processing temperature in kelvin.
        verify_plugin : bool
            Whether to verify the compiled NCrystal Liquids plugin.
        ncrystal : Any
            Optional NCrystal implementation for testing or injection.

        Returns
        -------
        NCrystalNativeState
            Native scattering state containing the active plugin.
        """
        require_finite_positive(temperature, message="Temperature must be finite and positive.")
        path = Path(ncmat_filepath).expanduser().resolve()
        load_liquid_ncmat(path)
        if verify_plugin:
            verify_liquids_plugin()
        nc = NC if ncrystal is None else ncrystal
        cfg = f"{path};temp={temperature:.12g}K"
        try:
            scatter = nc.createScatter(cfg)
        except Exception as exc:
            raise RuntimeError(
                f"NCrystal could not initialize native liquid NCMAT {path}: "
                f"{type(exc).__name__}: {exc}. Check the @DYNINFO VDOS grid "
                "(E_min/delta_E must be a positive integer), vdos_density "
                "syntax, and installed Liquids plugin."
            ) from exc
        return NCrystalNativeState(cfgstr=cfg, scatter=scatter)

    @staticmethod
    def native_cross_sections(state: "NCrystalNativeState", incident_energies_ev) -> np.ndarray:
        """Return native NCrystal scattering cross sections (barn per atom).

        Parameters
        ----------
        state : NCrystalNativeState
            Loaded native NCrystal scattering state.
        incident_energies_ev : object
            Incident neutron energies in electronvolts.

        Returns
        -------
        np.ndarray
            Total scattering cross sections in barns.
        """
        energies = np.asarray(incident_energies_ev, dtype=float)
        if (energies.ndim != 1 or not len(energies) or not np.isfinite(energies).all()
                or np.any(energies <= 0)):
            raise ValueError("Incident energies must be positive and finite.")
        scatter = state.scatter
        api = next((getattr(scatter, name) for name in
                    ("crossSectionIsotropic", "crossSectionNonOriented", "crossSection")
                    if getattr(scatter, name, None) is not None), None)
        if api is None:
            raise RuntimeError("NCrystal scatter object has no usable cross-section API.")
        values = np.asarray([float(api(float(e))) for e in energies])
        if not np.isfinite(values).all() or np.any(values < 0):
            raise RuntimeError("NCrystal returned invalid cross sections.")
        return values

    @staticmethod
    def native_final_states(state: "NCrystalNativeState", incident_energy_ev: float,
                            n_events: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample outgoing energy (eV) and cosine of neutron scatter angle.

        Parameters
        ----------
        state : NCrystalNativeState
            Loaded native NCrystal scattering state.
        incident_energy_ev : float
            Incident neutron energy in electronvolts.
        n_events : int
            Number of final-state scattering events to sample.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Outgoing energies and scattering-angle cosines.
        """
        require_finite_positive(incident_energy_ev, message="Incident energy must be positive.")
        if type(n_events) is not int or n_events < 1:
            raise ValueError("n_events must be a positive integer.")
        sampler = getattr(state.scatter, "sampleScatterIsotropic", None)
        if sampler is None:
            raise RuntimeError("NCrystal scatter object lacks isotropic sampling.")
        eout, cosine = np.empty(n_events), np.empty(n_events)
        for index in range(n_events):
            event = sampler(float(incident_energy_ev))
            if hasattr(event, "ekin") and hasattr(event, "mu"):
                eo, mu = event.ekin, event.mu
            else:
                try:
                    eo, mu = event
                except (TypeError, ValueError) as exc:
                    raise RuntimeError("Unexpected NCrystal event response.") from exc
            eout[index], cosine[index] = float(eo), float(mu)
        if (not np.isfinite(eout).all() or not np.isfinite(cosine).all()
                or np.any(eout < 0) or np.any(np.abs(cosine) > 1.0000001)):
            raise RuntimeError("NCrystal returned nonphysical final states.")
        return eout, np.clip(cosine, -1, 1)

    @staticmethod
    def export_native_liquid_ace(*args, **kwargs) -> None:
        """Reject unvalidated conversion of the plugin convolution to ACE.

        Parameters
        ----------
        *args : tuple
            Positional arguments supplied by a caller.
        **kwargs : dict
            Keyword arguments supplied by a caller.
        """
        raise UnsupportedLiquidAceExport(
            "NCrystal Liquids plugin-to-ENDF/ACE conversion has not been verified. "
            "Use native NCrystal output, or independently evaluated NJOY LEAPR/ACE."
        )
