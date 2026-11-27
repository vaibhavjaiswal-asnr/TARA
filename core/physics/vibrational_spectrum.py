"""Tabulate vibrational continua and discrete modes on a common grid."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from common.validation import require_finite_nonnegative
from core.data.input_pdos import PhononSpectra
from core.data.material_scattering import DiscreteOscillator


DEFAULT_OSCILLATOR_SIGMA_EV = 0.001
_GAUSSIAN_MARGIN_SIGMAS = 6.0


def _validate_sigma(oscillator_sigma_ev: float) -> float:
    """Validate the Gaussian broadening width in electronvolts.

    Parameters
    ----------
    oscillator_sigma_ev : float
        Standard deviation of the Gaussian mode representation in eV. Zero
        selects two-node linear tent deposition.

    Returns
    -------
    float
        Validated nonnegative width in eV.

    Raises
    ------
    ValueError
        If ``oscillator_sigma_ev`` is nonfinite or negative.
    """
    sigma = float(oscillator_sigma_ev)
    require_finite_nonnegative(sigma, message="oscillator_sigma_ev must be finite and nonnegative.")
    return sigma


def _resolve_uniform_grid(
    source: np.ndarray,
    modes: Sequence[DiscreteOscillator],
    energy_grid: ArrayLike | None,
    oscillator_sigma_ev: float,
) -> np.ndarray:
    """Build or validate the common uniform zero-origin energy grid.

    Parameters
    ----------
    source : numpy.ndarray
        Original continuum energy grid in eV.
    modes : sequence of DiscreteOscillator
        Discrete vibrational modes to include on the grid.
    energy_grid : array_like or None
        Optional caller-supplied uniform energy grid in eV.
    oscillator_sigma_ev : float
        Gaussian standard deviation for discrete oscillators in eV.

    Returns
    -------
    numpy.ndarray
        Validated 1-D uniform energy grid in eV starting at zero.

    Raises
    ------
    ValueError
        If the source grid is nonuniform when ``energy_grid`` is ``None``, or
        if the resolved grid is nonuniform, non-increasing, does not start at
        zero, or fails to cover the continuum support.
    """
    if energy_grid is None:
        grid = _build_uniform_grid(source, modes, oscillator_sigma_ev)
    else:
        grid = np.asarray(energy_grid, dtype=float)
    _validate_uniform_grid(grid, source_end=source[-1])
    return grid


def _build_uniform_grid(
    source: np.ndarray,
    modes: Sequence[DiscreteOscillator],
    oscillator_sigma_ev: float,
) -> np.ndarray:
    """Construct a zero-origin grid that resolves the continuum and modes.

    Parameters
    ----------
    source : numpy.ndarray
        Source continuum energy coordinates in eV.
    modes : sequence of DiscreteOscillator
        Modes whose positive weights require support on the grid.
    oscillator_sigma_ev : float
        Oscillator broadening standard deviation in eV.

    Returns
    -------
    numpy.ndarray
        Uniform energy grid in eV.

    Raises
    ------
    ValueError
        If the source spacing is nonuniform.
    """
    delta = float(source[1] - source[0])
    if not np.allclose(np.diff(source), delta, rtol=1e-8, atol=1e-12):
        raise ValueError(
            "Provide a uniform energy_grid for nonuniform source data."
        )
    active = [m.energy_ev for m in modes if m.weight > 0]
    if active:
        # Resolve each Gaussian and keep mode centers above zero.
        subdiv = int(np.ceil(delta / min(active)))
        if oscillator_sigma_ev > 0.0:
            subdiv = max(
                subdiv, int(np.ceil(3.0 * delta / oscillator_sigma_ev))
            )
        delta /= max(1, subdiv)
    margin = max(2.0 * delta, _GAUSSIAN_MARGIN_SIGMAS * oscillator_sigma_ev) if active else 0.0
    upper = max(float(source[-1]), max(active, default=0.0) + margin)
    return np.arange(int(np.ceil(upper / delta)) + 1, dtype=float) * delta


def _validate_uniform_grid(grid: np.ndarray, source_end: float) -> None:
    """Validate the shared energy grid and its continuum coverage.

    Parameters
    ----------
    grid : numpy.ndarray
        Candidate uniform energy grid in eV.
    source_end : float
        Upper endpoint of the continuum in eV.

    Raises
    ------
    ValueError
        If the grid is too short, nonuniform, or lacks the required support.
    """
    if (
        grid.ndim != 1
        or len(grid) < 3
        or not np.isfinite(grid).all()
        or grid[0] != 0.0
        or np.any(np.diff(grid) <= 0.0)
        or not np.allclose(np.diff(grid), grid[1], rtol=1e-8, atol=1e-12)
        or grid[-1] < source_end
    ):
        raise ValueError(
            "Grid must be uniform, start at zero and cover the continuum."
        )


def _interpolate_weighted_continuum(
    grid: np.ndarray,
    continuous_spectrum: PhononSpectra,
    continuous_weight: float,
) -> np.ndarray:
    """Interpolate the continuum onto ``grid`` and scale to ``continuous_weight``.

    Parameters
    ----------
    grid : numpy.ndarray
        Target uniform energy grid in eV.
    continuous_spectrum : PhononSpectra
        Unit-normalized source continuum spectrum.
    continuous_weight : float
        Nonnegative weight of the continuous vibrational contribution.

    Returns
    -------
    numpy.ndarray
        Weighted continuum density in 1/eV evaluated on ``grid``.

    Raises
    ------
    ValueError
        If ``continuous_weight`` is positive but the interpolated area is
        nonpositive.
    """
    if continuous_weight == 0.0:
        return np.zeros_like(grid)
    density = np.interp(
        grid,
        continuous_spectrum.energy_grid,
        continuous_spectrum.density_values,
        left=0.0,
        right=0.0,
    )
    area = float(np.trapezoid(density, grid))
    if area <= 0.0:
        raise ValueError("Target grid does not resolve the continuum.")
    return density * (continuous_weight / area)


def _deposit_linear_node(
    grid: np.ndarray,
    density: np.ndarray,
    energy: float,
    weight: float,
    spacing: float,
) -> None:
    """Deposit a single oscillator onto one or two adjacent interior nodes.

    Parameters
    ----------
    grid : numpy.ndarray
        Uniform energy grid in eV.
    density : numpy.ndarray
        Mutable density array in 1/eV updated in place.
    energy : float
        Snapped interior oscillator energy in eV.
    weight : float
        Oscillator weight.
    spacing : float
        Uniform grid spacing in eV.
    """
    upper = int(np.searchsorted(grid, energy, side="left"))
    if energy == grid[upper]:
        density[upper] += weight / spacing
    else:
        lower = upper - 1
        fraction = (energy - grid[lower]) / (grid[upper] - grid[lower])
        density[lower] += weight * (1.0 - fraction) / spacing
        density[upper] += weight * fraction / spacing


def _deposit_gaussian_oscillator(
    grid: np.ndarray,
    density: np.ndarray,
    energy: float,
    weight: float,
    sigma: float,
) -> None:
    """Deposit a resolved Gaussian oscillator on interior nodes in place.

    Parameters
    ----------
    grid : numpy.ndarray
        Uniform energy grid in eV.
    density : numpy.ndarray
        Mutable density array in 1/eV updated in place.
    energy : float
        Oscillator centroid energy in eV.
    weight : float
        Oscillator integrated weight.
    sigma : float
        Gaussian standard deviation in eV.

    Raises
    ------
    ValueError
        If the Gaussian peak has nonpositive area on the interior grid.
    """
    # Keep the Gaussian contribution off the endpoint nodes.
    peak = np.zeros_like(grid)
    peak[1:-1] = np.exp(-0.5 * ((grid[1:-1] - energy) / sigma) ** 2)
    area = float(np.trapezoid(peak, grid))
    if area <= 0.0:
        raise ValueError("Oscillator outside resolved vibrational energy grid.")
    density += weight * (peak / area)


def _validate_gaussian_support(
    grid: np.ndarray,
    active: Sequence[DiscreteOscillator],
    oscillator_sigma_ev: float,
    spacing: float,
    tol: float,
) -> None:
    """Check that the grid resolves and contains each Gaussian oscillator.

    Parameters
    ----------
    grid : numpy.ndarray
        Uniform energy grid in eV.
    active : sequence of DiscreteOscillator
        Nonempty collection of modes with positive weight.
    oscillator_sigma_ev : float
        Positive Gaussian standard deviation in eV.
    spacing : float
        Grid spacing in eV.
    tol : float
        Endpoint comparison tolerance in eV.

    Raises
    ------
    ValueError
        If the spacing exceeds sigma / 3 or a mode lacks six sigma of support.
    """
    if spacing > (oscillator_sigma_ev / 3.0) * (1.0 + 1e-12):
        raise ValueError("Gaussian modes require energy spacing at most sigma / 3.")
    margin = _GAUSSIAN_MARGIN_SIGMAS * oscillator_sigma_ev
    if (
        min(mode.energy_ev for mode in active) < margin - tol
        or max(mode.energy_ev for mode in active) > grid[-1] - margin + tol
    ):
        raise ValueError("Gaussian modes require six sigma of grid support on both sides.")


def _bracket_mode_energy(grid: np.ndarray, energy: float, tol: float) -> float:
    """Resolve a line center on the interval spanned by interior nodes.

    Parameters
    ----------
    grid : numpy.ndarray
        Uniform energy grid in eV.
    energy : float
        Oscillator center in eV.
    tol : float
        Allowable roundoff in eV at the interior endpoints.

    Returns
    -------
    float
        Oscillator energy clipped to the interior interval within tolerance.

    Raises
    ------
    ValueError
        If the oscillator lies outside the interior interval by more than
        the tolerance.
    """
    if not grid[1] - tol <= energy <= grid[-2] + tol:
        raise ValueError("Grid must bracket each mode with interior nodes.")
    return min(max(energy, grid[1]), grid[-2])


def _deposit_oscillators(
    grid: np.ndarray,
    density: np.ndarray,
    modes: Sequence[DiscreteOscillator],
    initial_weight: float,
    oscillator_sigma_ev: float,
) -> float:
    """Deposit discrete oscillators onto interior grid nodes in place.

    Positive widths use area-normalized Gaussians. A zero width selects
    conservative two-node deposition.

    Parameters
    ----------
    grid : numpy.ndarray
        Uniform energy grid in eV.
    density : numpy.ndarray
        Mutable density array in 1/eV to update in place.
    modes : sequence of DiscreteOscillator
        Discrete oscillators to deposit.
    initial_weight : float
        Initial continuum weight before adding oscillator weights.
    oscillator_sigma_ev : float
        Gaussian broadening width in eV.

    Returns
    -------
    float
        Total vibrational weight (continuum plus active oscillators).

    Raises
    ------
    ValueError
        If any active oscillator lies outside the interior interval by more
        than the numerical tolerance, or if total weight is nonpositive.
    """
    spacing = float(grid[1])
    total_weight = float(initial_weight)
    tol = 1e-12 * spacing
    active = [mode for mode in modes if mode.weight > 0.0]
    if oscillator_sigma_ev > 0.0 and active:
        _validate_gaussian_support(grid, active, oscillator_sigma_ev, spacing, tol)

    for mode in active:
        if oscillator_sigma_ev > 0.0:
            _deposit_gaussian_oscillator(
                grid, density, mode.energy_ev, mode.weight, oscillator_sigma_ev
            )
        else:
            energy = _bracket_mode_energy(grid, mode.energy_ev, tol)
            _deposit_linear_node(grid, density, energy, mode.weight, spacing)
        total_weight += mode.weight

    if total_weight <= 0.0:
        raise ValueError("Total vibrational weight must be positive.")
    return total_weight


def reconstruct_vibrational_spectrum(
    continuous_spectrum: PhononSpectra,
    continuous_weight: float,
    oscillators: Iterable[DiscreteOscillator],
    energy_grid: ArrayLike | None = None,
    *,
    oscillator_sigma_ev: float = DEFAULT_OSCILLATOR_SIGMA_EV,
) -> tuple[PhononSpectra, float]:
    """Combine continuum and discrete modes on a uniform energy grid.

    Parameters
    ----------
    continuous_spectrum : PhononSpectra
        Normalized continuum with energies in eV and densities in inverse eV.
    continuous_weight : float
        Nonnegative continuum contribution, excluding translation.
    oscillators : iterable of DiscreteOscillator
        Mode energies in eV and their integrated weights.
    energy_grid : array_like or None, optional
        Uniform grid starting at zero. Default subdivides source spacing to
        resolve ``oscillator_sigma_ev`` and extends beyond the highest mode.
    oscillator_sigma_ev : float, default=0.001
        Gaussian standard deviation in eV for discrete oscillators (matching
        the illustrative water reference). Set to ``0.0`` for conservative
        two-node deposition.

    Returns
    -------
    spectrum : PhononSpectra
        Independently normalized full vibrational distribution.
    weight : float
        Sum of continuum and mode weights, excluding translation.

    Raises
    ------
    ValueError
        If ``continuous_weight`` or ``oscillator_sigma_ev`` is negative or
        nonfinite, the grid violates uniformity or mode-support requirements,
        or total vibrational weight is nonpositive.

    Notes
    -----
    Gaussian broadening is an assumed representation of the LEAPR lines.
    Each mode is normalized independently to preserve its tabulated weight.
    Gaussian grids require at least three steps per sigma and six sigma of
    support on each side of the peak center.
    """
    modes = tuple(oscillators)
    weight = float(continuous_weight)
    require_finite_nonnegative(weight, message="Continuous weight must be finite and nonnegative.")
    sigma = _validate_sigma(oscillator_sigma_ev)

    grid = _resolve_uniform_grid(
        continuous_spectrum.energy_grid, modes, energy_grid, sigma
    )
    density = _interpolate_weighted_continuum(
        grid, continuous_spectrum, weight
    )
    total_weight = _deposit_oscillators(grid, density, modes, weight, sigma)
    # Normalize the full spectral shape and return its physical weight separately.
    spectrum = PhononSpectra(pd.Series(density / total_weight, index=grid))
    return spectrum, total_weight
