"""
Valley-aware rigid segment translation for phonon spectra.

This module provides a shape-preserving alternative to diffeomorphic spatial
warping. Spectral regions are translated rigidly, while ambiguous neighborhoods
around domain boundaries are removed and reconstructed through nearby valleys.

Unlike a cross-fade formulation, this implementation never adds two translated
copies of the spectrum together. The final spectrum is a single PCHIP
reconstruction through trusted translated samples.

The public ``apply_spatial_warp`` signature is intentionally compatible with
the TARA sampler spatial-operator interface.

Units
-----
``x_grid`` and ``boundaries`` are expressed in eV. ``shifts`` are expressed
in meV, matching the existing sampler configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numba import njit, prange
from scipy.interpolate import PchipInterpolator

from common.validation import (
    require_finite_array,
    require_ndim,
    require_nonnegative_array,
)


DEFAULT_VALLEY_SEARCH_RADIUS_MEV = 3.0
DEFAULT_VALLEY_SMOOTHING_MEV = 0.5
DEFAULT_RECONSTRUCTION_GUARD_MEV = 0.5
DEFAULT_SHIFT_GUARD_FACTOR = 0.55
DEFAULT_MIN_GUARD_POINTS = 3
DEFAULT_ORIGIN_MAX_DISTORTION = 0.5
DEFAULT_ORIGIN_SAFETY_FACTOR = 1.05
DEFAULT_ORIGIN_MIN_POINTS = 4

_EPS = 1e-14


@njit(cache=True)
def _choose_valley_index(x_grid, y_smooth, first, last, boundary, distance_scale):
    """Select the lowest-cost valley candidate inside one search window.

    Parameters
    ----------
    x_grid, y_smooth : numpy.ndarray
        Source energies and smoothed density used for valley detection.
    first, last : int
        Inclusive bounds of the search window on the grid.
    boundary : float
        Nominal boundary in eV.
    distance_scale : float
        Energy scale in eV used to normalize distance from the boundary.

    Returns
    -------
    int
        Grid index of the best candidate, preferring the earlier index on ties.
    """
    value_min = y_smooth[first]
    value_max = y_smooth[first]
    for grid_idx in range(first + 1, last + 1):
        value = y_smooth[grid_idx]
        if value < value_min:
            value_min = value
        if value > value_max:
            value_max = value

    value_span = max(value_max - value_min, _EPS)
    best_index = first
    best_cost = np.inf

    for grid_idx in range(first, last + 1):
        amplitude_cost = (
            (y_smooth[grid_idx] - value_min) / value_span
        )
        distance_cost = (
            abs(x_grid[grid_idx] - boundary) / distance_scale
        )
        cost = amplitude_cost + 0.08 * distance_cost

        if grid_idx > first and grid_idx < last:
            if (
                y_smooth[grid_idx] <= y_smooth[grid_idx - 1]
                and y_smooth[grid_idx] <= y_smooth[grid_idx + 1]
            ):
                cost -= 0.03

        if cost < best_cost:
            best_cost = cost
            best_index = grid_idx
    return best_index


@njit(inline="always")
def _valley_search_window(x_grid, nominal_boundaries, boundary_idx, search_radius_eV, dx):
    """Find the grid indices allowed for one valley search.

    Parameters
    ----------
    x_grid, nominal_boundaries : numpy.ndarray
        Source energies and desired internal boundaries in eV.
    boundary_idx : int
        Boundary whose search window is being calculated.
    search_radius_eV, dx : float
        Maximum search radius and mean grid spacing in eV.

    Returns
    -------
    tuple of int
        Inclusive search indices, or (-1, -1) for an empty window.
    """
    n_grid = x_grid.size
    n_boundaries = nominal_boundaries.size
    boundary = nominal_boundaries[boundary_idx]
    if boundary_idx == 0:
        lower_limit = x_grid[0] + dx
    else:
        lower_limit = (
            0.5
            * (nominal_boundaries[boundary_idx - 1] + boundary)
            + dx
        )

    if boundary_idx == n_boundaries - 1:
        upper_limit = x_grid[n_grid - 1] - dx
    else:
        upper_limit = (
            0.5
            * (boundary + nominal_boundaries[boundary_idx + 1])
            - dx
        )

    lower = max(boundary - search_radius_eV, lower_limit)
    upper = min(boundary + search_radius_eV, upper_limit)
    if upper <= lower:
        return -1, -1

    first = -1
    last = -1
    for grid_idx in range(n_grid):
        x_value = x_grid[grid_idx]
        if x_value >= lower and x_value <= upper:
            if first < 0:
                first = grid_idx
            last = grid_idx

    return first, last


@njit(inline="always")
def _enforce_boundary_spacing(effective, x_grid, dx) -> None:
    """Apply the minimum spacing and source-edge limits in place.

    Parameters
    ----------
    effective : numpy.ndarray
        Selected boundary positions to adjust.
    x_grid : numpy.ndarray
        Source energies in eV.
    dx : float
        Mean grid spacing in eV.
    """
    n_boundaries = effective.size
    n_grid = x_grid.size
    minimum_separation = max(dx, _EPS)
    for boundary_idx in range(1, n_boundaries):
        minimum_allowed = (
            effective[boundary_idx - 1] + minimum_separation
        )
        if effective[boundary_idx] <= minimum_allowed:
            effective[boundary_idx] = minimum_allowed

    lower_clip = x_grid[0] + minimum_separation
    upper_clip = x_grid[n_grid - 1] - minimum_separation
    for boundary_idx in range(n_boundaries):
        if effective[boundary_idx] < lower_clip:
            effective[boundary_idx] = lower_clip
        elif effective[boundary_idx] > upper_clip:
            effective[boundary_idx] = upper_clip


@njit
def _snap_boundaries_kernel(
    x_grid: np.ndarray,
    y_smooth: np.ndarray,
    nominal_boundaries: np.ndarray,
    search_radius_eV: float,
    dx: float,
) -> np.ndarray:
    """
    Search valley-aware boundary positions in compiled loops.

    Parameters
    ----------
    x_grid : numpy.ndarray
        Strictly increasing source energy grid.
    y_smooth : numpy.ndarray
        Smoothed density used only for valley detection.
    nominal_boundaries : numpy.ndarray
        Requested internal boundaries.
    search_radius_eV : float
        Maximum boundary displacement in eV.
    dx : float
        Mean grid spacing in eV.

    Returns
    -------
    numpy.ndarray
        Ordered effective boundary positions.
    """
    effective = nominal_boundaries.copy()
    n_boundaries = nominal_boundaries.size

    for boundary_idx in range(n_boundaries):
        boundary = nominal_boundaries[boundary_idx]

        first, last = _valley_search_window(
            x_grid, nominal_boundaries, boundary_idx, search_radius_eV, dx,
        )
        if first < 0 or last - first + 1 < 3:
            continue

        best_index = _choose_valley_index(
            x_grid, y_smooth, first, last, boundary, max(search_radius_eV, dx),
        )
        effective[boundary_idx] = x_grid[best_index]

    _enforce_boundary_spacing(effective, x_grid, dx)
    return effective


@njit
def _compute_reconstruction_guards_kernel(
    effective_boundaries: np.ndarray,
    segment_ranges: np.ndarray,
    segment_shifts_eV: np.ndarray,
    dx: float,
    base_guard_eV: float,
    shift_guard_factor: float,
    min_guard_points: int,
) -> np.ndarray:
    """
    Compute reconstruction guards without Python-domain loops.

    Parameters
    ----------
    effective_boundaries : numpy.ndarray
        Valley-aware internal boundaries in eV.
    segment_ranges : numpy.ndarray
        Effective segment ranges with shape ``(n_segments, 2)``.
    segment_shifts_eV : numpy.ndarray
        Translation assigned to each segment in eV.
    dx : float
        Mean source-grid spacing in eV.
    base_guard_eV : float
        Minimum requested guard half-width in eV.
    shift_guard_factor : float
        Factor multiplying neighboring relative shifts.
    min_guard_points : int
        Minimum number of grid intervals retained in the guard floor.

    Returns
    -------
    numpy.ndarray
        Guard half-width for each internal boundary.
    """
    n_boundaries = effective_boundaries.size
    guards = np.empty(n_boundaries, dtype=np.float64)
    sampling_floor = max(min_guard_points, 1) * abs(dx)

    for boundary_idx in range(n_boundaries):
        shift_difference = abs(
            segment_shifts_eV[boundary_idx + 1]
            - segment_shifts_eV[boundary_idx]
        )
        desired = max(
            base_guard_eV,
            sampling_floor,
            shift_guard_factor * shift_difference,
        )

        left_width = (
            segment_ranges[boundary_idx, 1]
            - segment_ranges[boundary_idx, 0]
        )
        right_width = (
            segment_ranges[boundary_idx + 1, 1]
            - segment_ranges[boundary_idx + 1, 0]
        )
        maximum = 0.45 * min(left_width, right_width)
        guards[boundary_idx] = (
            min(desired, maximum) if maximum > 0.0 else 0.0
        )

    return guards


@njit
def _collapse_sorted_nodes_kernel(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collapse numerically duplicate sorted nodes in compiled loops.

    Parameters
    ----------
    x_nodes : numpy.ndarray
        Sorted node coordinates.
    y_nodes : numpy.ndarray
        Density values corresponding to ``x_nodes``.
    tolerance : float
        Maximum coordinate separation treated as a duplicate group.

    Returns
    -------
    x_unique, y_unique : tuple of numpy.ndarray
        Strictly increasing coordinates and averaged density values.
    """
    n_nodes = x_nodes.size
    x_unique = np.empty(n_nodes, dtype=np.float64)
    y_unique = np.empty(n_nodes, dtype=np.float64)

    start = 0
    unique_count = 0
    while start < n_nodes:
        stop = start + 1
        while (
            stop < n_nodes
            and abs(x_nodes[stop] - x_nodes[start]) <= tolerance
        ):
            stop += 1

        x_sum = 0.0
        y_sum = 0.0
        for node_idx in range(start, stop):
            x_sum += x_nodes[node_idx]
            y_sum += y_nodes[node_idx]

        count = stop - start
        x_unique[unique_count] = x_sum / count
        y_unique[unique_count] = y_sum / count
        unique_count += 1
        start = stop

    return (
        x_unique[:unique_count].copy(),
        y_unique[:unique_count].copy(),
    )


@njit
def _pchip_endpoint_slope(
    h0: float,
    h1: float,
    m0: float,
    m1: float,
) -> float:
    """
    Return the one-sided PCHIP endpoint derivative.

    Parameters
    ----------
    h0, h1 : float
        Widths of the first two adjacent interpolation intervals.
    m0, m1 : float
        Secant slopes of the first two adjacent interpolation intervals.

    Returns
    -------
    float
        Shape-preserving endpoint derivative.
    """
    derivative = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)

    if np.sign(derivative) != np.sign(m0):
        return 0.0

    if (
        np.sign(m0) != np.sign(m1)
        and abs(derivative) > 3.0 * abs(m0)
    ):
        return 3.0 * m0

    return derivative


@njit(inline="always")
def _pchip_interval_slopes(x_nodes, y_nodes, n_nodes):
    """Compute adjacent node distances and secant slopes.

    Parameters
    ----------
    x_nodes, y_nodes : numpy.ndarray
        Coordinates and values of one trusted node row.
    n_nodes : int
        Number of valid nodes in the row.

    Returns
    -------
    tuple of numpy.ndarray
        Interval widths and secant slopes, in interval order.
    """
    h = np.empty(n_nodes - 1, dtype=np.float64)
    slopes = np.empty(n_nodes - 1, dtype=np.float64)
    for interval_idx in range(n_nodes - 1):
        h[interval_idx] = (
            x_nodes[interval_idx + 1] - x_nodes[interval_idx]
        )
        slopes[interval_idx] = (
            (y_nodes[interval_idx + 1] - y_nodes[interval_idx])
            / h[interval_idx]
        )

    return h, slopes


@njit
def _pchip_derivatives(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """
    Compute shape-preserving PCHIP node derivatives.

    Parameters
    ----------
    x_nodes : numpy.ndarray
        Padded row containing strictly increasing trusted coordinates.
    y_nodes : numpy.ndarray
        Padded row containing trusted density values.
    n_nodes : int
        Number of valid entries in the padded rows.

    Returns
    -------
    numpy.ndarray
        PCHIP derivative at each valid trusted node.
    """
    derivatives = np.empty(n_nodes, dtype=np.float64)

    if n_nodes == 2:
        slope = (
            (y_nodes[1] - y_nodes[0])
            / (x_nodes[1] - x_nodes[0])
        )
        derivatives[0] = slope
        derivatives[1] = slope
        return derivatives

    h, slopes = _pchip_interval_slopes(x_nodes, y_nodes, n_nodes)

    for node_idx in range(1, n_nodes - 1):
        left_slope = slopes[node_idx - 1]
        right_slope = slopes[node_idx]

        if (
            left_slope == 0.0
            or right_slope == 0.0
            or np.sign(left_slope) != np.sign(right_slope)
        ):
            derivatives[node_idx] = 0.0
            continue

        w1 = 2.0 * h[node_idx] + h[node_idx - 1]
        w2 = h[node_idx] + 2.0 * h[node_idx - 1]
        derivatives[node_idx] = (
            (w1 + w2)
            / (w1 / left_slope + w2 / right_slope)
        )

    derivatives[0] = _pchip_endpoint_slope(
        h[0],
        h[1],
        slopes[0],
        slopes[1],
    )
    derivatives[n_nodes - 1] = _pchip_endpoint_slope(
        h[n_nodes - 2],
        h[n_nodes - 3],
        slopes[n_nodes - 2],
        slopes[n_nodes - 3],
    )
    return derivatives


@njit(inline="always")
def _pchip_interval_value(x_nodes, y_nodes, derivatives, interval_idx, x_value) -> float:
    """Evaluate the cubic Hermite polynomial on one PCHIP interval.

    Parameters
    ----------
    x_nodes, y_nodes, derivatives : numpy.ndarray
        Trusted node coordinates, values, and PCHIP slopes.
    interval_idx : int
        Index of the interval's left node.
    x_value : float
        Energy within the interval.

    Returns
    -------
    float
        Interpolated density.
    """
    x_left = x_nodes[interval_idx]
    x_right = x_nodes[interval_idx + 1]
    h = x_right - x_left
    t = (x_value - x_left) / h
    t2 = t * t
    t3 = t2 * t

    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2

    return (
        h00 * y_nodes[interval_idx]
        + h10 * h * derivatives[interval_idx]
        + h01 * y_nodes[interval_idx + 1]
        + h11 * h * derivatives[interval_idx + 1]
    )


@njit(inline="always")
def _write_pchip_row(x_nodes, y_nodes, n_nodes, derivatives, x_output, non_negative, output) -> None:
    """Evaluate one PCHIP row without extrapolating beyond the trusted nodes.

    Parameters
    ----------
    x_nodes, y_nodes : numpy.ndarray
        Trusted coordinates and values for one sample.
    n_nodes : int
        Number of valid nodes.
    derivatives : numpy.ndarray
        Shape-preserving derivative at each valid node.
    x_output : numpy.ndarray
        Shared target energy grid.
    non_negative : bool
        Clamp negative interpolated values to zero when True.
    output : numpy.ndarray
        Zero-initialized output row.
    """
    n_output = x_output.size
    interval_idx = 0

    for output_idx in range(n_output):
        x_value = x_output[output_idx]
        if (
            x_value < x_nodes[0]
            or x_value > x_nodes[n_nodes - 1]
        ):
            continue

        if x_value == x_nodes[n_nodes - 1]:
            value = y_nodes[n_nodes - 1]
        else:
            while (
                interval_idx + 1 < n_nodes
                and x_nodes[interval_idx + 1] < x_value
            ):
                interval_idx += 1

            if interval_idx + 1 >= n_nodes:
                value = y_nodes[n_nodes - 1]
            else:
                value = _pchip_interval_value(x_nodes, y_nodes, derivatives, interval_idx, x_value)

        if non_negative and value < 0.0:
            value = 0.0
        output[output_idx] = value


@njit(parallel=True)
def _pchip_interpolate_rows(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    node_lengths: np.ndarray,
    x_output: np.ndarray,
    non_negative: np.ndarray,
) -> np.ndarray:
    """
    Evaluate many ragged PCHIP node sets on one shared output grid.

    Parameters
    ----------
    x_nodes, y_nodes : numpy.ndarray
        Padded trusted-node matrices with shape ``(n_samples, max_nodes)``.
    node_lengths : numpy.ndarray
        Number of valid nodes in each row.
    x_output : numpy.ndarray
        Shared output energy grid.
    non_negative : numpy.ndarray
        Whether each row should be clipped to non-negative density.

    Returns
    -------
    numpy.ndarray
        Reconstructed density matrix.
    """
    n_samples = x_nodes.shape[0]
    n_output = x_output.size
    output = np.zeros((n_samples, n_output), dtype=np.float64)

    for sample_idx in prange(n_samples):
        n_nodes = node_lengths[sample_idx]
        derivatives = _pchip_derivatives(
            x_nodes[sample_idx],
            y_nodes[sample_idx],
            n_nodes,
        )
        _write_pchip_row(
            x_nodes[sample_idx], y_nodes[sample_idx], n_nodes, derivatives,
            x_output, non_negative[sample_idx], output[sample_idx],
        )

    return output


@dataclass(frozen=True)
class _Segment:
    """
    Represent one source-energy interval and its rigid translation.

    Parameters
    ----------
    start : float
        Segment start energy in eV.
    end : float
        Segment end energy in eV.
    shift_eV : float
        Rigid translation assigned to the segment in eV.
    source_index : int or None
        Index of the original requested domain. ``None`` identifies an
        automatically inserted zero-shift background interval.
    """

    start: float
    end: float
    shift_eV: float
    source_index: Optional[int] = None


@dataclass(frozen=True)
class TranslationDiagnostics:
    """
    Store geometry used by the valley-aware reconstruction.

    Parameters
    ----------
    nominal_boundaries : numpy.ndarray
        Original internal boundaries in eV.
    effective_boundaries : numpy.ndarray
        Valley-snapped internal boundaries in eV.
    reconstruction_guard_widths : numpy.ndarray
        Half-width removed around each effective boundary in eV.
    segment_shifts_eV : numpy.ndarray
        Translation assigned to every covering segment.
    segment_ranges : tuple of tuple of float
        Effective source ranges after valley snapping.
    """

    nominal_boundaries: np.ndarray
    effective_boundaries: np.ndarray
    reconstruction_guard_widths: np.ndarray
    segment_shifts_eV: np.ndarray
    segment_ranges: Tuple[Tuple[float, float], ...]

    @property
    def crossfade_widths(self) -> np.ndarray:
        """
        Return a compatibility view of the former fade-width diagnostic.

        Returns
        -------
        numpy.ndarray
            Full removed width around each boundary in eV.
        """
        return 2.0 * self.reconstruction_guard_widths


def _validate_inputs(
    x_grid: np.ndarray,
    y_density: np.ndarray,
    boundaries: Sequence[Tuple[float, float]],
    shifts: np.ndarray,
) -> None:
    """
    Validate the spatial-translation inputs.

    Parameters
    ----------
    x_grid : numpy.ndarray
        Strictly increasing non-negative energy grid.
    y_density : numpy.ndarray
        Spectrum sampled on ``x_grid``.
    boundaries : sequence of tuple of float
        Requested non-overlapping source domains in eV.
    shifts : numpy.ndarray
        Per-domain translations in meV.

    Raises
    ------
    ValueError
        If the arrays, grid, boundaries, or shifts are inconsistent.
    """
    if x_grid.ndim != 1 or y_density.ndim != 1:
        raise ValueError("x_grid and y_density must be one-dimensional.")

    if x_grid.size != y_density.size:
        raise ValueError("x_grid and y_density must have the same length.")

    if x_grid.size < 2:
        raise ValueError("x_grid must contain at least two points.")

    if len(boundaries) != shifts.size:
        raise ValueError(
            "The number of spatial boundaries must match the number of shifts."
        )

    require_finite_array(x_grid, message="x_grid contains non-finite values.")

    require_finite_array(y_density, message="y_density contains non-finite values.")

    require_finite_array(shifts, message="shifts contains non-finite values.")

    # Examine adjacent grid differences to enforce ordering or locate transitions.
    if np.any(np.diff(x_grid) <= 0.0):
        raise ValueError("x_grid must be strictly increasing.")

    require_nonnegative_array(x_grid, message="The energy grid cannot contain negative energies.")

    for i, (start, end) in enumerate(boundaries):
        if not np.isfinite(start) or not np.isfinite(end):
            raise ValueError(f"Boundary {i} contains a non-finite value.")

        if end <= start:
            raise ValueError(
                f"Boundary {i} is invalid: end ({end}) must exceed start ({start})."
            )


def _mean_dx(x_grid: np.ndarray) -> float:
    """
    Return the mean grid spacing.

    Parameters
    ----------
    x_grid : numpy.ndarray
        Strictly increasing energy grid.

    Returns
    -------
    float
        Mean spacing in eV.
    """
    # Examine adjacent grid differences to enforce ordering or locate transitions.
    return float(np.mean(np.diff(x_grid)))


def _extend_output_grid_right(
    x_grid: np.ndarray,
    extension_eV: float,
) -> np.ndarray:
    """
    Extend the output grid toward positive energy only.

    Parameters
    ----------
    x_grid : numpy.ndarray
        Original energy grid.
    extension_eV : float
        Required positive-energy extension in eV.

    Returns
    -------
    numpy.ndarray
        Original or right-extended energy grid.
    """
    if extension_eV <= 0.0:
        return x_grid.copy()

    dx = _mean_dx(x_grid)
    n_extra = int(np.ceil(extension_eV / dx)) + 2
    extra = x_grid[-1] + np.arange(1, n_extra + 1, dtype=float) * dx
    # Assemble the grid segments in the order required by the reconstruction.
    return np.concatenate([x_grid, extra])


def _merge_covering_segments(segments, tolerance):
    """Coalesce neighboring source segments with the same translation.

    Parameters
    ----------
    segments : sequence of _Segment
        Covering source intervals in energy order.
    tolerance : float
        Endpoint tolerance in eV.

    Returns
    -------
    list of _Segment
        Covering intervals after coalescing equal shifts.
    """
    # Merge adjacent segments carrying the same translation.
    merged: List[_Segment] = []

    for segment in segments:
        if (
            merged
            and abs(merged[-1].shift_eV - segment.shift_eV) <= _EPS
            and abs(merged[-1].end - segment.start) <= tolerance
        ):
            previous = merged[-1]
            merged[-1] = _Segment(
                start=previous.start,
                end=segment.end,
                shift_eV=previous.shift_eV,
                source_index=(
                    previous.source_index
                    if previous.source_index == segment.source_index
                    else None
                ),
            )
        else:
            merged.append(segment)

    return merged


def _build_covering_segments(
    x_min: float,
    x_max: float,
    boundaries: Sequence[Tuple[float, float]],
    shifts_eV: np.ndarray,
    tolerance: float,
) -> List[_Segment]:
    """
    Build a complete non-overlapping segment cover.

    Parameters
    ----------
    x_min : float
        Minimum source energy.
    x_max : float
        Maximum source energy.
    boundaries : sequence of tuple of float
        Requested domains.
    shifts_eV : numpy.ndarray
        Per-domain shifts in eV.
    tolerance : float
        Numerical tolerance used when comparing adjacent boundaries.

    Returns
    -------
    list of _Segment
        Covering segments, including zero-shift gaps when needed.

    Raises
    ------
    ValueError
        If requested domains overlap.
    """
    requested = sorted(
        [
            (float(start), float(end), float(shift), i)
            for i, ((start, end), shift) in enumerate(
                zip(boundaries, shifts_eV)
            )
        ],
        key=lambda item: item[0],
    )

    segments: List[_Segment] = []
    cursor = x_min

    for start, end, shift, source_index in requested:
        if end <= x_min or start >= x_max:
            continue

        start = max(start, x_min)
        end = min(end, x_max)

        if start < cursor - tolerance:
            raise ValueError(
                "Spatial domains overlap. Segment translation requires "
                "non-overlapping source domains."
            )

        if abs(start - cursor) <= tolerance:
            start = cursor

        if start > cursor + tolerance:
            segments.append(
                _Segment(
                    start=cursor,
                    end=start,
                    shift_eV=0.0,
                    source_index=None,
                )
            )

        if end > start + tolerance:
            segments.append(
                _Segment(
                    start=start,
                    end=end,
                    shift_eV=shift,
                    source_index=source_index,
                )
            )
            cursor = end

    if cursor < x_max - tolerance:
        segments.append(
            _Segment(
                start=cursor,
                end=x_max,
                shift_eV=0.0,
                source_index=None,
            )
        )

    if not segments:
        segments = [
            _Segment(
                start=x_min,
                end=x_max,
                shift_eV=0.0,
                source_index=None,
            )
        ]

    return _merge_covering_segments(segments, tolerance)


def _gaussian_smooth(
    y_density: np.ndarray,
    sigma_eV: float,
    dx: float,
) -> np.ndarray:
    """
    Smooth the spectrum for robust valley detection.

    Parameters
    ----------
    y_density : numpy.ndarray
        Input spectrum.
    sigma_eV : float
        Gaussian standard deviation in eV.
    dx : float
        Mean energy-grid spacing in eV.

    Returns
    -------
    numpy.ndarray
        Smoothed spectrum.
    """
    if sigma_eV <= 0.0:
        return np.asarray(y_density, dtype=float).copy()

    sigma_points = sigma_eV / dx

    if sigma_points < 0.5:
        return np.asarray(y_density, dtype=float).copy()

    radius = max(1, int(np.ceil(4.0 * sigma_points)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_points) ** 2)
    kernel /= np.sum(kernel)

    # Pad the batch representation without changing the physical support of individual samples.
    padded = np.pad(
        np.asarray(y_density, dtype=float),
        (radius, radius),
        mode="edge",
    )
    return np.convolve(padded, kernel, mode="valid")


def _snap_boundaries_to_valleys(
    x_grid: np.ndarray,
    y_density: np.ndarray,
    nominal_boundaries: np.ndarray,
    search_radius_eV: float,
    smoothing_eV: float,
) -> np.ndarray:
    """
    Move internal boundaries toward nearby spectral valleys.

    Parameters
    ----------
    x_grid : numpy.ndarray
        Source energy grid.
    y_density : numpy.ndarray
        Source spectrum.
    nominal_boundaries : numpy.ndarray
        Internal segment boundaries.
    search_radius_eV : float
        Maximum displacement allowed for each boundary.
    smoothing_eV : float
        Smoothing scale used only during valley detection.

    Returns
    -------
    numpy.ndarray
        Ordered valley-aware boundary locations.
    """
    if nominal_boundaries.size == 0 or search_radius_eV <= 0.0:
        return nominal_boundaries.copy()

    dx = _mean_dx(x_grid)
    y_smooth = _gaussian_smooth(
        y_density=y_density,
        sigma_eV=smoothing_eV,
        dx=dx,
    )
    # The search contains nested domain/grid loops and is executed for every
    # translated sample, so keep it in compiled numerical code.
    return _snap_boundaries_kernel(
        x_grid=x_grid,
        y_smooth=y_smooth,
        nominal_boundaries=nominal_boundaries,
        search_radius_eV=search_radius_eV,
        dx=dx,
    )


def _effective_segment_ranges(
    x_grid: np.ndarray,
    effective_boundaries: np.ndarray,
) -> Tuple[Tuple[float, float], ...]:
    """
    Convert internal boundaries into source ranges.

    Parameters
    ----------
    x_grid : numpy.ndarray
        Source energy grid.
    effective_boundaries : numpy.ndarray
        Ordered internal boundaries.

    Returns
    -------
    tuple of tuple of float
        Effective segment ranges.
    """
    # Assemble the grid segments in the order required by the reconstruction.
    edges = np.concatenate(
        (
            [float(x_grid[0])],
            effective_boundaries,
            [float(x_grid[-1])],
        )
    )
    return tuple(
        (float(edges[i]), float(edges[i + 1]))
        for i in range(edges.size - 1)
    )


def _compute_reconstruction_guards(
    effective_boundaries: np.ndarray,
    segment_ranges: Tuple[Tuple[float, float], ...],
    segment_shifts_eV: np.ndarray,
    x_grid: np.ndarray,
    base_guard_eV: float,
    shift_guard_factor: float,
    min_guard_points: int,
) -> np.ndarray:
    """
    Compute source-space neighborhoods removed around each valley.

    The guard eliminates samples for which two independently translated
    segments would otherwise overlap visibly. Missing low-density regions are
    reconstructed later by one continuous interpolant.

    Parameters
    ----------
    effective_boundaries : numpy.ndarray
        Valley-aware internal boundaries.
    segment_ranges : tuple of tuple of float
        Effective source ranges.
    segment_shifts_eV : numpy.ndarray
        Segment translations in eV.
    x_grid : numpy.ndarray
        Source energy grid.
    base_guard_eV : float
        Minimum requested guard half-width.
    shift_guard_factor : float
        Factor multiplying the relative neighboring shift.
    min_guard_points : int
        Sampling floor for each guard half-width.

    Returns
    -------
    numpy.ndarray
        Guard half-widths in eV.
    """
    if effective_boundaries.size == 0:
        return np.empty(0, dtype=float)

    dx = abs(_mean_dx(x_grid))
    segment_ranges_array = np.asarray(segment_ranges, dtype=np.float64)

    return _compute_reconstruction_guards_kernel(
        effective_boundaries=effective_boundaries,
        segment_ranges=segment_ranges_array,
        segment_shifts_eV=segment_shifts_eV,
        dx=dx,
        base_guard_eV=base_guard_eV,
        shift_guard_factor=shift_guard_factor,
        min_guard_points=int(min_guard_points),
    )


def _origin_anchored_coordinates(
    x_source: np.ndarray,
    origin: float,
    shift_eV: float,
    dx: float,
    max_distortion: float = DEFAULT_ORIGIN_MAX_DISTORTION,
    safety_factor: float = DEFAULT_ORIGIN_SAFETY_FACTOR,
    min_points: int = DEFAULT_ORIGIN_MIN_POINTS,
) -> np.ndarray:
    """
    Map the first segment while keeping its first energy point fixed.

    The displacement follows the same right-sided raised-cosine principle used
    by the diffeomorphic implementation:

    * displacement is exactly zero at the first original energy;
    * displacement grows smoothly away from that point;
    * displacement reaches the requested rigid shift after a safe transition;
    * the mapped coordinates remain strictly increasing.

    Parameters
    ----------
    x_source : numpy.ndarray
        Source coordinates belonging to the first segment.
    origin : float
        First energy value of the original spectrum.
    shift_eV : float
        Requested first-segment shift in eV.
    dx : float
        Mean input-grid spacing in eV.
    max_distortion : float, default=0.5
        Maximum intended magnitude of the displacement derivative.
    safety_factor : float, default=1.05
        Multiplicative safety margin for the transition width.
    min_points : int, default=4
        Minimum number of grid intervals spanning the transition.

    Returns
    -------
    numpy.ndarray
        Monotonic output coordinates for the first segment.

    Raises
    ------
    ValueError
        If a monotonic anchored map cannot be constructed.
    """
    x_source = np.asarray(x_source, dtype=float)

    if x_source.size == 0 or abs(float(shift_eV)) < 1e-15:
        return x_source.copy()

    if max_distortion <= 0.0:
        raise ValueError("max_distortion must be positive.")

    safe_width = (
        safety_factor
        * np.pi
        * abs(float(shift_eV))
        / (2.0 * max_distortion)
    )
    transition_width = max(
        safe_width,
        max(int(min_points), 2) * abs(float(dx)),
    )

    distance = x_source - float(origin)
    step = np.zeros_like(x_source, dtype=float)

    full_shift = distance >= transition_width
    step[full_shift] = 1.0

    transition = (distance > 0.0) & (distance < transition_width)
    if np.any(transition):
        u = distance[transition] / transition_width
        step[transition] = 0.5 * (1.0 - np.cos(np.pi * u))

    x_mapped = x_source + float(shift_eV) * step

    # The first original coordinate is a true geometric anchor.
    at_origin = np.isclose(
        x_source,
        origin,
        rtol=0.0,
        atol=max(abs(float(dx)) * 1e-8, 1e-14),
    )
    x_mapped[at_origin] = float(origin)

    # Examine adjacent grid differences to enforce ordering or locate transitions.
    if x_mapped.size > 1 and np.any(np.diff(x_mapped) <= 0.0):
        raise ValueError(
            "Could not construct a monotonic freeze_origin transition for "
            "the first segment. Reduce the requested shift or increase the "
            "origin transition width."
        )

    return x_mapped


def _trusted_source_indices(x_grid, start, end, i, n_segments, guards):
    """Select source nodes after trimming the reconstruction guards.

    Parameters
    ----------
    x_grid : numpy.ndarray
        Original energy grid in eV.
    start, end : float
        Source segment bounds in eV.
    i, n_segments : int
        Segment index and total number of segments.
    guards : numpy.ndarray
        Guard widths in eV at the internal boundaries.

    Returns
    -------
    numpy.ndarray
        Source indices inside the trusted region or the fallback core.
    """
    lower = start
    upper = end

    if i > 0:
        lower += guards[i - 1]

    if i < n_segments - 1:
        upper -= guards[i]

    mask = (x_grid >= lower) & (x_grid <= upper)
    indices = np.flatnonzero(mask)

    # Narrow domains can lose most points when neighboring shifts differ.
    # Keep the deepest available core rather than mixing shifted copies.
    if indices.size < 2:
        midpoint = 0.5 * (start + end)
        center = int(np.argmin(np.abs(x_grid - midpoint)))
        lo = max(0, center - 1)
        hi = min(x_grid.size, center + 2)
        indices = np.arange(lo, hi)

        inside = (
            (x_grid[indices] >= start)
            & (x_grid[indices] <= end)
        )
        indices = indices[inside]
    return indices


def _build_trusted_segment_samples(
    x_grid: np.ndarray,
    y_density: np.ndarray,
    segment_ranges: Tuple[Tuple[float, float], ...],
    segment_shifts_eV: np.ndarray,
    reconstruction_guards: np.ndarray,
    freeze_origin: bool,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build rigidly translated trusted samples for every segment.

    Parameters
    ----------
    x_grid : numpy.ndarray
        Source energy grid.
    y_density : numpy.ndarray
        Source spectrum.
    segment_ranges : tuple of tuple of float
        Valley-aware segment ranges.
    segment_shifts_eV : numpy.ndarray
        Segment translations in eV.
    reconstruction_guards : numpy.ndarray
        Removed half-width around each internal boundary.
    freeze_origin : bool
        Whether the first original energy sample must remain anchored.

    Returns
    -------
    list of tuple of numpy.ndarray
        Per-segment translated ``(x, y)`` trusted samples.
    """
    dx = abs(_mean_dx(x_grid))
    translated_segments: List[Tuple[np.ndarray, np.ndarray]] = []

    for i, ((start, end), shift_eV) in enumerate(
        zip(segment_ranges, segment_shifts_eV)
    ):
        indices = _trusted_source_indices(
            x_grid, start, end, i, len(segment_ranges), reconstruction_guards,
        )

        if indices.size == 0:
            translated_segments.append(
                (np.empty(0, dtype=float), np.empty(0, dtype=float))
            )
            continue

        x_source = x_grid[indices]
        y_translated = y_density[indices].copy()

        if freeze_origin and i == 0:
            # Freeze the GEOMETRY of the first original energy point.
            # The first segment therefore begins on the original spectrum and
            # separates smoothly from it, exactly as an anchored displacement
            # field should. This is fundamentally different from restoring only
            # y_output[0] after a rigid translation.
            x_translated = _origin_anchored_coordinates(
                x_source=x_source,
                origin=float(x_grid[0]),
                shift_eV=float(shift_eV),
                dx=dx,
            )
        else:
            # All later trusted segment cores remain true rigid translations.
            x_translated = x_source + float(shift_eV)

        # Negative output energies are physically forbidden. For the frozen
        # first segment the anchored map begins exactly at x_grid[0].
        physical = x_translated >= x_grid[0] - 0.5 * dx
        x_translated = x_translated[physical]
        y_translated = y_translated[physical]

        translated_segments.append(
            (x_translated, y_translated)
        )

    return translated_segments


def _trim_translated_overlaps(
    translated_segments: List[Tuple[np.ndarray, np.ndarray]],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Remove ambiguous overlap between neighboring translated segment cores.

    Parameters
    ----------
    translated_segments : list of tuple of numpy.ndarray
        Rigidly translated trusted samples.

    Returns
    -------
    list of tuple of numpy.ndarray
        Ordered samples with neighboring overlaps removed.
    """
    trimmed = [
        (x.copy(), y.copy())
        for x, y in translated_segments
    ]

    for i in range(len(trimmed) - 1):
        left_x, left_y = trimmed[i]
        right_x, right_y = trimmed[i + 1]

        if left_x.size == 0 or right_x.size == 0:
            continue

        if left_x[-1] < right_x[0]:
            continue

        # One output energy must belong to one reconstructed branch only.
        # Use a midpoint cut inside the low-density valley neighborhood.
        cut = 0.5 * (left_x[-1] + right_x[0])

        left_keep = left_x <= cut
        right_keep = right_x > cut

        # Retain at least one trusted point per side when possible.
        if not np.any(left_keep):
            left_keep[0] = True

        if not np.any(right_keep):
            right_keep[-1] = True

        trimmed[i] = (
            left_x[left_keep],
            left_y[left_keep],
        )
        trimmed[i + 1] = (
            right_x[right_keep],
            right_y[right_keep],
        )

    return trimmed


def _merge_trusted_samples(
    translated_segments: List[Tuple[np.ndarray, np.ndarray]],
    x_grid: np.ndarray,
    y_density: np.ndarray,
    freeze_origin: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge translated segment cores into one ordered set of interpolation nodes.

    Parameters
    ----------
    translated_segments : list of tuple of numpy.ndarray
        Per-segment translated trusted samples.
    x_grid : numpy.ndarray
        Original energy grid.
    y_density : numpy.ndarray
        Original spectrum.
    freeze_origin : bool
        Whether to insert the original first sample as an exact anchor.

    Returns
    -------
    tuple of numpy.ndarray
        Strictly increasing trusted ``(x, y)`` nodes.
    """
    x_parts = []
    y_parts = []

    # With freeze_origin=True the first translated segment already contains
    # (x_grid[0], y_density[0]) as an exact geometric anchor. Do not insert a
    # second standalone origin node, because that recreates the detached-branch
    # behavior this method is intended to avoid.
    for x_values, y_values in translated_segments:
        if x_values.size == 0:
            continue
        x_parts.append(x_values)
        y_parts.append(y_values)

    if not x_parts:
        raise RuntimeError(
            "No trusted samples remain after segment translation."
        )

    # Assemble the grid segments in the order required by the reconstruction.
    x_nodes = np.concatenate(x_parts)
    # Assemble the grid segments in the order required by the reconstruction.
    y_nodes = np.concatenate(y_parts)

    order = np.argsort(x_nodes, kind="mergesort")
    x_nodes = x_nodes[order]
    y_nodes = y_nodes[order]

    dx = abs(_mean_dx(x_grid))
    tolerance = max(dx * 1e-6, 1e-14)

    # Collapse duplicate groups in compiled loops. This operation is repeated
    # for every translated sample and previously used nested Python ``while``
    # loops over the complete trusted-node set.
    x_nodes, y_nodes = _collapse_sorted_nodes_kernel(
        x_nodes,
        y_nodes,
        tolerance,
    )

    if x_nodes.size < 2:
        raise RuntimeError(
            "At least two trusted reconstruction nodes are required."
        )

    # Examine adjacent grid differences to enforce ordering or locate transitions.
    if np.any(np.diff(x_nodes) <= 0.0):
        raise RuntimeError(
            "Trusted reconstruction nodes are not strictly increasing."
        )

    return x_nodes, y_nodes


def _reconstruct_single_spectrum(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    x_output: np.ndarray,
    non_negative: bool,
) -> np.ndarray:
    """
    Reconstruct one continuous spectrum through trusted translated samples.

    Parameters
    ----------
    x_nodes : numpy.ndarray
        Strictly increasing trusted output-energy nodes.
    y_nodes : numpy.ndarray
        Spectrum values at the trusted nodes.
    x_output : numpy.ndarray
        Requested output grid.
    non_negative : bool
        Whether negative reconstructed values should be clipped.

    Returns
    -------
    numpy.ndarray
        Single continuous reconstructed spectrum.
    """
    # Use shape-preserving interpolation to avoid overshoot between trusted samples.
    interpolator = PchipInterpolator(
        x_nodes,
        y_nodes,
        extrapolate=False,
    )
    y_output = np.asarray(
        interpolator(x_output),
        dtype=float,
    )
    y_output[~np.isfinite(y_output)] = 0.0

    if non_negative:
        y_output[y_output < 0.0] = 0.0

    return y_output


def _trusted_segment_masks(x_grid, segment_ranges, reconstruction_guards):
    """Mark trusted source coordinates for each segment after guard trimming.

    Parameters
    ----------
    x_grid : numpy.ndarray
        Original energy grid in eV.
    segment_ranges : sequence of tuple of float
        Source-energy intervals in eV.
    reconstruction_guards : numpy.ndarray
        Guard half-widths at the internal boundaries in eV.

    Returns
    -------
    numpy.ndarray
        Segment-by-grid mask with 1 at trusted coordinates and 0 elsewhere.
    """
    trusted_masks = np.zeros(
        (len(segment_ranges), x_grid.size),
        dtype=float,
    )

    for i, (start, end) in enumerate(segment_ranges):
        lower = start
        upper = end

        if i > 0:
            lower += reconstruction_guards[i - 1]

        if i < len(segment_ranges) - 1:
            upper -= reconstruction_guards[i]

        trusted_masks[i] = (
            (x_grid >= lower)
            & (x_grid <= upper)
        ).astype(float)
    return trusted_masks


def build_translation_plan(
    x_grid: np.ndarray,
    y_density: np.ndarray,
    boundaries: List[Tuple[float, float]],
    shifts: np.ndarray,
    *,
    snap_boundaries_to_valleys: bool = True,
    valley_search_radius_meV: float = DEFAULT_VALLEY_SEARCH_RADIUS_MEV,
    valley_smoothing_meV: float = DEFAULT_VALLEY_SMOOTHING_MEV,
    reconstruction_guard_meV: float = DEFAULT_RECONSTRUCTION_GUARD_MEV,
    shift_guard_factor: float = DEFAULT_SHIFT_GUARD_FACTOR,
    min_guard_points: int = DEFAULT_MIN_GUARD_POINTS,
) -> Tuple[List[_Segment], np.ndarray, TranslationDiagnostics]:
    """
    Build the valley-aware rigid-translation geometry.

    Parameters
    ----------
    x_grid : numpy.ndarray
        Source energy grid in eV.
    y_density : numpy.ndarray
        Source spectrum.
    boundaries : list of tuple of float
        Requested source domains in eV.
    shifts : numpy.ndarray
        Per-domain translations in meV.
    snap_boundaries_to_valleys : bool, default=True
        Whether nominal boundaries should move toward nearby valleys.
    valley_search_radius_meV : float, default=3.0
        Maximum boundary adjustment in meV.
    valley_smoothing_meV : float, default=0.5
        Smoothing scale used only during valley detection.
    reconstruction_guard_meV : float, default=0.5
        Minimum source-space guard half-width around each valley.
    shift_guard_factor : float, default=0.55
        Factor multiplying the neighboring relative shift when selecting the
        reconstruction guard.
    min_guard_points : int, default=3
        Minimum number of grid intervals removed on each side of a boundary.

    Returns
    -------
    segments : list of _Segment
        Covering source segments.
    trusted_masks : numpy.ndarray
        Boolean-like trusted masks with shape ``(n_segments, n_grid)``.
    diagnostics : TranslationDiagnostics
        Effective boundaries, guard widths, shifts, and source ranges.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    y_density = np.asarray(y_density, dtype=float)
    shifts = np.asarray(shifts, dtype=float)

    _validate_inputs(
        x_grid=x_grid,
        y_density=y_density,
        boundaries=boundaries,
        shifts=shifts,
    )

    shifts_eV = shifts / 1000.0
    dx = abs(_mean_dx(x_grid))
    tolerance = max(2.0 * dx, _EPS)

    segments = _build_covering_segments(
        x_min=float(x_grid[0]),
        x_max=float(x_grid[-1]),
        boundaries=boundaries,
        shifts_eV=shifts_eV,
        tolerance=tolerance,
    )

    nominal_boundaries = np.asarray(
        [segment.end for segment in segments[:-1]],
        dtype=float,
    )

    if snap_boundaries_to_valleys:
        effective_boundaries = _snap_boundaries_to_valleys(
            x_grid=x_grid,
            y_density=y_density,
            nominal_boundaries=nominal_boundaries,
            search_radius_eV=max(
                0.0,
                float(valley_search_radius_meV) / 1000.0,
            ),
            smoothing_eV=max(
                0.0,
                float(valley_smoothing_meV) / 1000.0,
            ),
        )
    else:
        effective_boundaries = nominal_boundaries.copy()

    segment_ranges = _effective_segment_ranges(
        x_grid=x_grid,
        effective_boundaries=effective_boundaries,
    )

    segment_shifts_eV = np.asarray(
        [segment.shift_eV for segment in segments],
        dtype=float,
    )

    reconstruction_guards = _compute_reconstruction_guards(
        effective_boundaries=effective_boundaries,
        segment_ranges=segment_ranges,
        segment_shifts_eV=segment_shifts_eV,
        x_grid=x_grid,
        base_guard_eV=max(
            0.0,
            float(reconstruction_guard_meV) / 1000.0,
        ),
        shift_guard_factor=max(
            0.0,
            float(shift_guard_factor),
        ),
        min_guard_points=max(
            1,
            int(min_guard_points),
        ),
    )

    trusted_masks = _trusted_segment_masks(x_grid, segment_ranges, reconstruction_guards)

    diagnostics = TranslationDiagnostics(
        nominal_boundaries=nominal_boundaries,
        effective_boundaries=effective_boundaries,
        reconstruction_guard_widths=reconstruction_guards,
        segment_shifts_eV=segment_shifts_eV,
        segment_ranges=segment_ranges,
    )

    return segments, trusted_masks, diagnostics


def _translated_trusted_nodes(x_grid, y_density, diagnostics, *, freeze_origin):
    """Translate, trim, and merge the trusted reconstruction nodes.

    Parameters
    ----------
    x_grid, y_density : numpy.ndarray
        Input grid in eV and corresponding density.
    diagnostics : TranslationDiagnostics
        Effective segment geometry, shifts, and guards.
    freeze_origin : bool
        Whether the first point retains its original coordinate.

    Returns
    -------
    tuple of numpy.ndarray
        Monotonic reconstruction nodes and their density values.
    """
    segment_shifts_eV = diagnostics.segment_shifts_eV

    translated_segments = _build_trusted_segment_samples(
        x_grid=x_grid,
        y_density=y_density,
        segment_ranges=diagnostics.segment_ranges,
        segment_shifts_eV=segment_shifts_eV,
        reconstruction_guards=diagnostics.reconstruction_guard_widths,
        freeze_origin=freeze_origin,
    )

    translated_segments = _trim_translated_overlaps(
        translated_segments=translated_segments,
    )

    x_nodes, y_nodes = _merge_trusted_samples(
        translated_segments=translated_segments,
        x_grid=x_grid,
        y_density=y_density,
        freeze_origin=freeze_origin,
    )
    return x_nodes, y_nodes


def _finalize_scalar_density(x_grid, y_density, x_output, y_output, freeze_origin, preserve_area) -> None:
    """Apply the origin guard and optional area correction in place.

    Parameters
    ----------
    x_grid, y_density : numpy.ndarray
        Original energies and density.
    x_output, y_output : numpy.ndarray
        Reconstructed energies and density.
    freeze_origin : bool
        Retain the original first density value.
    preserve_area : bool
        Apply the existing integral-based density rescaling when True.
    """
    if freeze_origin:
        # The first node is part of the anchored coordinate map itself.
        # This assignment is only a roundoff guard, not the anchoring mechanism.
        y_output[0] = y_density[0]

    if preserve_area:
        # Integrate against the physical coordinate grid, accounting for nonuniform spacing.
        area_old = float(np.trapezoid(y_density, x_grid))
        # Integrate against the physical coordinate grid, accounting for nonuniform spacing.
        area_new = float(np.trapezoid(y_output, x_output))

        if abs(area_new) > _EPS:
            scale = area_old / area_new

            if freeze_origin:
                # Keep the first point anchored after global area correction.
                y_output[1:] *= scale
                y_output[0] = y_density[0]
            else:
                y_output *= scale


def _reconstruct_translated_spectrum(
    x_grid, y_density, x_nodes, y_nodes, freeze_origin, preserve_area,
):
    """Reconstruct and validate one spectrum from its translated nodes.

    Parameters
    ----------
    x_grid, y_density : numpy.ndarray
        Original energy grid in eV and density.
    x_nodes, y_nodes : numpy.ndarray
        Monotonic reconstruction coordinates and trusted density values.
    freeze_origin, preserve_area : bool
        Apply the origin guard and the integral correction, respectively.

    Returns
    -------
    tuple of numpy.ndarray
        Physical energy grid in eV and reconstructed density.

    Raises
    ------
    RuntimeError
        If the output grid contains negative energies or the density is nonfinite.
    """
    # Extend only as far as trusted translated support requires.
    extension_eV = max(
        0.0,
        float(x_nodes[-1] - x_grid[-1]),
    )
    x_output = _extend_output_grid_right(
        x_grid=x_grid,
        extension_eV=extension_eV,
    )

    y_output = _reconstruct_single_spectrum(
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        x_output=x_output,
        non_negative=bool(np.all(y_density >= -1e-15)),
    )

    _finalize_scalar_density(
        x_grid, y_density, x_output, y_output, freeze_origin, preserve_area,
    )

    if np.any(x_output < 0.0):
        raise RuntimeError(
            "Internal error: reconstructed energy grid contains negative values."
        )

    if not np.all(np.isfinite(y_output)):
        raise RuntimeError(
            "Internal error: reconstruction produced non-finite density values."
        )

    return x_output, y_output


def apply_spatial_warp(
    x_grid: np.ndarray,
    y_density: np.ndarray,
    boundaries: List[Tuple[float, float]],
    shifts: np.ndarray,
    freeze_origin: bool,
    preserve_shape: bool = True,
    *,
    snap_boundaries_to_valleys: bool = True,
    valley_search_radius_meV: float = DEFAULT_VALLEY_SEARCH_RADIUS_MEV,
    valley_smoothing_meV: float = DEFAULT_VALLEY_SMOOTHING_MEV,
    reconstruction_guard_meV: float = DEFAULT_RECONSTRUCTION_GUARD_MEV,
    shift_guard_factor: float = DEFAULT_SHIFT_GUARD_FACTOR,
    min_guard_points: int = DEFAULT_MIN_GUARD_POINTS,
    preserve_area: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Translate spectral regions rigidly and reconstruct valley transitions.

    The method differs fundamentally from cross-faded segment translation.
    Neighboring translated spectra are never summed. Samples near each
    valley-aware boundary are removed before translation, all remaining samples
    are moved rigidly, and one PCHIP reconstruction connects the trusted
    translated samples into a single continuous output spectrum.

    Parameters
    ----------
    x_grid : numpy.ndarray
        Strictly increasing non-negative energy grid in eV.
    y_density : numpy.ndarray
        Spectrum sampled on ``x_grid``.
    boundaries : list of tuple of float
        Non-overlapping requested source domains in eV.
    shifts : numpy.ndarray
        Per-domain translations in meV.
    freeze_origin : bool
        If True, anchor the first energy coordinate and its original spectrum
        value. The displacement of the first segment starts at zero at this
        point and increases smoothly to the requested shift. This makes the
        shifted spectrum depart continuously from the original low-energy edge
        instead of joining a separately translated branch afterward.
    preserve_shape : bool, default=True
        Retained for compatibility with the common spatial-operator interface.
        Trusted segment samples are always translated rigidly.
    snap_boundaries_to_valleys : bool, default=True
        Move domain boundaries toward nearby low-density valleys before
        selecting trusted segment cores.
    valley_search_radius_meV : float, default=3.0
        Maximum valley-boundary adjustment in meV.
    valley_smoothing_meV : float, default=0.5
        Gaussian smoothing scale used only for valley detection.
    reconstruction_guard_meV : float, default=0.5
        Minimum half-width removed around each effective boundary before
        translation.
    shift_guard_factor : float, default=0.55
        Additional guard proportional to the relative neighboring shift.
    min_guard_points : int, default=3
        Minimum number of source grid intervals removed on each side of a
        reconstructed boundary.
    preserve_area : bool, default=False
        If True, rescale the final spectrum to preserve the original integral.

    Returns
    -------
    x_output : numpy.ndarray
        Non-negative output energy grid in eV.
    y_output : numpy.ndarray
        Single reconstructed translated spectrum.

    Notes
    -----
    If a negatively shifted feature crosses the minimum allowed energy, the
    portion below the physical grid is clipped. ``freeze_origin=True`` anchors
    the first original sample but does not create negative output energies.
    """
    del preserve_shape

    x_grid = np.asarray(x_grid, dtype=float)
    y_density = np.asarray(y_density, dtype=float)
    shifts = np.asarray(shifts, dtype=float)

    _validate_inputs(
        x_grid=x_grid,
        y_density=y_density,
        boundaries=boundaries,
        shifts=shifts,
    )

    # Preserve the exact identity for the no-shift case.
    if shifts.size == 0 or np.all(np.abs(shifts) < 1e-12):
        return x_grid.copy(), y_density.copy()

    _, _, diagnostics = build_translation_plan(
        x_grid=x_grid,
        y_density=y_density,
        boundaries=boundaries,
        shifts=shifts,
        snap_boundaries_to_valleys=snap_boundaries_to_valleys,
        valley_search_radius_meV=valley_search_radius_meV,
        valley_smoothing_meV=valley_smoothing_meV,
        reconstruction_guard_meV=reconstruction_guard_meV,
        shift_guard_factor=shift_guard_factor,
        min_guard_points=min_guard_points,
    )

    x_nodes, y_nodes = _translated_trusted_nodes(
        x_grid, y_density, diagnostics, freeze_origin=freeze_origin,
    )

    return _reconstruct_translated_spectrum(
        x_grid, y_density, x_nodes, y_nodes, freeze_origin, preserve_area,
    )


def _self_test() -> None:
    """
    Run lightweight regression tests for the reconstruction.

    Raises
    ------
    AssertionError
        If an identity, translation, origin, or continuity invariant fails.
    """
    x = np.linspace(0.0, 0.200, 4001)
    dx = _mean_dx(x)

    y = (
        0.15 * x / x.max()
        + 1.00 * np.exp(-0.5 * ((x - 0.040) / 0.0012) ** 2)
        + 0.75 * np.exp(-0.5 * ((x - 0.100) / 0.0015) ** 2)
        + 1.10 * np.exp(-0.5 * ((x - 0.160) / 0.0010) ** 2)
    )

    boundaries = [
        (0.000, 0.067),
        (0.067, 0.133),
        (0.133, 0.200),
    ]

    # Zero shifts must be an exact no-op.
    x0, y0 = apply_spatial_warp(
        x_grid=x,
        y_density=y,
        boundaries=boundaries,
        shifts=np.zeros(3),
        freeze_origin=True,
    )

    if not np.array_equal(x0, x):
        raise AssertionError(
            "Zero-shift reconstruction unexpectedly changed the grid."
        )

    if not np.array_equal(y0, y):
        raise AssertionError(
            "Zero-shift reconstruction is not an exact identity."
        )

    # Isolated peaks must retain their requested translations.
    shifts_meV = np.asarray([-2.0, 1.5, 2.0])
    x1, y1 = apply_spatial_warp(
        x_grid=x,
        y_density=y,
        boundaries=boundaries,
        shifts=shifts_meV,
        freeze_origin=True,
    )

    expected_peaks = np.asarray([0.0380, 0.1015, 0.1620])
    search_windows = (
        (0.033, 0.043),
        (0.096, 0.107),
        (0.157, 0.167),
    )

    measured_peaks = []
    for lower, upper in search_windows:
        mask = (x1 >= lower) & (x1 <= upper)
        measured_peaks.append(
            float(x1[mask][np.argmax(y1[mask])])
        )

    if not np.allclose(
        measured_peaks,
        expected_peaks,
        atol=2.0 * dx,
    ):
        raise AssertionError(
            "Rigid feature translation failed: "
            f"{measured_peaks} != {expected_peaks}."
        )

    # The first original sample must remain exactly frozen.
    if y1[0] != y[0] or x1[0] != x[0]:
        raise AssertionError(
            "freeze_origin did not preserve the first spectrum point."
        )

    # The first segment must leave the baseline continuously. A post-hoc
    # single-point anchor would create an abnormally large first derivative.
    first_diffs = np.abs(np.diff(y1[:40]))
    # Examine adjacent grid differences to enforce ordering or locate transitions.
    reference_diffs = np.abs(np.diff(y[:40]))
    local_reference = max(
        float(np.max(reference_diffs)),
        float(np.percentile(first_diffs, 75)),
        1e-14,
    )
    if float(first_diffs[0]) > 5.0 * local_reference:
        raise AssertionError(
            "freeze_origin created a one-point discontinuity at the first "
            "energy sample."
        )

    # Opposing neighboring shifts must still reconstruct one finite curve.
    alternating_boundaries = [
        (0.000, 0.050),
        (0.050, 0.100),
        (0.100, 0.150),
        (0.150, 0.200),
    ]
    alternating_shifts = np.asarray([2.0, -2.0, 2.0, -2.0])

    xa, ya = apply_spatial_warp(
        x_grid=x,
        y_density=y,
        boundaries=alternating_boundaries,
        shifts=alternating_shifts,
        freeze_origin=True,
    )

    if np.any(xa < 0.0) or not np.all(np.isfinite(ya)):
        raise AssertionError(
            "Alternating-shift reconstruction is not physical and finite."
        )

    # No output step should resemble a one-point branch reset.
    local_scale = max(
        # Examine adjacent grid differences to enforce ordering or locate transitions.
        float(np.percentile(np.abs(np.diff(ya)), 95)),
        1e-12,
    )
    # Examine adjacent grid differences to enforce ordering or locate transitions.
    if float(np.max(np.abs(np.diff(ya)))) > 25.0 * local_scale:
        raise AssertionError(
            "Reconstruction contains an isolated discontinuous step."
        )

    print("segment_translation.py self-test: OK")


if __name__ == "__main__":
    _self_test()


def _pack_trusted_nodes(node_sets, max_nodes):
    """Pack ragged node arrays for the compiled row interpolator.

    Parameters
    ----------
    node_sets : sequence of tuple of numpy.ndarray
        Trusted energy and density arrays for each sample.
    max_nodes : int
        Largest number of trusted nodes in a sample.

    Returns
    -------
    tuple of numpy.ndarray
        Energy and density matrices and valid node counts per row.
    """
    n_samples = len(node_sets)
    # Pack ragged trusted nodes once, then reconstruct every spectrum in one
    # parallel compiled call. This removes the per-sample SciPy PCHIP object
    # construction/evaluation from the batch hot path.
    x_nodes_matrix = np.empty((n_samples, max_nodes), dtype=np.float64)
    y_nodes_matrix = np.empty((n_samples, max_nodes), dtype=np.float64)
    node_lengths = np.empty(n_samples, dtype=np.int64)

    for sample_idx, (x_nodes, y_nodes) in enumerate(node_sets):
        n_nodes = x_nodes.size
        node_lengths[sample_idx] = n_nodes
        x_nodes_matrix[sample_idx, :n_nodes] = x_nodes
        y_nodes_matrix[sample_idx, :n_nodes] = y_nodes
    return x_nodes_matrix, y_nodes_matrix, node_lengths


def _resolve_translation_output_grid(x_grid, output_grid, max_sample_length, dx):
    """Allocate or validate the common energy grid for a translated batch.

    Parameters
    ----------
    x_grid : numpy.ndarray
        Base energy grid in eV.
    output_grid : numpy.ndarray or None
        Optional capacity grid supplied by the caller.
    max_sample_length : int
        Largest required valid prefix length.
    dx : float
        Mean grid spacing used for extension, in eV.

    Returns
    -------
    numpy.ndarray
        Common output grid for all reconstructions.

    Raises
    ------
    ValueError
        If the supplied grid does not preserve the baseline prefix or lacks capacity.
    """
    if output_grid is None:
        n_extra = max_sample_length - x_grid.size
        if n_extra > 0:
            extra = x_grid[-1] + np.arange(
                1, n_extra + 1, dtype=np.float64
            ) * dx
            # Append points while preserving the original grid as the prefix.
            output_grid = np.concatenate((x_grid, extra))
        else:
            output_grid = x_grid.copy()
    else:
        output_grid = np.asarray(output_grid, dtype=np.float64)
        if output_grid.ndim != 1 or output_grid.size < x_grid.size:
            raise ValueError(
                "output_grid must be one-dimensional and contain the base grid."
            )
        if not np.allclose(
            output_grid[: x_grid.size],
            x_grid,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                "output_grid must preserve the baseline grid prefix."
            )
        if max_sample_length > output_grid.size:
            raise ValueError(
                "output_grid is too short for a translated batch sample."
            )
    return output_grid


def _validate_translation_batch(x_grid, y_density, shifts, boundaries) -> None:
    """Validate the dimensions of the grid, densities, and translation shifts.

    Parameters
    ----------
    x_grid, y_density : numpy.ndarray
        Source grid and one density row per sample.
    shifts : numpy.ndarray
        Translation in meV for each sample and domain.
    boundaries : sequence of tuple of float
        Energy bounds for the domains.
    """
    require_ndim(x_grid, 1, message="x_grid must be one-dimensional.")
    if y_density.ndim != 2 or y_density.shape[1] != x_grid.size:
        raise ValueError(
            "y_density must have shape (n_samples, len(x_grid))."
        )
    if shifts.ndim != 2 or shifts.shape[0] != y_density.shape[0]:
        raise ValueError(
            "shifts must be two-dimensional with one row per density sample."
        )
    if shifts.shape[1] != len(boundaries):
        raise ValueError(
            "The number of spatial-shift columns must match the number of domains."
        )


def _plan_translation_rows(x_grid, y_density, shifts, dx, plan, freeze_origin):
    """Build trusted reconstruction nodes and record each sample's output support.

    Parameters
    ----------
    x_grid, y_density : numpy.ndarray
        Base energy coordinates in eV and sample density rows.
    shifts : numpy.ndarray
        Sample-by-domain translations in meV.
    dx : float
        Mean spacing in eV used to extend the base grid.
    plan : callable
        Translation plan builder configured with the shared domain options.
    freeze_origin : bool
        Keep the first original coordinate anchored.

    Returns
    -------
    tuple
        Pairs of trusted node arrays, output lengths, maximum node count,
        and maximum output length.
    """
    n_samples = y_density.shape[0]
    sample_lengths = np.empty(n_samples, dtype=np.int64)
    node_sets = []
    max_nodes = 0
    max_sample_length = x_grid.size

    # Geometry remains sample dependent because valley locations depend on the
    # amplitude-modified spectrum. Build only the trusted node sets in Python;
    # the expensive cubic reconstruction of all rows is delegated to one
    # parallel Numba kernel below.
    for sample_idx in range(n_samples):
        sample_shifts = shifts[sample_idx]

        if sample_shifts.size == 0 or np.all(np.abs(sample_shifts) < 1e-12):
            x_nodes = x_grid.copy()
            y_nodes = y_density[sample_idx].copy()
        else:
            _, _, diagnostics = plan(
                y_density=y_density[sample_idx], shifts=sample_shifts,
            )
            x_nodes, y_nodes = _translated_trusted_nodes(
                x_grid, y_density[sample_idx], diagnostics, freeze_origin=freeze_origin,
            )

        extension_eV = max(0.0, float(x_nodes[-1] - x_grid[-1]))
        if extension_eV > 0.0:
            n_extra = int(np.ceil(extension_eV / dx)) + 2
        else:
            n_extra = 0

        sample_length = x_grid.size + n_extra
        sample_lengths[sample_idx] = sample_length
        max_sample_length = max(max_sample_length, sample_length)
        max_nodes = max(max_nodes, x_nodes.size)
        node_sets.append((x_nodes, y_nodes))

    return node_sets, sample_lengths, max_nodes, max_sample_length


def _finalize_batch_density(
    x_grid, y_density, output_grid, density_matrix, sample_lengths,
    freeze_origin, preserve_area,
) -> None:
    """Apply the density corrections and zero-fill beyond each sample's support.

    Parameters
    ----------
    x_grid, y_density : numpy.ndarray
        Original energy coordinates and density rows.
    output_grid, density_matrix : numpy.ndarray
        Shared output grid and reconstructed density rows updated in place.
    sample_lengths : numpy.ndarray
        Valid grid prefix length of each row.
    freeze_origin : bool
        Retain the original first density value per row.
    preserve_area : bool
        Apply the existing integral-based density rescaling when True.
    """
    n_samples = y_density.shape[0]
    if freeze_origin:
        density_matrix[:, 0] = y_density[:, 0]

    if preserve_area:
        # Integrate against the physical coordinate grid, accounting for nonuniform spacing.
        area_old = np.trapezoid(y_density, x_grid, axis=1)
        # Integrate against the physical coordinate grid, accounting for nonuniform spacing.
        area_new = np.trapezoid(density_matrix, output_grid, axis=1)
        valid_area = np.abs(area_new) > _EPS
        scale = np.ones(n_samples, dtype=np.float64)
        scale[valid_area] = area_old[valid_area] / area_new[valid_area]

        if freeze_origin:
            density_matrix[:, 1:] *= scale[:, None]
            density_matrix[:, 0] = y_density[:, 0]
        else:
            density_matrix *= scale[:, None]

    columns = np.arange(output_grid.size)[None, :]
    density_matrix[columns >= sample_lengths[:, None]] = 0.0


def apply_spatial_warp_batch(
    x_grid: np.ndarray,
    y_density: np.ndarray,
    boundaries: List[Tuple[float, float]],
    shifts: np.ndarray,
    freeze_origin: bool,
    preserve_shape: bool = True,
    *,
    output_grid: Optional[np.ndarray] = None,
    snap_boundaries_to_valleys: bool = True,
    valley_search_radius_meV: float = DEFAULT_VALLEY_SEARCH_RADIUS_MEV,
    valley_smoothing_meV: float = DEFAULT_VALLEY_SMOOTHING_MEV,
    reconstruction_guard_meV: float = DEFAULT_RECONSTRUCTION_GUARD_MEV,
    shift_guard_factor: float = DEFAULT_SHIFT_GUARD_FACTOR,
    min_guard_points: int = DEFAULT_MIN_GUARD_POINTS,
    preserve_area: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply valley-aware segment translation to a matrix sample batch.

    Parameters
    ----------
    x_grid : numpy.ndarray
        Shared baseline energy grid in eV.
    y_density : numpy.ndarray
        Amplitude-modified density matrix with shape
        ``(n_samples, n_energy)``.
    boundaries : list of tuple of float
        Shared requested source domains in eV.
    shifts : numpy.ndarray
        Per-sample domain translations with shape
        ``(n_samples, n_domains)`` in meV.
    freeze_origin : bool
        Whether the zero-energy point is geometrically anchored.
    preserve_shape : bool, default=True
        Compatibility argument forwarded to the scalar implementation.
    output_grid : numpy.ndarray, optional
        Canonical output grid used by the ensemble. If omitted, the longest
        grid generated by the batch is selected.
    snap_boundaries_to_valleys : bool, default=True
        Whether nominal boundaries should move toward nearby valleys.
    valley_search_radius_meV : float, default=3.0
        Maximum valley-boundary adjustment in meV.
    valley_smoothing_meV : float, default=0.5
        Gaussian smoothing scale used only for valley detection.
    reconstruction_guard_meV : float, default=0.5
        Minimum source-space guard half-width around each valley.
    shift_guard_factor : float, default=0.55
        Factor multiplying neighboring relative shifts when selecting guards.
    min_guard_points : int, default=3
        Minimum number of grid intervals removed around reconstructed boundaries.
    preserve_area : bool, default=False
        Whether each translated spectrum should preserve its source integral.

    Returns
    -------
    output_grid : numpy.ndarray
        Shared batch energy grid in eV.
    density_matrix : numpy.ndarray
        Zero-padded translated density matrix.
    sample_lengths : numpy.ndarray
        Physical support length of every translated sample.

    Notes
    -----
    Valley-aware geometry remains sample dependent, but repeated numerical
    loops are compiled and the final PCHIP reconstruction is evaluated for all
    rows by a parallel Numba kernel. The scalar SciPy implementation remains
    available as the numerical reference path.
    """
    x_grid = np.asarray(x_grid, dtype=np.float64)
    y_density = np.asarray(y_density, dtype=np.float64)
    shifts = np.asarray(shifts, dtype=np.float64)

    _validate_translation_batch(x_grid, y_density, shifts, boundaries)

    plan = partial(
        build_translation_plan,
        x_grid=x_grid,
        boundaries=boundaries,
        snap_boundaries_to_valleys=snap_boundaries_to_valleys,
        valley_search_radius_meV=valley_search_radius_meV,
        valley_smoothing_meV=valley_smoothing_meV,
        reconstruction_guard_meV=reconstruction_guard_meV,
        shift_guard_factor=shift_guard_factor,
        min_guard_points=min_guard_points,
    )
    dx = abs(_mean_dx(x_grid))
    node_sets, sample_lengths, max_nodes, max_sample_length = _plan_translation_rows(
        x_grid, y_density, shifts, dx, plan, freeze_origin,
    )

    output_grid = _resolve_translation_output_grid(x_grid, output_grid, max_sample_length, dx)

    x_nodes_matrix, y_nodes_matrix, node_lengths = _pack_trusted_nodes(node_sets, max_nodes)

    non_negative = np.all(y_density >= -1e-15, axis=1)
    density_matrix = _pchip_interpolate_rows(
        x_nodes=x_nodes_matrix,
        y_nodes=y_nodes_matrix,
        node_lengths=node_lengths,
        x_output=output_grid,
        non_negative=non_negative,
    )

    _finalize_batch_density(
        x_grid, y_density, output_grid, density_matrix,
        sample_lengths, freeze_origin, preserve_area,
    )

    if not np.all(np.isfinite(density_matrix)):
        raise RuntimeError(
            "Internal error: reconstruction produced non-finite density values."
        )

    return output_grid, density_matrix, sample_lengths
