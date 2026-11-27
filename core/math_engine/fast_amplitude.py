"""
Fast amplitude perturbation for TARA stochastic sampling.

This module applies requested amplitude shifts directly to the baseline spectrum
inside each domain and uses compact C2 smootherstep transitions at internal
boundaries. The topology-dependent weight matrix is cached so complete sample
batches can be evaluated through one dense matrix multiplication.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from common.validation import require_ndim, require_same_shape


class FastAmplitudePerturber:
    """
    Apply direct domain scaling with cached smooth boundary transitions.

    Parameters
    ----------
    x_grid : numpy.ndarray
        Monotonically increasing energy grid.
    y_density : numpy.ndarray
        Baseline density sampled on ``x_grid``.
    transition_points : int, default=8
        Approximate total number of grid points occupied by each smooth
        transition. The window is centered on each internal domain boundary.

    Notes
    -----
    For a fixed boundary topology, a weight matrix ``W`` is built once such
    that a complete stochastic batch is evaluated as

    ``multipliers = 1 + shifts @ W.T``.

    If ``shifts`` has shape ``(n_samples, n_domains)`` and ``W`` has shape
    ``(n_energy, n_domains)``, the resulting multiplier matrix has shape
    ``(n_samples, n_energy)``.
    """

    def __init__(
        self,
        x_grid: np.ndarray,
        y_density: np.ndarray,
        transition_points: int = 8,
    ) -> None:
        """Initialize the direct amplitude operator and its topology cache.

        Parameters
        ----------
        x_grid : np.ndarray
            Input coordinate grid.
        y_density : np.ndarray
            Spectral density sampled on the input grid.
        transition_points : int, optional
            Number of points in each boundary blend. Default is 8.

        Raises
        ------
        ValueError
            If the arrays are incompatible, the grid is nonmonotonic, or the
            transition uses fewer than two points.
        """
        self.x_grid = np.asarray(x_grid, dtype=np.float64)
        self.y_density = np.asarray(y_density, dtype=np.float64)

        shape_error = "x_grid and y_density must be one-dimensional arrays."
        require_ndim(self.x_grid, 1, message=shape_error)
        require_ndim(self.y_density, 1, message=shape_error)
        require_same_shape(self.x_grid, self.y_density, message="x_grid and y_density must have the same shape.")
        # Examine adjacent grid differences to enforce ordering or locate transitions.
        if self.x_grid.size > 1 and np.any(np.diff(self.x_grid) <= 0.0):
            raise ValueError("x_grid must be strictly increasing.")
        if transition_points < 2:
            raise ValueError("transition_points must be >= 2.")

        self.transition_points = int(transition_points)
        self._weight_cache: Dict[
            Tuple[Tuple[float, float], ...],
            np.ndarray,
        ] = {}

    @staticmethod
    def _smootherstep(t: np.ndarray) -> np.ndarray:
        """
        Evaluate the quintic C2 smootherstep polynomial.

        Parameters
        ----------
        t : numpy.ndarray
            Normalized coordinates in ``[0, 1]``.

        Returns
        -------
        numpy.ndarray
            Smooth transition values ``6t^5 - 15t^4 + 10t^3``.
        """
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @staticmethod
    def _cache_key(
        boundaries: List[Tuple[float, float]],
    ) -> Tuple[Tuple[float, float], ...]:
        """Return an immutable cache key for one domain topology.

        Parameters
        ----------
        boundaries : List[Tuple[float, float]]
            Spectral-domain boundaries used by the transformation.

        Returns
        -------
        Tuple[Tuple[float, float], ...]
            Pairs of energy bounds in their original domain order.
        """
        return tuple((float(start), float(end)) for start, end in boundaries)

    def _build_weights(
        self,
        boundaries: List[Tuple[float, float]],
    ) -> np.ndarray:
        """
        Build per-grid domain weights for one boundary topology.

        Parameters
        ----------
        boundaries : list of tuple of float
            Energy-domain boundaries in eV.

        Returns
        -------
        numpy.ndarray
            Weight matrix with shape ``(n_energy, n_domains)``.
        """
        n_domains = len(boundaries)
        n_points = self.x_grid.size

        if n_domains == 0:
            return np.empty((n_points, 0), dtype=np.float64)

        weights = np.zeros((n_points, n_domains), dtype=np.float64)

        # Exact bulk assignment: each grid point receives its domain shift.
        for domain_idx, (start, end) in enumerate(boundaries):
            if domain_idx == n_domains - 1:
                mask = (self.x_grid >= start) & (self.x_grid <= end)
            else:
                mask = (self.x_grid >= start) & (self.x_grid < end)
            weights[mask, domain_idx] = 1.0

        self._blend_boundary_weights(weights, boundaries)
        return weights

    def _blend_boundary_weights(
        self, weights: np.ndarray, boundaries: List[Tuple[float, float]],
    ) -> None:
        """Blend neighboring domain weights around each internal boundary.

        Parameters
        ----------
        weights : numpy.ndarray
            Grid-by-domain matrix to update in place.
        boundaries : list of tuple of float
            Lower and upper energy bounds of each domain in eV.
        """
        n_points, n_domains = weights.shape
        # Smooth only compact neighborhoods around internal boundaries.
        half_left = self.transition_points // 2
        half_right = self.transition_points - half_left

        for domain_idx in range(n_domains - 1):
            boundary = float(boundaries[domain_idx][1])
            # Center the blend on the first node at or above the boundary.
            center_idx = int(np.searchsorted(self.x_grid, boundary, side="left"))

            start_idx = max(0, center_idx - half_left)
            end_idx = min(n_points, center_idx + half_right)
            actual_n = end_idx - start_idx

            if actual_n < 2:
                continue

            t = np.linspace(0.0, 1.0, actual_n, dtype=np.float64)
            right_weight = self._smootherstep(t)
            left_weight = 1.0 - right_weight

            # Reset this local window so it remains a two-domain blend.
            weights[start_idx:end_idx, :] = 0.0
            weights[start_idx:end_idx, domain_idx] = left_weight
            weights[start_idx:end_idx, domain_idx + 1] = right_weight

    def _get_weights(
        self,
        boundaries: List[Tuple[float, float]],
    ) -> np.ndarray:
        """Return the cached topology weight matrix, building it if required.

        Parameters
        ----------
        boundaries : List[Tuple[float, float]]
            Spectral-domain boundaries used by the transformation.

        Returns
        -------
        np.ndarray
            Cached grid-by-domain weight matrix.
        """
        key = self._cache_key(boundaries)
        weights = self._weight_cache.get(key)
        if weights is None:
            weights = self._build_weights(boundaries)
            self._weight_cache[key] = weights
        return weights

    def apply_batch(
        self,
        boundaries: List[Tuple[float, float]],
        shifts: np.ndarray,
    ) -> np.ndarray:
        """
        Apply fractional amplitude shifts to a complete sample batch.

        Parameters
        ----------
        boundaries : list of tuple of float
            Energy-domain boundaries in eV.
        shifts : numpy.ndarray
            Fractional amplitude shifts with shape
            ``(n_samples, n_domains)``.

        Returns
        -------
        numpy.ndarray
            Perturbed density matrix with shape ``(n_samples, n_energy)``.

        Raises
        ------
        ValueError
            If ``shifts`` is not two-dimensional or its domain dimension does
            not match ``boundaries``.
        """
        shifts = np.asarray(shifts, dtype=np.float64)

        require_ndim(shifts, 2, message="Batch amplitude shifts must be two-dimensional.")
        if shifts.shape[1] != len(boundaries):
            raise ValueError(
                "The number of amplitude-shift columns must match the number "
                "of domains."
            )

        n_samples = shifts.shape[0]
        if len(boundaries) == 0 or not np.any(shifts):
            return np.broadcast_to(
                self.y_density,
                (n_samples, self.y_density.size),
            ).copy()

        weights = self._get_weights(boundaries)

        # Evaluate every sample-domain combination in one BLAS-backed matrix
        # multiplication instead of calling ``weights @ shift`` per sample.
        multipliers = 1.0 + shifts @ weights.T
        return multipliers * self.y_density[None, :]

    def apply(
        self,
        boundaries: List[Tuple[float, float]],
        shifts: np.ndarray,
    ) -> np.ndarray:
        """
        Apply fractional amplitude shifts to one sample.

        Parameters
        ----------
        boundaries : list of tuple of float
            Energy-domain boundaries in eV.
        shifts : numpy.ndarray
            One fractional amplitude shift per domain.

        Returns
        -------
        numpy.ndarray
            Perturbed one-dimensional density array.
        """
        shifts = np.asarray(shifts, dtype=np.float64)
        require_ndim(shifts, 1, message="Single-sample amplitude shifts must be one-dimensional.")

        # Keep the scalar API as a compatibility wrapper around the optimized
        # matrix implementation so both paths share exactly the same arithmetic.
        return self.apply_batch(
            boundaries=boundaries,
            shifts=shifts[None, :],
        )[0]
