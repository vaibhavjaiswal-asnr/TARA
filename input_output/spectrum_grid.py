"""Resample completed TARA spectra onto an input format's native grid.

Core may use dense, nonuniform internal energies. Output formats retain the
original uniform spacing and can extend the *upper* boundary when required by
the actual sampled spectrum. This module does not generate stochastic values.
"""

from __future__ import annotations

import numpy as np

from common.validation import (
    require_finite_array,
    require_ndim,
    require_nonnegative_array,
    require_same_shape,
    require_strictly_increasing,
)


def resample_on_source_spacing(
    reference_energy: np.ndarray,
    sample_energy: np.ndarray,
    sample_density: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one completed sample at the original grid spacing.

    Parameters
    ----------
    reference_energy : array_like
        Original LEAPR/NCMAT VDOS grid, in eV, uniformly spaced and increasing.
        Its existing coordinates, including their rounding, are not rewritten.
    sample_energy : array_like
        Full Core energy grid in eV (possibly with a fine Debye tail and a
        shifted, right-extended upper edge).
    sample_density : array_like
        Sampled density in 1/eV at ``sample_energy``; no normalization is done.

    Returns
    -------
    energy : numpy.ndarray
        The original reference grid with zero or more evenly spaced points
        appended at the upper edge. Original coordinates are unchanged.
    density : numpy.ndarray
        Linear interpolation of the complete sampled VDOS onto that grid.
        Values outside the sampled support are zero, not edge extrapolations.

    Raises
    ------
    ValueError
        On malformed, nonuniform reference grids or invalid sample values.

    Notes
    -----
    The number of extra points is determined by the sample's upper *support
    coordinate*. No low-energy points from Core's Debye reconstruction are
    transmitted to the serialized spectrum unless they fall on the input grid.
    """
    original = np.asarray(reference_energy, dtype=np.float64)
    source = np.asarray(sample_energy, dtype=np.float64)
    values = np.asarray(sample_density, dtype=np.float64)
    reference_error = "Original VDOS grid must be finite, increasing, and contain >=2 points."
    require_ndim(original, 1, message=reference_error)
    if original.size < 2:
        raise ValueError(reference_error)
    require_finite_array(original, message=reference_error)
    require_strictly_increasing(original, message=reference_error)
    spacing = float((original[-1] - original[0]) / (original.size - 1))
    if not np.allclose(np.diff(original), spacing, rtol=1e-6, atol=1e-12):
        raise ValueError("Original output VDOS grid must have uniform spacing.")
    sample_error = "Sampled VDOS grid and density must be finite, valid, and increasing."
    require_ndim(source, 1, message=sample_error)
    require_same_shape(values, source, message=sample_error)
    if source.size < 2:
        raise ValueError(sample_error)
    require_finite_array(source, message=sample_error)
    require_finite_array(values, message=sample_error)
    require_strictly_increasing(source, message=sample_error)
    require_nonnegative_array(values, message=sample_error)
    if original[0] < source[0] - 1e-12:
        raise ValueError("Sampled spectrum does not cover the original low-energy bound.")

    excess = float(source[-1] - original[-1])
    # Extend only the upper edge by whole source steps. The tolerance avoids adding a
    # node for roundoff.
    n_extra = max(0, int(np.ceil((excess - 1e-12) / spacing)))
    if n_extra:
        extra_energy = original[-1] + spacing * np.arange(1, n_extra + 1)
        target = np.concatenate((original, extra_energy))
    else:
        target = original.copy()
    output = np.interp(target, source, values, left=0.0, right=0.0)
    return target, output
