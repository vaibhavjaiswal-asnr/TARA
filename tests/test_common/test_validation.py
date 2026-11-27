"""Exercise the public validation helpers used by TARA domains."""

import numpy as np
import pytest

from common.validation import (
    require_existing_file,
    require_finite_array,
    require_finite_nonnegative,
    require_finite_positive,
    require_ndim,
    require_nonnegative_array,
    require_same_shape,
    require_strictly_increasing,
)


def test_scalar_validators():
    """Accept valid bounds and reject negative or nonfinite scalars."""
    require_finite_positive(295.0)
    require_finite_nonnegative(0.0)
    with pytest.raises(ValueError, match="temperature must be finite and positive"):
        require_finite_positive(0.0, name="temperature")
    for value in (-1.0, np.inf, -np.inf, np.nan):
        with pytest.raises(ValueError):
            require_finite_positive(value)
        with pytest.raises(ValueError):
            require_finite_nonnegative(value)


def test_array_shape_validators():
    """Require matching shapes independently of the requested array rank."""
    energy = np.array([0.0, 1.0])
    require_ndim(energy, 1)
    require_same_shape(energy, energy.copy())
    with pytest.raises(ValueError, match="energy must have 2 dimensions"):
        require_ndim(energy, 2, name="energy")
    with pytest.raises(ValueError, match="energy and density must have the same shape"):
        require_same_shape(energy, energy.reshape(2, 1), names=("energy", "density"))


def test_array_value_validators():
    """Check finite values and signs without modifying the inputs."""
    density = np.array([0.0, 1.0])
    require_finite_array(density)
    require_nonnegative_array(density)
    np.testing.assert_array_equal(density, [0.0, 1.0])
    for value in (np.inf, -np.inf, np.nan):
        with pytest.raises(ValueError, match="Legacy diagnostic"):
            require_finite_array(np.array([value]), message="Legacy diagnostic")
    with pytest.raises(ValueError):
        require_nonnegative_array(np.array([-1.0, 0.0]))


def test_increasing_grid():
    """Reject duplicates and decreases, including unsigned coordinates."""
    require_strictly_increasing(np.array([0, 1, 2], dtype=np.uint64))
    require_strictly_increasing(np.array([0.0]))
    for grid in (np.array([1, 0], dtype=np.uint64), np.array([1.0, 1.0])):
        with pytest.raises(ValueError, match="strictly increasing"):
            require_strictly_increasing(grid)


def test_existing_file(tmp_path):
    """Accept an existing file and reject missing paths and directories.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for testing path validation.
    """
    source = tmp_path / "input.leapr"
    source.touch()
    require_existing_file(str(source))
    for path in (tmp_path, tmp_path / "missing.leapr"):
        with pytest.raises(FileNotFoundError, match="LEAPR template unavailable"):
            require_existing_file(path, message="LEAPR template unavailable")
