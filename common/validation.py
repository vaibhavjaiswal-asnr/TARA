"""Validate scalar, array, and path arguments shared across TARA domains."""

import math
from pathlib import Path

import numpy as np


def require_finite_positive(
    value: float, *, name: str = "value", message: str | None = None,
) -> None:
    """Require a finite scalar greater than zero.

    Parameters
    ----------
    value : float
        Scalar to validate.
    name : str, optional
        Argument name used in the default diagnostic.
    message : str or None, optional
        Existing caller diagnostic to preserve.

    Raises
    ------
    ValueError
        If the value is nonfinite or nonpositive.
    """
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(message if message is not None else f"{name} must be finite and positive.")


def require_finite_nonnegative(
    value: float, *, name: str = "value", message: str | None = None,
) -> None:
    """Require a finite scalar greater than or equal to zero.

    Parameters
    ----------
    value : float
        Scalar to validate.
    name : str, optional
        Argument name used in the default diagnostic.
    message : str or None, optional
        Existing caller diagnostic to preserve.

    Raises
    ------
    ValueError
        If the value is nonfinite or negative.
    """
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(message if message is not None else f"{name} must be finite and nonnegative.")


def require_ndim(
    array: np.ndarray, ndim: int, *, name: str = "array", message: str | None = None,
) -> None:
    """Require the expected array rank.

    Parameters
    ----------
    array : numpy.ndarray
        Array to validate.
    ndim : int
        Required number of dimensions.
    name : str, optional
        Argument name used in the default diagnostic.
    message : str or None, optional
        Existing caller diagnostic to preserve.

    Raises
    ------
    ValueError
        If the number of dimensions differs from ``ndim``.
    """
    if array.ndim != ndim:
        raise ValueError(message if message is not None else f"{name} must have {ndim} dimensions.")


def require_same_shape(
    first: np.ndarray,
    second: np.ndarray,
    *,
    names: tuple[str, str] = ("first array", "second array"),
    message: str | None = None,
) -> None:
    """Require two arrays to have the same shape.

    Parameters
    ----------
    first, second : numpy.ndarray
        Arrays to compare.
    names : tuple of str, optional
        Argument names used in the default diagnostic.
    message : str or None, optional
        Existing caller diagnostic to preserve.

    Raises
    ------
    ValueError
        If the shapes differ.
    """
    if first.shape != second.shape:
        raise ValueError(message if message is not None else f"{names[0]} and {names[1]} must have the same shape.")


def require_finite_array(
    array: np.ndarray, *, name: str = "array", message: str | None = None,
) -> None:
    """Require every entry in a numeric array to be finite.

    Parameters
    ----------
    array : numpy.ndarray
        Numeric array to validate.
    name : str, optional
        Argument name used in the default diagnostic.
    message : str or None, optional
        Existing caller diagnostic to preserve.

    Raises
    ------
    ValueError
        If any entry is nonfinite.
    """
    if not np.isfinite(array).all():
        raise ValueError(message if message is not None else f"{name} must contain only finite values.")


def require_nonnegative_array(
    array: np.ndarray, *, name: str = "array", message: str | None = None,
) -> None:
    """Reject negative entries in a real-valued array.

    Parameters
    ----------
    array : numpy.ndarray
        Numeric array to check. Apply a separate finiteness check when required.
    name : str, optional
        Argument name used in the default diagnostic.
    message : str or None, optional
        Existing caller diagnostic to preserve.

    Raises
    ------
    ValueError
        If any entry is negative.
    """
    if np.any(array < 0.0):
        raise ValueError(message if message is not None else f"{name} must contain no negative values.")


def require_strictly_increasing(
    grid: np.ndarray, *, name: str = "grid", message: str | None = None,
) -> None:
    """Require every coordinate to exceed the preceding coordinate.

    Parameters
    ----------
    grid : numpy.ndarray
        One-dimensional numeric coordinates. The caller checks rank,
        finiteness, and any minimum point count needed by its operation.
    name : str, optional
        Argument name used in the default diagnostic.
    message : str or None, optional
        Existing caller diagnostic to preserve.

    Raises
    ------
    ValueError
        If adjacent coordinates are repeated or unordered.
    """
    if not np.all(grid[1:] > grid[:-1]):
        raise ValueError(message if message is not None else f"{name} must be strictly increasing.")


def require_existing_file(
    path: str | Path, *, name: str = "path", message: str | None = None,
) -> None:
    """Require a path to point to an existing file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to check. A symbolic link to an existing file is accepted.
    name : str, optional
        Argument name used in the default diagnostic.
    message : str or None, optional
        Existing caller diagnostic to preserve.

    Raises
    ------
    FileNotFoundError
        If the path is missing or does not identify a file.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(message if message is not None else f"{name} does not identify a file: {path}")
