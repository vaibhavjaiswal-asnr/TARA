"""Read the array payload of a native liquid scattering capture."""

from pathlib import Path
from typing import Any

import numpy as np


def read_sabtap_arrays(
    json_path: Path, record: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read dimensionless alpha, beta, and S from a parsed capture record.

    Parameters
    ----------
    json_path : pathlib.Path
        Record path used to locate an adjacent binary payload.
    record : dict of str to Any
        Parsed SABData record with inline arrays or a binary-file reference.

    Returns
    -------
    alpha, beta, sab : numpy.ndarray
        One-dimensional transfer grids and a beta-major scattering table
        with shape ``(beta.size, alpha.size)``.

    Raises
    ------
    ValueError
        If the payload dimensions disagree with the declared grids.
    """
    if "binary_file" in record:
        return _read_binary_payload(json_path, record)

    alpha = np.asarray(record["alpha"], dtype=np.float64)
    beta = np.asarray(record["beta"], dtype=np.float64)
    sab = np.asarray(record["sab"], dtype=np.float64)
    if sab.shape != (len(beta), len(alpha)):
        raise ValueError("SABData JSON table shape does not match grids")
    return alpha, beta, sab


def _read_binary_payload(
    json_path: Path, record: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a native float64 payload into its consecutive array sections.

    Parameters
    ----------
    json_path : pathlib.Path
        Capture record path.
    record : dict of str to Any
        Header declaring the binary file and both grid lengths.

    Returns
    -------
    alpha, beta, sab : numpy.ndarray
        Transfer grids and scattering values in beta-major order.

    Raises
    ------
    ValueError
        If the declared grid lengths are nonpositive or the file size is
        inconsistent with those lengths.
    """
    n_alpha = int(record["n_alpha"])
    n_beta = int(record["n_beta"])
    if n_alpha <= 0 or n_beta <= 0:
        raise ValueError("SABData binary grid counts must be positive")

    grid_end = n_alpha + n_beta
    binary_path = json_path.parent / record["binary_file"]
    expected_values = grid_end + n_alpha * n_beta
    if binary_path.stat().st_size != expected_values * np.dtype(np.float64).itemsize:
        raise ValueError("SABData binary payload size mismatch")
    payload = np.fromfile(binary_path, dtype=np.float64)
    if payload.size != expected_values:
        raise ValueError("SABData binary payload size mismatch")

    # The writer stores the two grids first, followed by contiguous beta rows.
    return payload[:n_alpha], payload[n_alpha:grid_end], payload[grid_end:].reshape(n_beta, n_alpha)
