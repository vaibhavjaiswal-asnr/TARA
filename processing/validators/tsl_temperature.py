"""Read thermal-scattering temperatures from ENDF-6 MF7/MT4 records.

The first beta table contains the primary temperature (TAB1) followed by
optional additional temperatures (LIST).  Reading this small fixed-width
record sequence avoids parsing the entire, potentially very large, S(a,b)
file for a single temperature preflight.
"""

from __future__ import annotations

from collections.abc import Iterator
import math
from pathlib import Path
import re

from common.validation import require_finite_positive


_ENDF_EXPONENT = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))([+-]\d+)$")


def _number(field: str) -> float:
    """Convert an ENDF eleven-column numeric field to a float.

    Parameters
    ----------
    field : str
        Numeric field, including ENDF's implicit exponent notation.

    Returns
    -------
    float
        Parsed numerical value.
    """
    value = field.strip().replace("D", "E").replace("d", "E")
    if "E" not in value and "e" not in value:
        match = _ENDF_EXPONENT.fullmatch(value)
        if match is not None:
            value = f"{match.group(1)}E{match.group(2)}"
    return float(value)


def _field(record: str, index: int) -> float:
    """Extract one of the six standard ENDF record fields.

    Parameters
    ----------
    record : str
        Fixed-width ENDF record.
    index : int
        Zero-based field position, from zero to five.

    Returns
    -------
    float
        Parsed field value.
    """
    return _number(record[index * 11:(index + 1) * 11])


def _section_records(path: Path) -> Iterator[str]:
    """Yield records of the first MF7/MT4 section only.

    Parameters
    ----------
    path : pathlib.Path
        Thermal-scattering ENDF file.

    Yields
    ------
    str
        One fixed-width MF7/MT4 record.

    Raises
    ------
    ValueError
        If no MF7/MT4 section can be found.
    """
    found = False
    with path.open("r", encoding="ascii", errors="replace") as stream:
        for record in stream:
            if len(record) < 75:
                continue
            try:
                mf, mt = int(record[70:72]), int(record[72:75])
            except ValueError:
                continue
            if mf == 7 and mt == 4:
                found = True
                yield record
            elif found:
                break
    if not found:
        raise ValueError(f"No ENDF MF7/MT4 thermal-scattering section in {path}.")


def _next_record(records: Iterator[str], path: Path) -> str:
    """Read one required MF7/MT4 record or reject a truncated section.

    Parameters
    ----------
    records : iterator of str
        Active thermal-section record iterator.
    path : pathlib.Path
        Source ENDF file for the error message.

    Returns
    -------
    str
        Next record.

    Raises
    ------
    ValueError
        If a mandatory ENDF record is missing.
    """
    try:
        return next(records)
    except StopIteration as exc:
        raise ValueError(f"Truncated MF7/MT4 temperature data in {path}.") from exc


def _skip_values(records: Iterator[str], count: int, path: Path) -> None:
    """Advance past a LIST/TAB payload of six values per ENDF line.

    Parameters
    ----------
    records : iterator of str
        Section records being parsed.
    count : int
        Number of scalar values in the payload.
    path : pathlib.Path
        Input ENDF path for diagnostics.

    Raises
    ------
    ValueError
        If the payload size is negative or records are missing.
    """
    if count < 0:
        raise ValueError(f"Negative MF7/MT4 array length in {path}.")
    for _ in range((count + 5) // 6):
        _next_record(records, path)


def read_tsl_temperatures(file_path: str) -> tuple[float, ...]:
    """Read physical temperatures represented in ENDF MF7/MT4.

    The first beta block uses ``TAB1(T0, beta, LT, ..., NR, NP)`` followed
    by ``LT`` ``LIST(Ti, beta, LI, ..., NP, 0)`` records.  Temperature
    selection therefore does not depend on an unrelated LEAPR template.

    Parameters
    ----------
    file_path : str
        Path to a generated or externally supplied ENDF-6 TSL evaluation.

    Returns
    -------
    tuple of float
        Available physical temperatures in kelvin, in ascending order.

    Raises
    ------
    FileNotFoundError
        If the ENDF file does not exist or is empty.
    ValueError
        If the MF7/MT4 first beta-temperature records are invalid.
    """
    path = Path(file_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty TSL ENDF: {path}")
    records = _section_records(path)
    try:
        return _read_first_beta_temperatures(records, path)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot read TSL temperatures from {path}: {exc}") from exc
    finally:
        records.close()


def _read_first_beta_temperatures(records: Iterator[str], path: Path) -> tuple[float, ...]:
    """Read the first beta table while preserving ENDF record consumption.

    Parameters
    ----------
    records : Iterator[str]
        Generator yielding the original fixed-width MF7/MT4 records.
    path : pathlib.Path
        Source path for diagnostics.

    Returns
    -------
    tuple of float
        Unique physical temperatures in kelvin, in ascending order.

    Notes
    -----
    The caller retains ownership of the section generator and closes it.
    """
    _next_record(records, path)  # MF7/MT4 HEAD.
    b_header = _next_record(records, path)  # LIST defining B(N).
    _skip_values(records, int(_field(b_header, 4)), path)
    beta_header = _next_record(records, path)  # TAB2 interpolation grid.
    beta_ranges, beta_count = int(_field(beta_header, 4)), int(_field(beta_header, 5))
    if beta_count < 1:
        raise ValueError(f"MF7/MT4 has no tabulated beta values in {path}.")
    # Each TAB2 interpolation range contributes two fields before the first beta table.
    _skip_values(records, 2 * beta_ranges, path)
    first_beta = _next_record(records, path)  # TAB1 T0 and LT.
    t0, extra_count = _field(first_beta, 0), int(_field(first_beta, 2))
    if extra_count < 0:
        raise ValueError(f"Invalid MF7/MT4 LT={extra_count} in {path}.")
    temperatures = [t0]
    if extra_count:
        temperatures.extend(_read_extra_beta_temperatures(records, path, first_beta, extra_count))
    if not all(math.isfinite(t) and t > 0.0 for t in temperatures):
        raise ValueError(f"Invalid MF7/MT4 temperature values in {path}: {temperatures}.")
    return tuple(sorted(set(temperatures)))


def _read_extra_beta_temperatures(records, path, first_beta, extra_count):
    """Read the additional temperature LIST records from the first beta table.

    Parameters
    ----------
    records : iterator of str
        MF7/MT4 record stream after the first TAB1 header.
    path : pathlib.Path
        Source file for diagnostics.
    first_beta : str
        TAB1 header giving the interpolation and alpha counts.
    extra_count : int
        Number of additional temperature records declared by LT.

    Returns
    -------
    list of float
        Additional physical temperatures in kelvin, in file order.
    """
    temperatures = []
    alpha_ranges = int(_field(first_beta, 4))
    alpha_count = int(_field(first_beta, 5))
    # Each TAB1 interpolation pair and each alpha/value pair occupy two fields.
    _skip_values(records, 2 * (alpha_ranges + alpha_count), path)
    for _ in range(extra_count):
        extra = _next_record(records, path)
        temperatures.append(_field(extra, 0))
        _skip_values(records, int(_field(extra, 4)), path)
    return temperatures


def validate_tsl_temperature(
    file_path: str, requested_temperature: float, *, sample_id: str = "sample_000"
) -> tuple[float, ...]:
    """Require that one reference TSL evaluation supports the ACE temperature.

    Parameters
    ----------
    file_path : str
        Representative ENDF MF7/MT4 file.
    requested_temperature : float
        Desired NJOY THERMR/ACER temperature in kelvin.
    sample_id : str, optional
        Identifier reported on a mismatch.

    Returns
    -------
    tuple of float
        Temperatures found in the reference evaluation.

    Raises
    ------
    ValueError
        If the requested temperature is not encoded in the TSL evaluation.
    """
    requested = float(requested_temperature)
    require_finite_positive(requested, message="ACE temperature must be finite and positive (kelvin).")
    available = read_tsl_temperatures(file_path)
    if not any(math.isclose(t, requested, rel_tol=1e-4, abs_tol=1e-4) for t in available):
        raise ValueError(
            f"TSL temperature mismatch for {sample_id}: ACE requests "
            f"{requested:g} K but {file_path} contains {list(available)} K. "
            "NJOY THERMR cannot use a different TSL temperature. "
            "Regenerate the ENDF ensemble with "
            "generate_ensemble_sab_tapes(..., temperature=ACE_TEMPERATURE) "
            "and use the same value for ACE; if that argument was already "
            "supplied, verify the installed TARA source and regenerated files."
        )
    return available
