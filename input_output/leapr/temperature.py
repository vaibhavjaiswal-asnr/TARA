"""Temperature selection policy for single-temperature LEAPR sample generation."""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

from common.validation import require_finite_positive
from input_output.leapr.exceptions import LeaprFormatError

if TYPE_CHECKING:
    from .core import LeaprFile


TEMPERATURE_MARGIN_K = 5.0


def available_leapr_temperatures(leapr_file: LeaprFile) -> tuple[float, ...]:
    """Return the physical temperatures recorded in an entire LEAPR template.

    Negative LEAPR temperature cards reuse the preceding phonon spectrum; their
    absolute magnitudes are still physical processing temperatures.

    Parameters
    ----------
    leapr_file : LeaprFile
        Fully parsed original LEAPR input, including negative temperature cards.

    Returns
    -------
    tuple of float
        Unique, ascending physical temperatures in kelvin.

    Raises
    ------
    LeaprFormatError
        If the template has no usable first positive-temperature spectrum or
        contains an invalid temperature card.
    """
    temperatures = _validated_template_temperatures(leapr_file)
    return tuple(sorted({abs(value) for value in temperatures}))


def _validated_template_temperatures(leapr_file: LeaprFile) -> list[float]:
    """Read and validate the signed temperature cards in template order.

    Parameters
    ----------
    leapr_file : LeaprFile
        Parsed template containing the temperature blocks.

    Returns
    -------
    list of float
        Signed temperature values in kelvin.

    Raises
    ------
    LeaprFormatError
        If a card is invalid, a reuse card precedes its source, or no
        positive-temperature spectrum is present.
    """
    temperatures = []
    has_spectrum = False
    for block in leapr_file.temp_blocks:
        try:
            value = float(block["temp"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise LeaprFormatError("Invalid temperature card in LEAPR template.") from exc
        if not math.isfinite(value) or value == 0.0:
            raise LeaprFormatError(f"Invalid LEAPR temperature: {value!r} K.")
        if value > 0.0:
            has_spectrum = True
        elif not has_spectrum:
            raise LeaprFormatError(
                "A negative LEAPR temperature cannot precede the first "
                "positive-temperature phonon spectrum."
            )
        temperatures.append(value)
    if not has_spectrum:
        raise LeaprFormatError("LEAPR template contains no positive-temperature spectrum.")
    return temperatures


def resolve_leapr_temperature(
    leapr_file: LeaprFile,
    temperature: float | None = None,
    *,
    warn_on_reuse: bool = True,
) -> float:
    """Select one output temperature within the template's supported range.

    An interior request may reuse the first positive-temperature spectrum.
    Requests up to five kelvin beyond either boundary also reuse it with a
    warning; larger extrapolations are rejected before NJOY execution.

    Parameters
    ----------
    leapr_file : LeaprFile
        Original parsed LEAPR template, including all negative temperature cards.
    temperature : float or None, optional
        Requested output temperature in kelvin. ``None`` selects the first
        positive-temperature block, preserving the legacy behavior.
    warn_on_reuse : bool, optional
        Emit warnings when the requested temperature differs from the spectrum
        block's temperature. Disable only when a parent batch has warned once.

    Returns
    -------
    float
        Validated, unrounded temperature to encode in the single output block.

    Raises
    ------
    LeaprFormatError
        If the template contains invalid temperature records.
    ValueError
        If the request is not finite and positive, or is more than five kelvin
        beyond the original template's temperature boundaries.
    """
    cards = _validated_template_temperatures(leapr_file)
    original = cards[0]
    requested = original if temperature is None else _validate_requested_temperature(
        temperature, "Requested LEAPR temperature must be positive and finite.",
    )
    physical_temperatures = [abs(value) for value in cards]
    minimum, maximum = min(physical_temperatures), max(physical_temperatures)
    distance_outside = _validate_temperature_range(requested, minimum, maximum)

    if warn_on_reuse and not math.isclose(requested, original, abs_tol=1e-10, rel_tol=0.0):
        message = _reuse_warning_message(requested, minimum, maximum, distance_outside)
        warnings.warn(message, UserWarning, stacklevel=2)
    return requested


def select_leapr_temperature_block(
    leapr_file: LeaprFile,
    temperature: float | None = None,
    *,
    tolerance_k: float = 1e-5,
) -> tuple[dict, float]:
    """Select the physical source block at an explicitly tabulated temperature.

    A negative Card 10 temperature references the preceding positive block;
    it does not supply its own oscillator or diffusion constants. This
    selector never silently substitutes the first template block for a
    different liquid-water temperature.

    Parameters
    ----------
    leapr_file : LeaprFile
        Parsed complete source template.
    temperature : float or None, optional
        Exact requested tabulated temperature; ``None`` uses the first block.
    tolerance_k : float, optional
        Floating-point matching tolerance in kelvin, not an interpolation band.

    Returns
    -------
    tuple of (dict, float)
        Source block containing all physical parameters and selected T in K.

    Raises
    ------
    ValueError
        If a requested temperature is not explicitly tabulated.
    LeaprFormatError
        If a negative temperature appears before any positive block.
    """
    cards = _validated_template_temperatures(leapr_file)
    requested = cards[0] if temperature is None else _validate_requested_temperature(
        temperature, "Requested LEAPR temperature must be finite and positive.",
    )
    source = _find_leapr_temperature_block(leapr_file, cards, requested, tolerance_k)
    return _require_tabulated_temperature(source, requested, cards)


def _find_leapr_temperature_block(
    leapr_file: LeaprFile, cards: list[float], requested: float, tolerance_k: float,
) -> tuple[dict, float] | None:
    """Find the matching card and the positive block that supplies its spectrum.

    Parameters
    ----------
    leapr_file : LeaprFile
        Parsed LEAPR template.
    cards : list of float
        Validated signed temperatures in template order, in kelvin.
    requested : float
        Requested temperature in kelvin.
    tolerance_k : float
        Absolute matching tolerance in kelvin.

    Returns
    -------
    tuple of (dict, float) or None
        Source block and matching physical temperature, if present.
    """
    source = leapr_file.temp_blocks[0]
    for block, value in zip(leapr_file.temp_blocks, cards):
        # A negative Card 10 temperature reuses the last positive block.
        if value > 0.0:
            source = block
        if math.isclose(abs(value), requested, abs_tol=tolerance_k, rel_tol=0.0):
            return source, abs(value)
    return None


def _require_tabulated_temperature(
    match: tuple[dict, float] | None, requested: float, cards: list[float],
) -> tuple[dict, float]:
    """Require a source block at the requested tabulated temperature.

    Parameters
    ----------
    match : tuple of (dict, float) or None
        Matching source block and temperature, if found.
    requested : float
        Requested temperature in kelvin.
    cards : list of float
        Validated signed temperatures in kelvin.

    Returns
    -------
    tuple of (dict, float)
        Matching source block and tabulated physical temperature.

    Raises
    ------
    ValueError
        If no card matches the requested temperature.
    """
    if match is not None:
        return match

    temperatures = tuple(sorted({abs(value) for value in cards}))
    raise ValueError(
        f'LEAPR temperature {requested:g} K is not explicitly tabulated. '
        f'Available temperatures: {temperatures}. '
        'For liquids, select an existing block; physical interpolation is not implemented.'
    )


def _validate_requested_temperature(temperature: float, error_message: str) -> float:
    """Convert the requested temperature and require a finite positive value.

    Parameters
    ----------
    temperature : float
        Requested temperature in kelvin.
    error_message : str
        Error text used by the calling selector.

    Returns
    -------
    float
        Validated temperature in kelvin.

    Raises
    ------
    ValueError
        If the request cannot be converted or is nonpositive or nonfinite.
    """
    try:
        requested = float(temperature)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error_message) from exc
    require_finite_positive(requested, message=error_message)
    return requested


def _validate_temperature_range(requested: float, minimum: float, maximum: float) -> float:
    """Require that the request stays within the permitted template margin.

    Parameters
    ----------
    requested : float
        Requested temperature in kelvin.
    minimum, maximum : float
        Lower and upper template temperatures in kelvin.

    Returns
    -------
    float
        Nonnegative distance outside the template range in kelvin.

    Raises
    ------
    ValueError
        If the distance exceeds the five-kelvin reuse margin.
    """
    distance_outside = max(minimum - requested, requested - maximum, 0.0)
    if distance_outside > TEMPERATURE_MARGIN_K:
        raise ValueError(
            f"Requested LEAPR temperature {requested:g} K is {distance_outside:g} K "
            f"outside the original range [{minimum:g}, {maximum:g}] K "
            f"(maximum permitted overshoot: {TEMPERATURE_MARGIN_K:g} K). "
            "Edit the input LEAPR template explicitly to use this temperature."
        )
    return distance_outside


def _reuse_warning_message(
    requested: float, minimum: float, maximum: float, distance_outside: float
) -> str:
    """Describe the established LEAPR spectrum reuse policy in kelvin.

    Parameters
    ----------
    requested : float
        Selected target temperature in kelvin.
    minimum, maximum : float
        Original template temperature bounds in kelvin.
    distance_outside : float
        Nonnegative overshoot beyond the template bounds, in kelvin.

    Returns
    -------
    str
        Warning text with the historical wording and formatting.
    """
    if distance_outside > 0.0:
        message = (
            f"Reusing the LEAPR phonon spectrum at {requested:g} K, "
            f"which is {distance_outside:g} K outside the original "
            f"temperature boundaries [{minimum:g}, {maximum:g}] K."
        )
    else:
        message = (
            f"Reusing the LEAPR phonon spectrum at {requested:g} K "
            f"because it lies inside the original temperature range "
            f"[{minimum:g}, {maximum:g}] K."
        )
    return message
