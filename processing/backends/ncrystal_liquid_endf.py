"""Use NCrystal's own EndfFile writer with captured Liquids-plugin SABData.

Target: unmodified NCrystal 4.4.6; capture from legacy SABScatter path.
This adapter does not generate raw ENDF records manually. It populates the
Python data contract expected by ``NCrystal._ncmat2endf_impl.EndfFile`` and
delegates MF1/MF7 serialization to NCrystal and ``endf-parserpy``.

By default the liquid exporter uses the solid exporter's symmetric ENDF
representation, obtained from the captured nonpositive-beta branch.
The original signed kernel remains available for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from common.validation import require_finite_nonnegative, require_finite_positive
from processing.backends.liquid_capture.payload import read_sabtap_arrays


DEFAULT_LIQUID_EMAX_EV = 0.1
DEFAULT_LIQUID_EPSILON = 20.0
DEFAULT_LIQUID_VDOSLUX = 0


T0_K = 293.6  # NCrystal's LAT=1 reference temperature in kelvin.
SUPPORTED_NCRYSTAL_PREFIX = "4.4.6"
ADAPTER_REVISION = "2026-11-17-v2-symmetric-upstream-EndfFile"


@dataclass(frozen=True)
class LiquidKernel:
    """Validated in-memory liquid scattering kernel for upstream ENDF export.

    Parameters
    ----------
    alpha : numpy.ndarray
        1-D dimensionless momentum-transfer grid (neutron-mass convention).
    beta : numpy.ndarray
        1-D signed dimensionless energy-transfer grid containing zero.
    sab_beta_alpha : numpy.ndarray
        2-D scattering law array of shape ``(len(beta), len(alpha))``.
    mass_amu : float
        Target atomic mass in atomic mass units.
    bound_xs_barn : float
        Bound scattering cross section in barns.
    temperature_K : float
        Material temperature in kelvin.
    suggested_emax_eV : float
        Maximum supported incident neutron energy in electronvolts.
    """

    # Scattering-law coordinates and values.
    alpha: np.ndarray
    beta: np.ndarray
    sab_beta_alpha: np.ndarray

    # Principal scatterer and capture limits.
    mass_amu: float
    bound_xs_barn: float
    temperature_K: float
    suggested_emax_eV: float


def _positive_float(name: str, value: Any) -> float:
    """Validate and convert a mandatory positive finite scalar parameter.

    Parameters
    ----------
    name : str
        Parameter name for diagnostic error messages.
    value : Any
        Input scalar value to validate.

    Returns
    -------
    float
        Validated positive floating-point number.

    Raises
    ------
    ValueError
        If ``value`` is ``None``, non-numeric, non-finite, or non-positive.
    """
    if value is None:
        raise ValueError(f"{name} must be supplied explicitly")
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    require_finite_positive(x, message=f"{name} must be a finite positive number, got {value!r}")
    return x


def normalise_capture(data: Dict[str, Any]) -> LiquidKernel:
    """Validate an ``extractKnl``-compatible capture dictionary.

    Parameters
    ----------
    data : dict of str to Any
        Dictionary containing ``alpha``, ``beta``, ``sab``, ``mass_amu``,
        ``scatxs``, ``temperature``, and ``suggested_emax``.

    Returns
    -------
    LiquidKernel
        Validated, immutable kernel record with 2-D ``(beta, alpha)`` layout.

    Raises
    ------
    ValueError
        If required keys are missing or grids/values violate physical contracts.
    """
    required = {
        "alpha",
        "beta",
        "sab",
        "mass_amu",
        "scatxs",
        "temperature",
        "suggested_emax",
    }
    if not isinstance(data, dict) or not required <= data.keys():
        raise ValueError(f"Capture must be a dictionary with keys {sorted(required)}")

    a = np.array(data["alpha"], dtype=np.float64, copy=True)
    b = np.array(data["beta"], dtype=np.float64, copy=True)
    s = np.array(data["sab"], dtype=np.float64, copy=True)
    if a.ndim != 1 or b.ndim != 1 or a.size < 2 or b.size < 3:
        raise ValueError("Alpha/beta must be 1D grids with at least 2/3 points")
    if (
        s.ndim not in (1, 2)
        or s.size != a.size * b.size
        or (s.ndim == 2 and s.shape != (b.size, a.size))
    ):
        raise ValueError("SAB size inconsistent with beta-major alpha-fastest layout")

    s = s.reshape((b.size, a.size))
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(np.isfinite(s))):
        raise ValueError("Non-finite values in captured kernel")
    if a[0] < 0 or np.any(np.diff(a) <= 0) or np.any(np.diff(b) <= 0):
        raise ValueError("Alpha/beta grids must be strictly increasing, alpha >= 0")
    if not (b[0] < 0 < b[-1] and np.any(b == 0)):
        raise ValueError("This adapter requires beta to span both signs and include zero")
    if np.any(s < 0):
        raise ValueError("Negative SAB data: no silent clipping is permitted")

    return LiquidKernel(
        alpha=a,
        beta=b,
        sab_beta_alpha=s,
        mass_amu=_positive_float("mass_amu", data["mass_amu"]),
        bound_xs_barn=_positive_float("scatxs", data["scatxs"]),
        temperature_K=_positive_float("temperature", data["temperature"]),
        suggested_emax_eV=_positive_float("suggested_emax", data["suggested_emax"]),
    )


def _pair_signed_beta_indices(b: np.ndarray) -> list[Tuple[int, int]]:
    """Locate matching ``(-beta, +beta)`` index pairs on a signed beta grid.

    Parameters
    ----------
    b : numpy.ndarray
        Strictly increasing 1-D signed beta grid.

    Returns
    -------
    list of tuple of int
        Pairs of ``(neg_index, pos_index)`` where ``b[neg_index] ≈ -b[pos_index]``.
    """
    good_neg = np.flatnonzero(b < 0)
    pos = np.flatnonzero(b > 0)
    aligned: list[Tuple[int, int]] = []
    if not good_neg.size:
        return aligned
    for ip in pos:
        j = int(np.argmin(np.abs(b[good_neg] + b[ip])))
        neg = int(good_neg[j])
        if np.isclose(b[neg], -b[ip], rtol=2e-7, atol=2e-10):
            aligned.append((neg, int(ip)))
    return aligned


def detailed_balance_report(
    k: LiquidKernel,
    signal_floor: float = 1e-10,
) -> Dict[str, Any]:
    """Compute diagnostic detailed-balance residuals across paired beta nodes.

    Parameters
    ----------
    k : LiquidKernel
        Captured liquid scattering kernel.
    signal_floor : float, default=1e-10
        Relative amplitude threshold below which near-zero tails are excluded.

    Returns
    -------
    dict of str to Any
        Summary counts and median/99th-percentile/max absolute log residuals.
    """
    b, s = k.beta, k.sab_beta_alpha
    aligned = _pair_signed_beta_indices(b)
    if not aligned:
        return {
            "paired_beta": 0,
            "points": 0,
            "median_abs_log": None,
            "p99_abs_log": None,
            "max_abs_log": None,
        }

    residuals: list[float] = []
    peak = np.maximum(s.max(axis=0), 1e-300)
    for neg, ip in aligned:
        low, high = s[neg, :], s[ip, :]
        amp = np.maximum(low, high)
        mask = (low > 0) & (high > 0) & (amp > signal_floor * peak)
        if np.any(mask):
            r = np.log(low[mask]) - np.log(high[mask]) - b[ip]
            residuals.extend(np.abs(r).tolist())

    if not residuals:
        return {
            "paired_beta": len(aligned),
            "points": 0,
            "median_abs_log": None,
            "p99_abs_log": None,
            "max_abs_log": None,
        }
    v = np.asarray(residuals, dtype=np.float64)
    return {
        "paired_beta": len(aligned),
        "points": int(len(v)),
        "median_abs_log": float(np.median(v)),
        "p99_abs_log": float(np.quantile(v, 0.99)),
        "max_abs_log": float(np.max(v)),
    }


def _import_upstream(version_prefix: str) -> Tuple[Any, Any, Any, str]:
    """Import NCrystal's internal ENDF builder and verify its exact version.

    Parameters
    ----------
    version_prefix : str
        Required NCrystal version string (e.g., ``"4.4.6"``).

    Returns
    -------
    NC, upstream, EndfMetaData, actual_version : tuple
        Loaded NCrystal modules, metadata class, and verified version string.

    Raises
    ------
    RuntimeError
        If the installed NCrystal version or internal API does not match.
    """
    import NCrystal as NC
    from NCrystal import _ncmat2endf_impl as upstream
    from NCrystal.ncmat2endf import EndfMetaData

    actual = str(upstream.nc_core.get_version())
    if actual != version_prefix:
        raise RuntimeError(
            f"NCrystal {actual!r} is installed, but the adapter was checked "
            f"against {version_prefix!r}. Update/test the adapter contract "
            "explicitly instead of bypassing this compatibility check."
        )
    for attr in ("EndfFile", "_endf_roundoff"):
        if not hasattr(upstream, attr):
            raise RuntimeError(f"NCrystal internal ENDF API lacks {attr}")
    return NC, upstream, EndfMetaData, actual


def _find_element_atomdata(
    NC: Any,
    ncmat_path: Path,
    element: str,
    temperature_K: float,
) -> Tuple[Any, float]:
    """Locate the unique ``DynInfo`` atom data and temperature for ``element``.

    Parameters
    ----------
    NC : module
        Imported ``NCrystal`` module.
    ncmat_path : pathlib.Path
        Path to the source NCMAT file.
    element : str
        Element symbol to select.
    temperature_K : float
        Material temperature in kelvin.

    Returns
    -------
    atom_data : Any
        NCrystal ``AtomData`` instance for the selected element.
    temperature : float
        Temperature recorded in the ``DynInfo`` block in kelvin.

    Raises
    ------
    ValueError
        If ``element`` does not match Phased ``dyninfos`` uniquely.
    """
    cfg = f"{ncmat_path.resolve()};temp={temperature_K:.12g}K"
    info = NC.createInfo(cfg)
    matches = [
        di for di in info.dyninfos if di.atomData.elementName() == element
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one dynamic-info component for "
            f"{element}, found {len(matches)}"
        )
    di = matches[0]
    return di.atomData, float(di.temperature)


def build_upstream_data(
    k: LiquidKernel,
    *,
    element: str,
    atomdata: Any,
    effective_temperature_K: float,
    ncmat_path: Path,
    source_sha256: str,
    upstream: Any,
) -> SimpleNamespace:
    """Construct the ``NuclearData`` stand-in expected by ``upstream.EndfFile``.

    NCrystal's ``EndfFile`` expects ``alpha`` and ``beta`` scaled by ``T / 293.6``
    and ``sab_total`` shaped as ``(N_beta, N_alpha)``. ``EndfFile`` then applies
    the ``1 / awr`` mass scaling and ``LAT=1`` conventions during serialization.

    Parameters
    ----------
    k : LiquidKernel
        Validated liquid scattering kernel.
    element : str
        Target element symbol.
    atomdata : Any
        NCrystal ``AtomData`` object for ``element``.
    effective_temperature_K : float
        Effective temperature in kelvin for the SCT limit.
    ncmat_path : pathlib.Path
        Source NCMAT file path.
    source_sha256 : str
        SHA-256 hex digest of the source NCMAT file.
    upstream : module
        Imported ``NCrystal._ncmat2endf_impl`` module.

    Returns
    -------
    types.SimpleNamespace
        Data container compatible with ``upstream.EndfFile``.

    Raises
    ------
    ValueError
        If atomic weight ratio or rounded grids are invalid.
    """
    T = k.temperature_K
    awr = k.mass_amu / float(upstream.nc_constants.const_neutron_mass_amu)
    if not (awr > 0):
        raise ValueError("Invalid atomic weight ratio")

    scaled_alpha = np.asarray(
        upstream._endf_roundoff(k.alpha * (T / upstream.T0)), dtype=np.float64
    )
    scaled_beta = np.asarray(
        upstream._endf_roundoff(k.beta * (T / upstream.T0)), dtype=np.float64
    )
    if np.any(np.diff(scaled_alpha) <= 0) or np.any(np.diff(scaled_beta) <= 0):
        raise ValueError(
            "ENDF numeric rounding collapsed alpha/beta grid points; "
            "explicit resampling is required, refusing silent change"
        )
    if not np.any(scaled_beta == 0.0):
        raise ValueError("Rounded beta grid lost beta=0")

    negative_part = scaled_beta[scaled_beta <= 0]
    if len(negative_part) < 2:
        raise ValueError("Symmetric export would have too few beta points")
    positive_beta = -negative_part[::-1].copy()
    positive_beta[0] = 0.0

    sigma_free = k.bound_xs_barn * (awr / (1.0 + awr)) ** 2
    z = int(atomdata.Z())
    elem_entry = SimpleNamespace(
        alpha=scaled_alpha,
        beta_total=scaled_beta,
        beta=positive_beta,
        sab_total=[k.sab_beta_alpha.copy()],
        teff=[float(effective_temperature_K)],
        elastic=None,  # Only MF7/MT4 is captured; never invent elastic MT2.
        awr=awr,
        za=1000 * z,
        zsymam=f"{z:3d}-" + element.ljust(2) + "    ",
        sigma_free=sigma_free,
    )
    return SimpleNamespace(
        elements={element: elem_entry},
        temperatures=(float(T),),
        ncmat_cfg=str(ncmat_path.resolve()),
        comments=[
            "EXPERIMENTAL: captured final liquid SABData from unmodified plugin",
            f"Source SHA256: {source_sha256}",
            "No elastic MF7/MT2 included; component-only inelastic law",
        ],
        elastic_mode="none",
    )


def _assert_close_scalar(
    label: str,
    actual: float,
    expected: float,
    rtol: float = 2e-4,
) -> None:
    """Assert that two physical scalars agree within relative tolerance.

    Parameters
    ----------
    label : str
        Quantity name for error reporting.
    actual, expected : float
        Values to compare.
    rtol : float, default=2e-4
        Relative tolerance.

    Raises
    ------
    ValueError
        If ``actual`` and ``expected`` disagree beyond ``rtol``.
    """
    if not math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=1e-12):
        raise ValueError(f"{label} mismatch: material {actual}, capture {expected}")


def _verify_s_tables(
    sec: Dict[str, Any],
    ref: Dict[str, Any],
    expected_nbeta: int,
    expected_nalpha: int,
) -> None:
    """Verify every beta record's alpha grid, S values, and interpolation flags.

    Parameters
    ----------
    sec : dict of str to Any
        Parsed MF7/MT4 section from ``endf-parserpy``.
    ref : dict of str to Any
        Reference ``7/4`` dictionary prepared for ``EndfFile``.
    expected_nbeta : int
        Expected number of beta records.
    expected_nalpha : int
        Expected number of alpha points per record.

    Raises
    ------
    RuntimeError
        If any table length, monotonicity, interpolation law, or value fails.
    """
    for j in range(1, expected_nbeta + 1):
        er = ref["S_table"][j]
        got = sec["S_table"][j]
        if (
            len(er["alpha"]) != expected_nalpha
            or len(er["S"]) != expected_nalpha
            or len(got["alpha"]) != expected_nalpha
            or len(got["S"]) != expected_nalpha
        ):
            raise RuntimeError(
                f"S table {j}: expected {expected_nalpha} alpha/S points"
            )
        for name in ("alpha", "S"):
            av = np.asarray(got[name], dtype=np.float64)
            ev = np.asarray(er[name], dtype=np.float64)
            if av.shape != ev.shape:
                raise RuntimeError(f"S table {j} {name}: wrong shape after readback")
            if not np.all(np.isfinite(av)):
                raise RuntimeError(f"S table {j} {name}: nonfinite ENDF values")
            tolerance = np.maximum(
                1e-13 * max(1.0, float(np.max(np.abs(ev)))),
                2e-4 * np.abs(ev),
            )
            if np.any(np.abs(av - ev) > tolerance):
                raise RuntimeError(f"S table {j} {name}: altered on readback")
        if np.any(np.diff(got["alpha"]) <= 0):
            raise RuntimeError("Serialized alpha grid is not increasing")
        for key in ("NBT", "INT"):
            if got[key] != er[key]:
                raise RuntimeError("Alpha interpolation changed")


def _verify_readback_grids(sec, ref, lasym):
    """Verify the MF7 table dimensions, beta coordinates, and symmetry.

    Parameters
    ----------
    sec, ref : dict
        Parsed MF7/MT4 section and expected NCrystal source.
    lasym : int
        Expected MF7 symmetry flag.

    Returns
    -------
    tuple
        Parsed beta array and expected counts of beta and alpha points.
    """
    expected_nbeta = int(ref["NB"])
    if len(ref["beta"]) != expected_nbeta or len(ref["S_table"]) != expected_nbeta:
        raise RuntimeError("NCrystal source dictionary has inconsistent beta counts")
    if len(sec["beta"]) != expected_nbeta or len(sec["S_table"]) != expected_nbeta:
        raise RuntimeError("ENDF readback beta or S-table count mismatch")
    if "NB" in sec and int(sec["NB"]) != expected_nbeta:
        raise RuntimeError("ENDF readback NB mismatch")

    expected_nalpha = len(ref["S_table"][1]["alpha"])
    if expected_nalpha < 2:
        raise RuntimeError("NCrystal source has too few alpha points")

    b_expected = np.asarray(
        [ref["beta"][i] for i in range(1, expected_nbeta + 1)], dtype=np.float64
    )
    b_actual = np.asarray(
        [sec["beta"][i] for i in range(1, expected_nbeta + 1)], dtype=np.float64
    )
    if not np.all(np.isfinite(b_actual)) or not np.allclose(
        b_actual, b_expected, rtol=2e-4, atol=1e-10
    ):
        raise RuntimeError("ENDF readback beta grids differ")
    if np.any(np.diff(b_actual) <= 0):
        raise RuntimeError("Serialized beta grid is not increasing")
    if lasym == 0 and b_actual[0] != 0.0:
        raise RuntimeError("Symmetric MF7 beta grid must start at zero")
    if lasym == 1 and not (b_actual.min() < 0 < b_actual.max()):
        raise RuntimeError("Signed-beta branch missing after ENDF serialization")

    return b_actual, expected_nbeta, expected_nalpha


def _verify_readback_metadata(sec, ref, epsilon_e_over_kt, energy_limit):
    """Check the serialized scattering coefficients and material temperature.

    Parameters
    ----------
    sec, ref : dict
        Parsed MF7/MT4 section and expected NCrystal source.
    epsilon_e_over_kt : float
        Approved dimensionless B(2) threshold.
    energy_limit : float
        Approved B(4) upper energy in eV.
    """
    if not math.isclose(float(sec["B"][2]), epsilon_e_over_kt, rel_tol=2e-4):
        raise RuntimeError("ENDF readback B(2) differs from approved threshold")
    if not math.isclose(float(sec["B"][4]), energy_limit, rel_tol=2e-4):
        raise RuntimeError("ENDF readback B(4) differs from energy limit")
    for key in (1, 3, 6):
        if not math.isclose(
            float(sec["B"][key]), float(ref["B"][key]), rel_tol=2e-4, abs_tol=1e-12
        ):
            raise RuntimeError(f"ENDF readback B({key}) differs from NCrystal input")
    if not math.isclose(float(sec["T0"]), float(ref["T0"]), rel_tol=2e-4):
        raise RuntimeError("ENDF readback material temperature differs")


def _verify_written(
    path: Path,
    expected: Dict[str, Any],
    *,
    lat: int,
    lasym: int,
    epsilon_e_over_kt: float,
    energy_limit: float,
    kernel: LiquidKernel,
    smin: float,
    natom: int,
) -> Dict[str, Any]:
    """Read back the written ENDF file with ``endf-parserpy`` and check fidelity.

    Parameters
    ----------
    path : pathlib.Path
        Written ENDF file path.
    expected : dict of str to Any
        Internal dictionary passed to ``EndfFile.write``.
    lat : int
        Expected ``LAT`` temperature-scaling flag.
    lasym : int
        Expected ``LASYM`` symmetry flag.
    epsilon_e_over_kt : float
        Approved dimensionless ``B(2)`` threshold.
    energy_limit : float
        Approved upper energy limit ``B(4)`` in eV.
    kernel : LiquidKernel
        Validated original capture.
    smin : float
        S(alpha, beta) threshold used during export.
    natom : int
        Principal scatterer multiplicity.

    Returns
    -------
    dict of str to Any
        Summary of readback beta bounds and grid point counts.

    Raises
    ------
    RuntimeError
        If any parsed MF7/MT4 flag, header scalar, or table differs.
    """
    import endf_parserpy

    Parser = getattr(endf_parserpy, "EndfParserPy", None) or endf_parserpy.EndfParser
    parsed = Parser(explain_missing_variable=True, cache_dir=False).parsefile(str(path))
    sec = parsed[7][4]
    ref = expected["7/4"]

    for field, want in (("LAT", lat), ("LASYM", lasym), ("LLN", 0), ("NS", 0)):
        if int(sec[field]) != int(want):
            raise RuntimeError(f"ENDF readback {field} mismatch")

    b_actual, expected_nbeta, expected_nalpha = _verify_readback_grids(sec, ref, lasym)
    _verify_readback_metadata(sec, ref, epsilon_e_over_kt, energy_limit)
    for key in ('NBT', 'INT'):
        if sec['beta_interp'][key] != ref['beta_interp/' + key]:
            raise RuntimeError('Beta interpolation changed on readback')
    _verify_s_tables(sec, ref, expected_nbeta, expected_nalpha)
    teff = sec["teff0_table"]
    for key in ("Teff0", "Tint", "NBT", "INT"):
        if not np.allclose(teff[key], ref["teff0_table/" + key], rtol=2e-6):
            raise RuntimeError("Effective-temperature table changed on readback")

    _verify_capture_values(sec, kernel, smin, natom, lasym=lasym)

    return {
        "readback_beta_min": float(b_actual.min()),
        "readback_beta_max": float(b_actual.max()),
        "readback_nbeta": int(len(b_actual)),
        "readback_nalpha": expected_nalpha,
    }


def _validate_export_arguments(
    k: LiquidKernel,
    *,
    capture_index: int,
    expected_capture_count: int,
    element: str,
    lasym: int,
    mat: int,
    teff_evidence: str,
    effective_temperature_K: float,
    epsilon_e_over_kt: float,
    emax_eV: float,
    downstream_emax_eV: float,
    smin: float,
    natom: int,
) -> Tuple[float, float, float]:
    """Validate all physical and administrative parameters before ENDF export.

    Parameters
    ----------
    k : LiquidKernel
        Validated kernel and its maximum supported incident energy.
    capture_index, expected_capture_count : int
        Capture index and required number of identified captures.
    element : str
        Selected element label.
    lasym, mat, natom : int
        MF7 symmetry flag, ENDF material identifier, and principal atom count.
    teff_evidence : str
        Effective-temperature provenance.
    effective_temperature_K : float
        Effective temperature in kelvin.
    epsilon_e_over_kt : float
        Dimensionless B(2) threshold.
    emax_eV, downstream_emax_eV : float
        Export and downstream processing energy limits in eV.
    smin : float
        Minimum retained scattering-law value.

    Returns
    -------
    teff, eps, emax : tuple of float
        Validated effective temperature (K), dimensionless B(2), and Emax (eV).

    Raises
    ------
    ValueError
        If an argument violates the physical or ENDF export constraints.
    """
    if type(capture_index) is not int or capture_index < 0:
        raise ValueError("capture_index must be a nonnegative integer")
    if expected_capture_count != 1:
        raise ValueError(
            "Provide exactly one identified capture per component; "
            "select and validate components before export"
        )
    if element not in ("H", "D", "O"):
        raise ValueError("Only explicitly identified H/D/O components accepted")
    if lasym not in (0, 1):
        raise ValueError("Supported upstream LASYM modes: 0 or 1")
    if not isinstance(mat, int) or not (1 <= mat <= 9999):
        raise ValueError("MAT must be an integer within 1..9999")
    if type(natom) is not int or natom < 1:
        raise ValueError("natom must be a positive integer")
    if not isinstance(teff_evidence, str) or not teff_evidence.strip():
        raise ValueError("Document provenance for the independently reviewed Teff")

    teff = _positive_float("effective_temperature_K", effective_temperature_K)
    eps = _positive_float("epsilon_e_over_kt (dimensionless B2)", epsilon_e_over_kt)
    emax = _positive_float("emax_eV", emax_eV)
    downstream = _positive_float("downstream_emax_eV", downstream_emax_eV)
    if not math.isclose(emax, downstream, rel_tol=1e-12):
        raise ValueError("MF7 and downstream NJOY Emax must match")
    if emax > k.suggested_emax_eV * (1 + 1e-8):
        raise ValueError("Requested Emax exceeds captured kernel suggested Emax")
    require_finite_nonnegative(smin, message="smin must be nonnegative and finite")
    return teff, eps, emax


def _prepare_export_paths(source_ncmat_path, output_path, force):
    """Validate the NCMAT source and protect the requested destination.

    Parameters
    ----------
    source_ncmat_path, output_path : str or pathlib.Path
        Source Liquids NCMAT and target ENDF file.
    force : bool
        Permit replacing an existing destination.

    Returns
    -------
    tuple
        Resolved source, its SHA-256 hash, and the resolved destination.
    """
    source = Path(source_ncmat_path).expanduser().resolve()
    if not source.is_file() or "@CUSTOM_LIQUIDS" not in source.read_text(encoding="utf-8"):
        raise ValueError("Source must be an existing Liquids-plugin NCMAT")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    target = Path(output_path).expanduser().resolve()
    if target.exists() and not force:
        raise FileExistsError(
            f"Output exists; pass force=True only if intended: {target}"
        )

    return source, source_sha256, target


def _verified_source_atom(ncrystal, source, element, kernel):
    """Check that source NCMAT species data agree with the capture.

    Parameters
    ----------
    ncrystal : module
        Imported NCrystal module.
    source : pathlib.Path
        Source NCMAT path.
    element : str
        Captured species label.
    kernel : LiquidKernel
        Normalized kernel including the temperature and atom data.

    Returns
    -------
    object
        Verified NCrystal atom data.
    """
    atom, matT = _find_element_atomdata(ncrystal, source, element, kernel.temperature_K)
    _assert_close_scalar("Temperature", matT, kernel.temperature_K, rtol=1e-8)
    _assert_close_scalar("Mass [amu]", atom.averageMassAMU(), kernel.mass_amu, rtol=3e-4)
    _assert_close_scalar(
        "Bound scattering XS [barn]", atom.scatteringXS(), kernel.bound_xs_barn, rtol=3e-4
    )

    return atom


def _build_liquid_endf(
    upstream, metadata_type, upstream_data, report, *, element, mat, lasym, smin, emax, eps, natom,
):
    """Construct the upstream writer and apply liquid MF7 metadata.

    Parameters
    ----------
    upstream : module
        NCrystal ENDF implementation.
    metadata_type : type
        NCrystal ENDF metadata class.
    upstream_data : object
        Kernel converted to the upstream exporter data structure.
    report : dict
        Signed-beta detailed-balance diagnostics.
    element : str
        Target species label.
    mat, lasym, natom : int
        ENDF material, symmetry flag, and principal scatterer count.
    smin : float
        Minimum S(alpha, beta) threshold.
    emax, eps : float
        Upper energy in eV and dimensionless B(2) threshold.

    Returns
    -------
    object
        Configured upstream EndfFile.
    """
    if lasym == 0:
        upstream_data.comments.append("Symmetric MF7 derived from captured beta <= 0 branch.")
        mismatch = report["p99_abs_log"]
        if mismatch is not None:
            upstream_data.comments.append(f"Signed capture balance log residual p99: {mismatch:.6g}")
    metadata = metadata_type()
    endf_obj = upstream.EndfFile(
        element,
        upstream_data,
        mat,
        metadata,
        smin=float(smin),
        emax=emax,
        lasym=lasym,
        include_gif=False,
        isotopic_expansion=False,
        verbosity=0,
    )

    # Patch B(2) explicitly (NCrystal 4.4.6 sets B(2)=B(4)=Emax) and B(6) for NATOM.
    d = endf_obj._endf_dict
    if int(d["7/4"]["LASYM"]) != lasym or not math.isclose(
        float(d["7/4"]["B"][4]), emax, rel_tol=1e-12
    ):
        raise RuntimeError("Unexpected NCrystal MF7 internal dictionary contract")
    d["7/4"]["B"][2] = eps
    d["7/4"]["B"][6] = float(natom)
    d["7/4"]["B"][1] *= natom
    d["0/0"]["TAPEDESCR"] = "Liquid SABData via NCrystal 4.4.6 EndfFile"
    d["1/451"]["DESCRIPTION"][1] = (
        "EXPERIMENTAL NCrystal Liquids SABData adapter; NOT ACE VALIDATED".ljust(66)
    )

    return endf_obj


def _publish_liquid_endf(
    endf_obj, target, source, source_sha256, *, kernel, eps, emax, lasym, smin, natom, readback, force,
):
    """Stage, verify, and publish one ENDF file.

    Parameters
    ----------
    endf_obj : object
        NCrystal ENDF writer ready to serialize.
    target, source : pathlib.Path
        Resolved ENDF destination and NCMAT source.
    source_sha256 : str
        Source identity expected before publication.
    kernel : LiquidKernel
        Captured kernel used in readback.
    eps, emax : float
        Dimensionless B(2) and upper energy in eV.
    lasym, natom : int
        Symmetry flag and principal scatterer count.
    smin : float
        S(alpha, beta) threshold used by the writer.
    readback, force : bool
        Verify the staged file and allow replacing the destination, respectively.

    Returns
    -------
    tuple
        Output size in bytes, its SHA-256 hash, and readback metrics.
    """
    expected = endf_obj._endf_dict
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".nc_endf_stage_", dir=target.parent
    ) as td:
        stage = Path(td) / target.name
        endf_obj.write(stage.name, force=False, outdir=td)
        if not stage.is_file() or stage.stat().st_size == 0:
            raise RuntimeError("NCrystal writer did not produce an ENDF file")
        verified = (
            _verify_written(
                stage,
                expected,
                lat=1,
                lasym=lasym,
                epsilon_e_over_kt=eps,
                energy_limit=emax,
                kernel=kernel, smin=smin, natom=natom,
            )
            if readback
            else {"readback_skipped": True}
        )
        if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha256:
            raise RuntimeError("Source NCMAT changed during export")
        output_bytes = stage.stat().st_size
        output_sha256 = hashlib.sha256(stage.read_bytes()).hexdigest()
        # Hard-link publication preserves the no-overwrite contract if the
        # destination appears while the staged file is being verified.
        if force:
            os.replace(stage, target)
        else:
            os.link(stage, target)

    return output_bytes, output_sha256, verified


def write_liquid_with_ncrystal(
    data: Dict[str, Any],
    output_path: str | Path,
    *,
    source_ncmat_path: str | Path,
    element: str = "H",
    mat: int = 9901,
    effective_temperature_K: float,
    teff_evidence: str,
    epsilon_e_over_kt: float,
    emax_eV: float,
    downstream_emax_eV: float,
    lasym: int = 0,
    natom: int = 1,
    capture_index: int = 0,
    expected_capture_count: int = 1,
    force: bool = False,
    smin: float = 1e-100,
    version_prefix: str = SUPPORTED_NCRYSTAL_PREFIX,
    readback: bool = True,
) -> Dict[str, Any]:
    """Write a captured liquid ``SABData`` dictionary to ENDF-6 via ``EndfFile``.

    Parameters
    ----------
    data : dict of str to Any
        Captured ``extractKnl``-style kernel dictionary.
    output_path : str or pathlib.Path
        Destination ENDF-6 file path.
    source_ncmat_path : str or pathlib.Path
        Source liquid NCMAT file path containing ``@CUSTOM_LIQUIDS``.
    element : {"H", "D", "O"}, default="H"
        Explicitly identified scattering species.
    mat : int, default=9901
        ENDF MAT identifier in ``[1, 9999]``.
    effective_temperature_K : float
        Effective temperature in kelvin for the SCT approximation.
    teff_evidence : str
        Provenance description for ``effective_temperature_K``.
    epsilon_e_over_kt : float
        Dimensionless ``B(2)`` threshold parameter.
    emax_eV : float
        Upper energy limit ``B(4)`` in eV.
    downstream_emax_eV : float
        Downstream NJOY upper energy limit in eV (must match ``emax_eV``).
    lasym : {0, 1}, default=0
        Use the solid exporter’s one-sided, symmetric representation with
        zero. Set one to retain both captured beta branches for comparison.
    natom : int, default=1
        Principal scatterer atom count. B(1) is scaled by this count
        together with B(6). The default is a per-atom evaluation.
    capture_index : int, default=0
        Index of the captured kernel for provenance reporting.
    expected_capture_count : int, default=1
        Number of captures expected for this component.
    force : bool, default=False
        Whether to overwrite an existing ``output_path``.
    smin : float, default=1e-100
        Minimum S value threshold passed to ``EndfFile``.
    version_prefix : str, default="4.4.6"
        Required NCrystal version string.
    readback : bool, default=True
        Whether to parse and verify the written file with ``endf-parserpy``.

    Returns
    -------
    dict of str to Any
        Export manifest including checksums, parameters, and readback metrics.
    """
    k = normalise_capture(data)
    teff, eps, emax = _validate_export_arguments(
        k,
        capture_index=capture_index,
        expected_capture_count=expected_capture_count,
        element=element,
        lasym=lasym,
        mat=mat,
        teff_evidence=teff_evidence,
        effective_temperature_K=effective_temperature_K,
        epsilon_e_over_kt=epsilon_e_over_kt,
        emax_eV=emax_eV,
        downstream_emax_eV=downstream_emax_eV,
        smin=smin,
        natom=natom,
    )

    source, source_sha256, target = _prepare_export_paths(
        source_ncmat_path, output_path, force,
    )
    NC, upstream, EndfMetaData, ncrystal_version = _import_upstream(version_prefix)
    atom = _verified_source_atom(NC, source, element, k)

    upstream_data = build_upstream_data(
        k, element=element, atomdata=atom, effective_temperature_K=teff,
        ncmat_path=source, source_sha256=source_sha256, upstream=upstream,
    )
    report = detailed_balance_report(k)
    endf_obj = _build_liquid_endf(
        upstream, EndfMetaData, upstream_data, report, element=element,
        mat=mat, lasym=lasym, smin=smin, emax=emax, eps=eps, natom=natom,
    )
    output_bytes, output_sha256, verified = _publish_liquid_endf(
        endf_obj, target, source, source_sha256, kernel=k, eps=eps,
        emax=emax, lasym=lasym, smin=smin, natom=natom, readback=readback, force=force,
    )

    return dict(
        output_file=str(target),
        output_bytes=output_bytes,
        output_sha256=output_sha256,
        source_ncmat_sha256=source_sha256,
        ncrystal_version=ncrystal_version,
        interface="NCrystal._ncmat2endf_impl.EndfFile",
        component=element,
        capture_index=capture_index,
        temperature_K=k.temperature_K,
        effective_temperature_K=teff,
        teff_evidence=teff_evidence,
        b2_dimensionless=eps,
        b4_emax_eV=emax,
        natom=natom,
        lasym=lasym,
        mat=mat,
        alpha_points=len(k.alpha),
        beta_points=len(k.beta),
        smin=smin,
        db_diagnostic=report,
        **verified,
    )


def load_extractknl_npz(path: str | Path) -> Dict[str, Any]:
    """Read an ``extractKnl``-compatible ``.npz`` archive.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the ``.npz`` file.

    Returns
    -------
    dict of str to Any
        Dictionary containing the 7 required ``extractKnl`` keys.

    Raises
    ------
    ValueError
        If any required key is missing from the archive.
    """
    with np.load(path, allow_pickle=False) as z:
        keys = (
            "alpha",
            "beta",
            "sab",
            "mass_amu",
            "scatxs",
            "temperature",
            "suggested_emax",
        )
        if any(k not in z for k in keys):
            raise ValueError(f"Not an extractKnl-compatible capture: {path}")
        return {
            k: z[k].item() if z[k].ndim == 0 else z[k].copy() for k in keys
        }


def load_sabtap_json(path: str | Path) -> Dict[str, Any]:
    """Convert a raw ``sab_XXXX.json`` dump from the C++ hook to ``extractKnl`` format.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to ``sab_XXXX.json`` (supports both inline JSON arrays and adjacent
        ``.bin`` payloads).

    Returns
    -------
    dict of str to Any
        Validated ``extractKnl``-compatible dictionary with 1-D C-order ``sab``.

    Raises
    ------
    ValueError
        If the JSON layout or array dimensions are invalid.
    """
    json_path = Path(path)
    record = json.loads(json_path.read_text(encoding="utf-8"))
    if record.get("layout") != "sab[beta_index][alpha_index]":
        raise ValueError("Unexpected SABData JSON layout")

    alpha, beta, sab = read_sabtap_arrays(json_path, record)

    out = dict(
        alpha=alpha,
        beta=beta,
        sab=sab.ravel(order="C"),
        mass_amu=float(record["element_mass_amu"]),
        scatxs=float(record["bound_xs_barn"]),
        temperature=float(record["temperature_K"]),
        suggested_emax=float(record["suggested_emax_eV"]),
    )
    normalise_capture(out)
    return out


def _prepare_sample_export(state, atom_label, temperature, effective_temperature_K):
    """Verify the capture's source identity and prepare its export values.

    Parameters
    ----------
    state : LiquidAnalysisState
        Captured sample with kernel, backend thermodynamics, and provenance.
    atom_label : str
        Target species label.
    temperature : float
        Physical material temperature in kelvin.
    effective_temperature_K : float or None
        User-supplied effective temperature, if provided.

    Returns
    -------
    tuple
        Generated NCMAT content, extractKnl-style dictionary, and effective temperature in kelvin.
    """
    from input_output.ncrystal.liquids import parse_custom_liquids

    kernel = state.kernels[atom_label]
    material = state.provenance["material_ncmat"]
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    if digest != state.provenance["material_sha256"]:
        raise RuntimeError("Captured material identity does not match export")

    settings = next(
        x for x in parse_custom_liquids(material) if x.atom == atom_label
    )
    teff = effective_temperature_K
    if teff is None:
        teff = (
            settings.ws * state.vdos_analysis["teff"]
            + settings.wt * temperature
        ) / (settings.ws + settings.wt)

    data = dict(
        alpha=kernel.alpha,
        beta=kernel.beta,
        sab=kernel.sab,
        mass_amu=kernel.mass_amu,
        scatxs=kernel.bound_xs,
        temperature=kernel.temperature,
        suggested_emax=kernel.suggested_emax,
    )
    return material, data, teff


def generate_liquid_tape(
    *,
    sample_id: str,
    sample_index: int,
    spectrum: Any,
    output_path: str | Path,
    base_ncmat_file: str | Path,
    temperature: float,
    atom_label: str,
    capture_library: str | Path,
    epsilon_e_over_kt: float = DEFAULT_LIQUID_EPSILON,
    emax_eV: float = DEFAULT_LIQUID_EMAX_EV,
    effective_temperature_K: Optional[float] = None,
    natom: int = 1,
    vdoslux: int = DEFAULT_LIQUID_VDOSLUX,
    timeout: float = 600,
    force: bool = False,
) -> Tuple[str, Optional[str], Optional[str]]:
    """Capture one sampled liquid and write its component MF7 ENDF tape.

    Parameters
    ----------
    sample_id : str
        Stable sample identifier (e.g., ``"sample_000"``).
    sample_index : int
        Reference (``0``) or stochastic sample index.
    spectrum : PhononSpectra
        Sample-specific solid-like VDOS.
    output_path, base_ncmat_file, capture_library : str or pathlib.Path
        Destination ENDF path, source NCMAT template, and compiled C++ adapter.
    temperature : float
        Material temperature in kelvin.
    atom_label : str
        Target liquid species label (e.g., ``"H"``).
    epsilon_e_over_kt, emax_eV : float, optional
        Research defaults for MF7 B(2) and B(4) are 20 and 0.1 eV.
        The upper energy limit must be supported by the captured kernel.
    effective_temperature_K : float, optional
        Explicit effective temperature in kelvin. When omitted, uses the
        weighted solid-like VDOS + translational prescription for this sample.
    natom : int, default=1
        Principal scatterer multiplicity ``B(6)`` (e.g., ``2`` for H in H2O).
    vdoslux : int, default=0
        Native liquid kernel resolution setting.
    timeout : float, default=600
        Subprocess capture timeout in seconds.
    force : bool, default=False
        Whether to overwrite an existing ``output_path``.

    Returns
    -------
    tuple of (str, str or None, str or None)
        ``(sample_id, output_path, None)`` on success, or
        ``(sample_id, None, error_message)`` on failure.
    """
    from processing.backends.ncrystal_liquid_analysis import (
        NCrystalLiquidAnalysisProcessor,
    )

    try:
        processor = NCrystalLiquidAnalysisProcessor(
            base_ncmat_file,
            temperature,
            capture_library=capture_library,
            vdoslux=vdoslux,
            timeout=timeout,
        )
        state = processor.create_analysis_state(
            sample_index, spectrum, atom_label, energy_grid=[0.0253]
        )
        material, data, teff = _prepare_sample_export(
            state, atom_label, temperature, effective_temperature_K,
        )
        with tempfile.TemporaryDirectory(prefix="liquid_mf7_") as temporary:
            source = Path(temporary) / "sample.ncmat"
            source.write_text(material, encoding="utf-8")
            write_liquid_with_ncrystal(
                data,
                output_path,
                source_ncmat_path=source,
                element=atom_label,
                effective_temperature_K=teff,
                teff_evidence=(
                    "Sample-specific liquid VDOS/translation prescription"
                    if effective_temperature_K is None
                    else "Caller supplied"
                ),
                epsilon_e_over_kt=epsilon_e_over_kt,
                emax_eV=emax_eV,
                downstream_emax_eV=emax_eV,
                natom=natom,
                force=force,
            )
        return sample_id, str(output_path), None
    except Exception as exc:
        return sample_id, None, f"{type(exc).__name__}: {exc}"


def _verify_capture_values(section, kernel, smin, natom, *, lasym):
    """Compare parsed MF7 coordinates and values directly with the capture.

    Parameters
    ----------
    section : dict
        Parsed MF7/MT4 section.
    kernel : LiquidKernel
        Original unsymmetrized capture.
    smin : float
        Applied threshold in the serialized representation.
    natom : int
        Principal-atom multiplicity.
    lasym : int
        Zero checks the reversed nonpositive capture branch, and one checks
        the full signed capture.

    Raises
    ------
    RuntimeError
        If physical coordinates, normalization, or table values changed.

    Notes
    -----
    The exponential uses the ENDF-rounded beta coordinate, matching the
    pinned writer. The original capture remains the source of S values.
    """
    from NCrystal.constants import const_neutron_mass_amu
    ratio = kernel.temperature_K / T0_K
    awr = kernel.mass_amu / const_neutron_mass_amu
    beta = np.asarray([section['beta'][j] for j in sorted(section['beta'])])
    # NCrystal computes the ENDF factor on the original signed coordinates.
    # A symmetric table stores the nonpositive rows on a nonnegative grid.
    if lasym == 0:
        source_rows = np.flatnonzero(kernel.beta <= 0)[::-1]
        expected_beta = -kernel.beta[source_rows] * ratio
        signed_beta = -beta
    else:
        source_rows = np.arange(kernel.beta.size)
        expected_beta = kernel.beta * ratio
        signed_beta = beta
    if beta.shape != expected_beta.shape or not np.allclose(
        beta, expected_beta, rtol=2e-6, atol=1e-12
    ):
        raise RuntimeError('MF7 beta differs from captured physical coordinates')
    sigma_free = kernel.bound_xs_barn * (awr / (1 + awr)) ** 2
    if not math.isclose(section['B'][1] / section['B'][6], sigma_free, rel_tol=2e-6):
        raise RuntimeError('MF7 per-atom cross-section normalization changed')
    if section['B'][6] != natom:
        raise RuntimeError('MF7 principal atom count changed')
    for index, key in enumerate(sorted(section['S_table'])):
        row = section['S_table'][key]
        if not np.allclose(row['alpha'], kernel.alpha * ratio / awr,
                           rtol=3e-6, atol=1e-12):
            raise RuntimeError('MF7 alpha differs from captured physical coordinates')
        expected = (
            kernel.sab_beta_alpha[source_rows[index]]
            * np.exp(signed_beta[index] / ratio / 2)
        )
        expected = np.where(expected < smin, 0.0, expected)
        if not np.allclose(row['S'], expected, rtol=2e-4, atol=max(smin * 2, 1e-300)):
            raise RuntimeError('MF7 S values differ from captured kernel')
