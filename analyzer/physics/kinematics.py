"""Shared thermal-scattering kinematics.

Analyzer uses one canonical scattering-law representation internally:

    rows    -> alpha
    columns -> signed beta
    values  -> full S(alpha, beta)

Backend-native representations are normalized by :func:`canonicalize_sab`.
A compact table with non-negative beta coordinates is interpreted as
S(alpha, -|beta|) and expanded with detailed balance. A table containing both
negative and positive beta coordinates is treated as an already full signed
S(alpha, beta) table.

`Dsf` owns the fixed physical state and every calculation. `DsfResult` is only
an incident-energy data container. The `Dsf` instance caches one result per
incident energy so fixed state is stored once and per-Ein work is not repeated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.constants import physical_constants
from scipy.interpolate import RegularGridInterpolator

from common.validation import require_finite_array, require_strictly_increasing


_KB_EV = physical_constants["Boltzmann constant in eV/K"][0]
_NEUTRON_MASS_AMU = physical_constants["neutron mass in u"][0]


def canonicalize_sab(sab: pd.DataFrame) -> pd.DataFrame:
    """Return the canonical full signed-beta S(alpha, beta) table.

    The backend coordinate grids are never rounded, snapped, merged, sorted,
    averaged, or otherwise modified. Canonicalization changes the matrix only
    when the input uses the compact NJOY convention.

    The accepted input conventions are listed below.

    full signed form
        Beta contains both negative and positive values. The table is already
        S(alpha, beta) and is returned unchanged (as a copy).
    compact form
        Beta is non-negative and contains at least one positive value. The
        values are interpreted as S(alpha, -|beta|). The negative branch uses
        the compact values directly and the positive branch is reconstructed
        by detailed balance:

        S(alpha, +beta) = exp(-beta) S(alpha, -beta).

    Parameters
    ----------
    sab : pandas.DataFrame
        Scattering-law table with alpha on the row index and beta on the
        columns.

    Returns
    -------
    pandas.DataFrame
        Full signed-beta S(alpha, beta). Existing signed beta coordinates are
        preserved exactly. Compact non-negative beta coordinates are mirrored
        exactly to construct the missing branch.

    Raises
    ------
    ValueError
        If coordinates or values are non-finite, alpha/beta coordinates are
        not strictly increasing, or the beta domain is negative-only / zero-
        only and therefore cannot be identified as an accepted convention.
    """
    table = pd.DataFrame(sab, copy=True)

    alpha = np.asarray(table.index, dtype=float)
    beta = np.asarray(table.columns, dtype=float)
    values = table.to_numpy(dtype=float)

    _validate_sab_table(alpha, beta, values)

    has_negative = bool(np.any(beta < 0.0))
    has_positive = bool(np.any(beta > 0.0))

    # Already full signed S(alpha,beta): preserve the backend grid and values
    # exactly. No detailed-balance operation is performed here.
    if has_negative and has_positive:
        return table

    if has_negative and not has_positive:
        raise ValueError(
            "sab beta coordinates are negative-only. Provide either a full "
            "signed beta table or compact non-negative |beta| values for "
            "S(alpha, -|beta|)."
        )

    if not has_positive:
        raise ValueError("sab beta coordinates must span a non-zero beta range.")

    return _expand_compact_sab(table, beta, values)


def _validate_sab_table(alpha: np.ndarray, beta: np.ndarray, values: np.ndarray) -> None:
    """Validate native backend grids without modifying coordinates or values.

    Parameters
    ----------
    alpha, beta : numpy.ndarray
        Native dimensionless kinematic coordinates, in backend order.
    values : numpy.ndarray
        Scattering-law matrix with shape ``(len(alpha), len(beta))``.

    Raises
    ------
    ValueError
        If values are nonfinite or coordinates are not strictly increasing.
    """
    if alpha.size < 2 or beta.size < 2:
        raise ValueError(
            "sab must contain at least two alpha points and two beta points."
        )
    require_finite_array(alpha, message="sab coordinates must be finite.")
    require_finite_array(beta, message="sab coordinates must be finite.")
    require_finite_array(values, message="sab values must be finite.")

    # Do not alter backend grids. RegularGridInterpolator requires strictly
    # increasing coordinates, so validate the native coordinates as received.
    require_strictly_increasing(alpha, message="sab alpha coordinates must be strictly increasing.")
    require_strictly_increasing(beta, message="sab beta coordinates must be strictly increasing.")




def _expand_compact_sab(table: pd.DataFrame, beta: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    """Mirror compact NJOY beta values and apply detailed balance once.

    Parameters
    ----------
    table : pandas.DataFrame
        Original compact table, retaining axis names and alpha coordinates.
    beta : numpy.ndarray
        Nonnegative, strictly increasing compact beta coordinates.
    values : numpy.ndarray
        Native compact ``S(alpha, -|beta|)`` matrix.

    Returns
    -------
    pandas.DataFrame
        Full signed beta representation on exactly mirrored coordinates.
    """
    # This compact form stores the physical negative-beta branch.
    positive = beta > 0.0
    beta_abs = beta[positive]
    negative_values = values[:, positive]
    positive_values = negative_values * np.exp(-beta_abs)[None, :]
    full_beta, full_values = _assemble_signed_beta(
        beta_abs, beta[beta == 0.0], negative_values, values[:, beta == 0.0],
        positive_values,
    )
    return pd.DataFrame(
        full_values,
        index=table.index.copy(),
        columns=pd.Index(full_beta, name=table.columns.name),
    )


def _assemble_signed_beta(
    beta_abs: np.ndarray,
    beta_zero: np.ndarray,
    negative_values: np.ndarray,
    zero_values: np.ndarray,
    positive_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble decoded branches in increasing signed-beta order.

    Parameters
    ----------
    beta_abs : numpy.ndarray
        Positive beta coordinates in increasing order.
    beta_zero : numpy.ndarray
        The source zero coordinate, if present.
    negative_values : numpy.ndarray
        Negative-branch S values in increasing absolute-beta order.
    zero_values : numpy.ndarray
        Physical S values for the optional zero coordinate.
    positive_values : numpy.ndarray
        Positive-branch S values in increasing absolute-beta order.

    Returns
    -------
    tuple of numpy.ndarray
        Signed beta coordinates and their alpha-by-beta matrix.
    """
    # Reverse the negative branch so its coordinates increase toward zero.
    beta = np.concatenate((-beta_abs[::-1], beta_zero, beta_abs))
    values = np.concatenate((negative_values[:, ::-1], zero_values, positive_values), axis=1)
    return beta, values


def decode_njoy_mf7_sab(
    sab: pd.DataFrame,
    *,
    lasym: int,
    lln: int,
) -> pd.DataFrame:
    """Decode an NJOY ENDF MF=7/MT=4 table to physical signed-beta S.

    Parameters
    ----------
    sab : pandas.DataFrame
        Raw MF7 alpha-by-beta table, **before** detailed-balance expansion.
        Beta is dimensionless and follows ENDF's native tabulation: for
        ``LASYM=0`` it is nonnegative; for ``LASYM=1`` it is signed.
    lasym : {0, 1}
        ENDF MF7 ``LASYM`` flag: 0 denotes symmetric S and 1 denotes
        an already asymmetric scattering law on a signed-beta grid.
    lln : {0, 1}
        ENDF MF7 ``LLN`` flag: 0 denotes direct values of S and 1 denotes
        **natural logarithms** ln(S). Do not confuse this with a raw
        LEAPR input's ``ilog`` setting.

    Returns
    -------
    pandas.DataFrame
        Physical non-symmetric S(alpha,beta) on a full signed beta grid,
        where beta = (E_out - E_in)/(k_B*T). No interpolation or averaging
        of the source alpha or beta nodes is performed.

    Notes
    -----
    For symmetric MF7 data and b > 0, the two physical branches are

        S(alpha,-b) = exp(+b/2) * S_sym(alpha,b)
        S(alpha,+b) = exp(-b/2) * S_sym(alpha,b).

    An asymmetric MF7 table is already signed and is not reflected or
    modified by detailed balance. The common ``canonicalize_sab`` function
    deliberately retains its existing behavior for non-MF7 callers.

    Raises
    ------
    ValueError
        If flags, grids, stored values, or decoded S values are invalid.
    """
    if type(lasym) is not int or lasym not in (0, 1):
        raise ValueError("MF7 LASYM must be 0 (symmetric) or 1 (asymmetric).")
    if type(lln) is not int or lln not in (0, 1):
        raise ValueError("MF7 LLN must be 0 (S) or 1 (natural log S).")

    table = pd.DataFrame(sab, copy=True)
    alpha = table.index.to_numpy(dtype=float)
    beta = table.columns.to_numpy(dtype=float)
    stored = table.to_numpy(dtype=float)
    _validate_sab_table(alpha, beta, stored)

    if lln == 0 and np.any(stored < 0.0):
        raise ValueError("MF7 directly stored scattering-law values must be nonnegative.")

    if lasym == 1:
        values = _decode_asymmetric_mf7(beta, stored, lln)
        return pd.DataFrame(values, index=table.index.copy(), columns=table.columns.copy())

    full_beta, values = _decode_symmetric_mf7(beta, stored, lln)
    return pd.DataFrame(
        values,
        index=table.index.copy(),
        columns=pd.Index(full_beta, name=table.columns.name),
    )


def _decode_asymmetric_mf7(
    beta: np.ndarray, stored: np.ndarray, lln: int,
) -> np.ndarray:
    """Decode MF7 values that are already tabulated on both beta branches.

    Parameters
    ----------
    beta : numpy.ndarray
        Source beta coordinates.
    stored : numpy.ndarray
        ENDF scattering-law values before converting logarithms.
    lln : {0, 1}
        Storage flag for direct values or natural logarithms.

    Returns
    -------
    numpy.ndarray
        Physical S values on the source coordinates.
    """
    if not (np.any(beta < 0.0) and np.any(beta > 0.0)):
        raise ValueError("MF7 LASYM=1 requires a full signed-beta table.")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        values = np.exp(stored) if lln == 1 else stored.copy()
    if not np.isfinite(values).all():
        raise ValueError("MF7 decoded scattering-law values must be finite.")
    return values


def _decode_symmetric_mf7(
    beta: np.ndarray, stored: np.ndarray, lln: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover both physical branches of a symmetric MF7 table.

    Parameters
    ----------
    beta : numpy.ndarray
        Nonnegative source beta coordinates.
    stored : numpy.ndarray
        Symmetric S values or natural logarithms.
    lln : {0, 1}
        Storage flag for direct values or natural logarithms.

    Returns
    -------
    tuple of numpy.ndarray
        Signed beta coordinates and physical S values.
    """
    positive = beta > 0.0
    if np.any(beta < 0.0) or not np.any(positive):
        raise ValueError("MF7 LASYM=0 requires nonnegative beta with a positive point.")

    zero = beta == 0.0
    beta_abs = beta[positive]
    half_beta = 0.5 * beta_abs[None, :]
    positive_stored = stored[:, positive]
    zero_values = stored[:, zero]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if lln == 1:
            # Combine the exponents before evaluating exp so an intermediate
            # underflow cannot discard a representable physical value.
            negative_values = np.exp(positive_stored + half_beta)
            positive_values = np.exp(positive_stored - half_beta)
            zero_values = np.exp(zero_values)
        else:
            negative_values = positive_stored * np.exp(half_beta)
            positive_values = positive_stored * np.exp(-half_beta)

    full_beta, values = _assemble_signed_beta(
        beta_abs, beta[zero], negative_values, zero_values, positive_values,
    )
    if not np.isfinite(values).all():
        raise ValueError("MF7 symmetric-to-physical conversion overflowed.")
    return full_beta, values


@dataclass
class DsfResult:
    """Data calculated for one incident neutron energy.

    Parameters
    ----------
    Ein : float
        Incident neutron energy in eV.
    data : pd.DataFrame or None, optional
        Tabulated dynamic structure factor. Default is None.
    kernel : pd.Series or None, optional
        Scattering kernel for the selected incident energy. Default is None.
    angular_distribution : pd.Series or None, optional
        Angular distribution for the selected incident energy. Default is
        None.
    energy_distribution : pd.Series or None, optional
        Outgoing energy distribution. Default is None.
    ratios : pd.Series or None, optional
        Integrated scattering ratios. Default is None.
    """

    # Incident energy.
    Ein: float

    # Scattering observables.
    data: Optional[pd.DataFrame] = None
    kernel: Optional[pd.Series] = None
    angular_distribution: Optional[pd.Series] = None
    energy_distribution: Optional[pd.Series] = None
    ratios: Optional[pd.Series] = None


class Dsf:
    """Dynamic-structure calculator for one fixed full S(alpha, beta).

    The scattering law is canonicalized once during initialization. Fixed state
    consists of the signed S(alpha,beta) table, temperature, target mass and
    angular grid. Incident energy is the only variable state.
    """

    def __init__(
        self,
        sab: pd.DataFrame,
        temperature: float,
        mass_amu: float,
        theta: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize a scattering-law calculator and its interpolation cache.

        Parameters
        ----------
        sab : pandas.DataFrame
            Signed alpha-beta scattering-law table.
        temperature : float
            Material temperature in kelvin.
        mass_amu : float
            Scattering target mass in atomic mass units.
        theta : numpy.ndarray or None, optional
            Angular grid in degrees; defaults to 0 through 180 degrees.
        """
        self.temperature = self._validate_positive(temperature, "temperature")
        self.mass_amu = self._validate_positive(mass_amu, "mass_amu")

        self.sab = canonicalize_sab(sab)
        self.alpha = self.sab.index.to_numpy(dtype=float)
        self.beta = self.sab.columns.to_numpy(dtype=float)

        self.kbT = _KB_EV * self.temperature
        self.A = self.mass_amu / _NEUTRON_MASS_AMU
        self.aws = ((self.A + 1.0) / self.A) ** 2
        self.norm_prefactor = self.aws / (2.0 * self.kbT)

        self.theta = self._prepare_theta(theta)
        self.mu = np.sort(np.unique(np.cos(np.deg2rad(self.theta))))
        self.beta_eval = self._prepare_beta_evaluation_grid()

        self.interpolator = RegularGridInterpolator(
            (self.alpha, self.beta),
            self.sab.to_numpy(dtype=float),
            bounds_error=False,
            fill_value=0.0,
        )

        self._results: Dict[float, DsfResult] = {}

    # ------------------------------------------------------------------
    # Fixed-state preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_positive(value: float, name: str) -> float:
        """Convert a physical parameter to a strictly positive number.

        Parameters
        ----------
        value : float
            Parameter to validate.
        name : str
            Parameter name for diagnostic messages.

        Returns
        -------
        float
            Validated positive value.

        Raises
        ------
        ValueError
            If ``value`` is not positive.
        """
        value = float(value)
        if value <= 0.0:
            raise ValueError(f"{name} must be strictly positive.")
        return value

    @staticmethod
    def _prepare_theta(theta: Optional[np.ndarray]) -> np.ndarray:
        """Return the configured angular grid in degrees.

        Parameters
        ----------
        theta : numpy.ndarray or None
            Optional user-supplied angular grid.

        Returns
        -------
        numpy.ndarray
            Angular evaluation grid.
        """
        if theta is None:
            return np.arange(0.0, 181.0, 1.0)
        return np.asarray(theta, dtype=float)

    def _prepare_beta_evaluation_grid(self) -> np.ndarray:
        """Insert one midpoint between each pair of native beta coordinates.

        Returns
        -------
        numpy.ndarray
            Sorted evaluation coordinates containing the native nodes and their
            midpoints.
        """
        left = self.beta[:-1]
        right = self.beta[1:]
        midpoint = 0.5 * (left + right)
        return np.sort(np.unique(np.concatenate((self.beta, midpoint))))

    # ------------------------------------------------------------------
    # Result/cache access
    # ------------------------------------------------------------------

    def result(self, Ein: float) -> DsfResult:
        """Retrieve or initialize the cached result for one incident energy.

        Parameters
        ----------
        Ein : float
            Finite positive incident neutron energy in eV.

        Returns
        -------
        DsfResult
            Result keyed by incident energy. Calculations populate its fields on
            demand.

        Raises
        ------
        ValueError
            If the incident energy is invalid.
        """
        ein = self._validate_positive(Ein, "Ein")
        if ein not in self._results:
            self._results[ein] = DsfResult(Ein=ein)
        return self._results[ein]

    @property
    def results(self) -> Dict[float, DsfResult]:
        """Expose the current per-energy result cache.

        Returns
        -------
        dict of float to DsfResult
            Cached results keyed by incident neutron energy in eV.
        """
        return self._results

    # ------------------------------------------------------------------
    # Core DSF calculation
    # ------------------------------------------------------------------

    def compute_data(self, Ein: float) -> pd.DataFrame:
        """Calculate and cache the physical DSF for one incident energy.

        Parameters
        ----------
        Ein : float
            Incident neutron energy in eV.

        Returns
        -------
        pd.DataFrame
            Cached two-dimensional differential scattering data.
        """
        result = self.result(Ein)
        if result.data is not None:
            return result.data

        eout, beta = self._outgoing_energy_grid(result.Ein)
        alpha = self._alpha_grid(result.Ein, eout)
        values = self._interpolate(alpha, beta)
        values = self._normalize(values, result.Ein, eout)

        result.data = self._to_dataframe(values, eout)
        return result.data

    def _outgoing_energy_grid(
        self,
        Ein: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a physical Eout grid from signed beta with Eout >= 0.

        Signed beta follows beta = (Eout - Ein) / kBT. The source S(alpha,beta)
        grid is only the interpolation source; outgoing energy remains an
        independent physical integration coordinate. Extra points are inserted
        on the downscattering branch so low incident energies are not forced to
        coincide with coarse native beta nodes.

        Parameters
        ----------
        Ein : float
            Incident neutron energy in eV.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Outgoing energy coordinates and signed energy-transfer grid.
        """
        # The lower physical beta bound corresponds to zero outgoing energy.
        beta_min_physical = -Ein / self.kbT

        native_down = self.beta_eval[
            (self.beta_eval < 0.0)
            & (self.beta_eval > beta_min_physical)
        ]
        # Insert downscattering nodes even when no native beta node lies between zero
        # and the kinematic limit.
        beta_down = np.unique(
            np.concatenate(
                (
                    native_down,
                    np.linspace(beta_min_physical, 0.0, 5),
                    np.array([beta_min_physical, 0.0]),
                )
            )
        )

        beta_up = self.beta_eval[self.beta_eval > 0.0]
        beta = np.concatenate((beta_down, beta_up))
        eout = Ein + beta * self.kbT
        eout = np.clip(eout, 0.0, None)

        return eout, beta

    def _alpha_grid(self, Ein: float, eout: np.ndarray) -> np.ndarray:
        """Compute momentum transfers for each angular and outgoing-energy pair.

        Parameters
        ----------
        Ein : float
            Incident neutron energy in eV.
        eout : np.ndarray
            Outgoing neutron energies in eV.

        Returns
        -------
        np.ndarray
            Dimensionless momentum transfers with shape
            ``(n_angles, n_outgoing_energies)``.

        Notes
        -----
        The kinematic relation is
        ``alpha = (Ein + eout - 2 * mu * sqrt(Ein * eout)) / (A * k_B * T)``.
        Here `mu` is the laboratory-frame scattering-angle cosine and `A`
        is the ratio of the target mass to the neutron mass.
        """
        # Broadcast scattering-angle cosines against outgoing energies.
        mu = self.mu[:, np.newaxis]
        return (Ein + eout - 2.0 * mu * np.sqrt(Ein * eout)) / (self.A * self.kbT)

    def _interpolate(
        self,
        alpha: np.ndarray,
        beta: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the cached scattering-law interpolator.

        Parameters
        ----------
        alpha, beta : numpy.ndarray
            Dimensionless momentum- and energy-transfer coordinates.

        Returns
        -------
        numpy.ndarray
            Interpolated scattering-law values.
        """
        beta_grid = np.broadcast_to(beta, alpha.shape)
        points = np.stack((alpha, beta_grid), axis=-1)
        return self.interpolator(points)

    def _normalize(
        self,
        values: np.ndarray,
        Ein: float,
        eout: np.ndarray,
    ) -> np.ndarray:
        """Scale interpolated values by the scattering prefactor.

        Parameters
        ----------
        values : numpy.ndarray
            Unnormalized scattering-law values.
        Ein : float
            Incident neutron energy in eV.
        eout : numpy.ndarray
            Outgoing neutron energies in eV.

        Returns
        -------
        numpy.ndarray
            Scaled scattering-law values.
        """
        factor = self.norm_prefactor * np.sqrt(eout / Ein)
        return values * factor[np.newaxis, :]

    def _to_dataframe(
        self,
        values: np.ndarray,
        eout: np.ndarray,
    ) -> pd.DataFrame:
        """Label the differential scattering matrix with physical coordinates.

        Parameters
        ----------
        values : numpy.ndarray
            Differential values with one row per cosine and one column per
            outgoing energy.
        eout : numpy.ndarray
            Outgoing neutron energies in eV.

        Returns
        -------
        pandas.DataFrame
            Differential values indexed by scattering-angle cosine, with
            outgoing energy along the columns.
        """
        return pd.DataFrame(
            values,
            index=pd.Index(self.mu, name="mu"),
            columns=pd.Index(eout, name="Eout"),
        )

    # ------------------------------------------------------------------
    # Derived projections
    # ------------------------------------------------------------------

    def compute_kernel(self, Ein: float) -> pd.Series:
        """Integrate the double-differential data over scattering-angle cosine.

        Parameters
        ----------
        Ein : float
            Incident neutron energy in eV.

        Returns
        -------
        pandas.Series
            Scattering kernel per unit outgoing energy, indexed by outgoing
            neutron energy in eV.
        """
        result = self.result(Ein)
        if result.kernel is None:
            result.kernel = self._kernel_from_data(
                self.compute_data(result.Ein),
                name=result.Ein,
            )
        return result.kernel

    @staticmethod
    def _kernel_from_data(
        data: pd.DataFrame,
        name: Optional[float] = None,
    ) -> pd.Series:
        """Integrate an angular scattering distribution over cosine.

        Parameters
        ----------
        data : pandas.DataFrame
            Differential scattering data indexed by cosine and outgoing energy.
        name : float or None, optional
            Name for the resulting kernel.

        Returns
        -------
        pandas.Series
            Integrated scattering kernel versus outgoing energy.
        """
        return pd.Series(
            np.trapezoid(
                data.to_numpy(dtype=float),
                x=data.index.to_numpy(dtype=float),
                axis=0,
            ),
            index=data.columns,
            name=name,
        )

    def compute_angular_distribution(
        self,
        Ein: float,
        angle_axis: str = "mu",
    ) -> pd.Series:
        """Integrate the double-differential cross section over outgoing energy.

        Parameters
        ----------
        Ein : float
            Incident neutron energy in eV.
        angle_axis : str, optional
            Label the distribution by scattering-angle cosine ("mu", the
            default). Other values select the scattering angle in degrees. The values
            remain differential in cosine in both cases.

        Returns
        -------
        pandas.Series
            Distribution per unit cosine, indexed by the selected
            angular coordinate and named by the incident energy.
        """
        result = self.result(Ein)
        result.angular_distribution = self._angular_from_data(
            self.compute_data(result.Ein),
            angle_axis=angle_axis,
            name=result.Ein,
        )
        return result.angular_distribution

    @staticmethod
    def _angular_from_data(
        data: pd.DataFrame,
        angle_axis: str = "mu",
        name: Optional[float] = None,
    ) -> pd.Series:
        """Integrate differential scattering data over outgoing energy.

        Parameters
        ----------
        data : pandas.DataFrame
            Differential scattering data indexed by cosine and outgoing energy.
        angle_axis : str, default='mu'
            Whether to index the result by cosine or scattering angle.
        name : float or None, optional
            Name for the resulting distribution.

        Returns
        -------
        pandas.Series
            Angular distribution in the selected coordinate.
        """
        mu = data.index.to_numpy(dtype=float)
        values = np.trapezoid(
            data.to_numpy(dtype=float),
            x=data.columns.to_numpy(dtype=float),
            axis=1,
        )
        index = (
            mu
            if angle_axis == "mu"
            else np.rad2deg(np.arccos(np.clip(mu, -1.0, 1.0)))
        )
        return pd.Series(np.abs(values), index=index, name=name)

    def compute_energy_distribution(
        self,
        Ein: float,
        beta_grid: Optional[np.ndarray] = None,
    ) -> pd.Series:
        """Represent the full scattering kernel on a signed beta grid.

        Parameters
        ----------
        Ein : float
            Incident neutron energy in eV.
        beta_grid : np.ndarray, optional
            Requested signed beta coordinates. When omitted, preserve the
            kernel's native outgoing-energy grid, including down-scattering.

        Returns
        -------
        pd.Series
            Kernel values indexed by signed beta. Values outside the native
            support of an explicitly requested beta grid are NaN.

        Notes
        -----
        Beta is defined as (Eout - Ein) / (k_B * T). The kernel ordinates
        retain their original per-outgoing-energy normalization; changing
        the coordinate index does not apply a density Jacobian.
        """
        result = self.result(Ein)
        kernel = self.compute_kernel(result.Ein)

        beta_coords = (
            kernel.index.to_numpy(dtype=float) - result.Ein
        ) / self.kbT
        kernel_values = kernel.to_numpy(dtype=float)

        if beta_grid is None:
            grid, values = beta_coords, kernel_values.copy()
        else:
            grid = np.asarray(beta_grid, dtype=float)
            values = self._interpolate_beta_distribution(beta_coords, kernel_values, grid)

        result.energy_distribution = pd.Series(
            values, index=pd.Index(grid, name="beta"), name=result.Ein,
        )
        return result.energy_distribution

    @staticmethod
    def _interpolate_beta_distribution(
        beta_coords: np.ndarray, kernel_values: np.ndarray, beta_grid: np.ndarray,
    ) -> np.ndarray:
        """Interpolate kernel values onto a user-supplied beta grid.

        Parameters
        ----------
        beta_coords : numpy.ndarray
            Signed beta coordinates of the native kernel.
        kernel_values : numpy.ndarray
            Native ordinates per unit outgoing energy.
        beta_grid : numpy.ndarray
            Strictly increasing requested beta coordinates.

        Returns
        -------
        numpy.ndarray
            Interpolated ordinates with NaN outside the native beta support.
        """
        if (
            beta_grid.ndim != 1
            or beta_grid.size < 2
            or not np.isfinite(beta_grid).all()
            or np.any(np.diff(beta_grid) <= 0.0)
        ):
            raise ValueError(
                "beta_grid must be a finite, strictly increasing "
                "one-dimensional array with at least two points."
            )

        # Points outside the sampled support remain undefined.
        order = np.argsort(beta_coords)
        return np.interp(
            beta_grid, beta_coords[order], kernel_values[order],
            left=np.nan, right=np.nan,
        )

    def compute_ratios(self, Ein: float) -> pd.Series:
        """Compute scattering fractions from the continuous kernel.

        Parameters
        ----------
        Ein : float
            Incident neutron energy in eV.

        Returns
        -------
        pd.Series
            Up- and downscattering fractions and a zero elastic fraction
            for a positive continuous integral. All fractions are NaN
            when the integral is nonpositive.
        """
        result = self.result(Ein)
        if result.ratios is not None:
            return result.ratios

        kernel = self.compute_kernel(result.Ein)
        eout = kernel.index.to_numpy(dtype=float)
        values = kernel.to_numpy(dtype=float)

        total = float(np.trapezoid(values, x=eout))
        if total > 0.0:
            # Both integrals use the zero-transfer point as their shared endpoint.
            up = self._integrate_branch(values, eout, eout >= result.Ein) / total
            down = self._integrate_branch(values, eout, eout <= result.Ein) / total
            elastic = 0.0
        else:
            up = down = elastic = np.nan

        result.ratios = pd.Series(
            {
                "upscattering": up,
                "downscattering": down,
                "elastic": elastic,
            },
            name=result.Ein,
        )
        return result.ratios

    @staticmethod
    def _integrate_branch(
        values: np.ndarray,
        eout: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """Integrate the selected outgoing-energy branch.

        Parameters
        ----------
        values : numpy.ndarray
            Integrand values.
        eout : numpy.ndarray
            Outgoing neutron energies in eV.
        mask : numpy.ndarray
            Boolean mask selecting the integration domain.

        Returns
        -------
        float
            Trapezoidal integral of the branch, or zero if undersampled.
        """
        if np.count_nonzero(mask) <= 1:
            return 0.0
        return float(np.trapezoid(values[mask], x=eout[mask]))

    # ------------------------------------------------------------------
    # Multi-Ein orchestration
    # ------------------------------------------------------------------

    def evaluate(
        self,
        Ein: float,
        *,
        calc_data: bool = False,
        calc_kernel: bool = False,
        calc_angular: bool = False,
        calc_energy: bool = False,
        calc_ratios: bool = False,
        angle_axis: str = "mu",
        beta_grid: Optional[np.ndarray] = None,
    ) -> DsfResult:
        """Populate only the requested fields of one cached result.

        Parameters
        ----------
        Ein : float
            Incident neutron energy in eV.
        calc_data : bool
            Whether to calculate the two-dimensional differential data.
        calc_kernel : bool
            Whether to calculate the scattering kernel.
        calc_angular : bool
            Whether to calculate the angular distribution.
        calc_energy : bool
            Whether to calculate the energy-transfer distribution.
        calc_ratios : bool
            Whether to calculate up- and downscattering fractions.
        angle_axis : str
            Angular-coordinate convention for the distribution.
        beta_grid : Optional[np.ndarray]
            Optional signed dimensionless energy-transfer grid.

        Returns
        -------
        DsfResult
            Cached result populated with the requested quantities.
        """
        result = self.result(Ein)

        if calc_data:
            self.compute_data(result.Ein)
        if calc_kernel:
            self.compute_kernel(result.Ein)
        if calc_angular:
            self.compute_angular_distribution(result.Ein, angle_axis)
        if calc_energy:
            self.compute_energy_distribution(result.Ein, beta_grid)
        if calc_ratios:
            self.compute_ratios(result.Ein)

        return result

    def evaluate_many(
        self,
        ein_arr: Iterable[float],
        **kwargs,
    ) -> Dict[float, DsfResult]:
        """Evaluate the requested observables at several incident energies.

        Parameters
        ----------
        ein_arr : array_like
            Incident neutron energies in eV.
        **kwargs : dict
            Observable flags and coordinate options forwarded to `evaluate`.

        Returns
        -------
        dict of float to DsfResult
            Cached results keyed by incident neutron energy in eV.
        """
        return {
            float(ein): self.evaluate(float(ein), **kwargs)
            for ein in np.atleast_1d(ein_arr)
        }


# ---------------------------------------------------------------------------
# Backward-compatible functional API
# ---------------------------------------------------------------------------


def compute_dsf(
    sab: pd.DataFrame,
    Ein: float,
    T: float,
    mass_amu: float,
    theta: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Evaluate the double-differential scattering data for one energy.

    Parameters
    ----------
    sab : pandas.DataFrame
        Scattering law with alpha rows and beta columns.
    Ein : float
        Incident neutron energy in eV.
    T : float
        Material temperature in kelvin.
    mass_amu : float
        Target mass in atomic mass units.
    theta : array_like or None, optional
        Scattering angles in degrees. The default grid uses integer angles
        from 0 through 180 degrees.

    Returns
    -------
    pandas.DataFrame
        Differential scattering values with cosine rows and outgoing-energy
        columns in eV.
    """
    return Dsf(sab, T, mass_amu, theta).compute_data(Ein)


def compute_scattering_kernel(dsf: pd.DataFrame) -> pd.Series:
    """Integrate a differential scattering table over cosine.

    Parameters
    ----------
    dsf : pandas.DataFrame
        Differential scattering values with cosine rows and outgoing-energy
        columns in eV.

    Returns
    -------
    pandas.Series
        Scattering kernel indexed by outgoing neutron energy in eV.
    """
    return Dsf._kernel_from_data(dsf)


def compute_angular_distribution(
    dsf: pd.DataFrame,
    angle_axis: str = "mu",
) -> pd.Series:
    """Integrate a differential scattering table over outgoing energy.

    Parameters
    ----------
    dsf : pandas.DataFrame
        Differential scattering values with cosine rows and outgoing-energy
        columns in eV.
    angle_axis : str, optional
        Use "mu" (the default) for cosine or another value for the scattering
        angle in degrees. The values remain differential in cosine.

    Returns
    -------
    pandas.Series
        Angular distribution per unit cosine, indexed by the selected
        angular coordinate.
    """
    return Dsf._angular_from_data(dsf, angle_axis=angle_axis)


def compute_ratios(
    sab: pd.DataFrame,
    Ein_arr: np.ndarray,
    T: float,
    mass_amu: float,
) -> pd.DataFrame:
    """Calculate scattering fractions across incident neutron energies.

    Parameters
    ----------
    sab : pandas.DataFrame
        Scattering law with alpha rows and beta columns.
    Ein_arr : array_like
        Incident neutron energies in eV.
    T : float
        Material temperature in kelvin.
    mass_amu : float
        Target mass in atomic mass units.

    Returns
    -------
    pandas.DataFrame
        Up-, down-, and elastic-scattering fractions indexed by incident
        neutron energy in eV.
    """
    model = Dsf(sab, T, mass_amu)
    rows = {
        float(ein): model.compute_ratios(float(ein))
        for ein in np.atleast_1d(Ein_arr)
    }
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "Ein"
    return frame
