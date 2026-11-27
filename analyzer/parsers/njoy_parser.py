"""
Extraction adapter for the NJOY backend.
"""
import numpy as np
import pandas as pd
from scipy.constants import physical_constants, k, hbar

from common.validation import require_same_shape
from analyzer.parsers.base_tsl_parser import BaseTslParser
from analyzer.physics.data_models import PhysicsData, ThermodynamicsData, DifferentialData, IntegralData
from analyzer.backends.njoy_engine import NjoyEngine
from analyzer.physics.kinematics import decode_njoy_mf7_sab

from analyzer.physics.thermodynamics import (
    _with_debye_tail,
    calc_teff_numba, calc_msd_numba, calc_cv_numba, calc_entropy_numba,
    calc_zero_point_energy_numba, calc_free_energy_numba,
    calc_internal_energy_numba, calc_log_moment_numba
)


class NjoyParser(BaseTslParser):
    """Extract NJOY ENDF observables into the shared Analyzer physics schema."""

    def parse(self, sample_index: int, engine: NjoyEngine, **kwargs) -> PhysicsData:
        """Parse the requested observables from one evaluated sample.

        Parameters
        ----------
        sample_index : int
            Identifier of the stochastic sample being evaluated.
        engine : NjoyEngine
            In-memory backend state from which the requested data are extracted.
        **kwargs : Any
            Observable flags. Set ``calc_thermodynamics=False`` for cross-section-only studies.

        Returns
        -------
        PhysicsData
            Requested observables from the available NJOY tapes.
        """
        thermo = self._thermodynamics_for(engine, kwargs.get("calc_thermodynamics", True))
        diff = DifferentialData()
        integ = IntegralData()

        self._extract_cross_sections(engine, integ, **kwargs)
        self._extract_kinematics(engine, diff, integ, **kwargs)

        return PhysicsData(sample_index=sample_index, thermo=thermo, diff=diff, integ=integ)

    def _extract_thermodynamics(self, engine: NjoyEngine) -> ThermodynamicsData:
        """Calculate thermodynamic observables from the evaluated spectrum.

        Parameters
        ----------
        engine : NjoyEngine
            In-memory backend state from which the requested data are extracted.

        Returns
        -------
        ThermodynamicsData
            Spectrum-derived thermodynamic and displacement quantities.
        """
        pdos, E = _with_debye_tail(engine.pdos, engine.energy_grid)
        temp = engine.temperature
        mass_amu = engine.mass_amu

        teff = calc_teff_numba(pdos, E, temp)
        msd = calc_msd_numba(pdos, E, temp, mass_amu)

        debye_waller, b_factor = self._displacement_factors(msd, mass_amu, temp)
        return ThermodynamicsData(
            teff=teff,
            msd=msd,
            debye_waller=debye_waller,
            b_factor=b_factor,
            cv=calc_cv_numba(pdos, E, temp),
            entropy=calc_entropy_numba(pdos, E, temp),
            zero_point_energy=calc_zero_point_energy_numba(pdos, E),
            free_energy=calc_free_energy_numba(pdos, E, temp),
            internal_energy=calc_internal_energy_numba(pdos, E, temp),
            log_moment=calc_log_moment_numba(pdos, E),
        )

    @staticmethod
    def _displacement_factors(msd, mass_amu, temperature):
        """Calculate the Debye–Waller coefficient and crystallographic B factor.

        Parameters
        ----------
        msd : float
            Mean-square displacement in square ångströms.
        mass_amu : float
            Target mass in atomic mass units.
        temperature : float
            Material temperature in kelvin.

        Returns
        -------
        tuple of float
            Dimensionless Debye–Waller coefficient and B factor in square
            ångströms.
        """
        if not msd:
            return 0.0, 0.0

        # Convert displacement and mass to SI for the coefficient.
        msd_m2 = msd * (1e-10)**2
        mass_kg = mass_amu * physical_constants['atomic mass constant'][0]
        debye_waller = msd_m2 * (2 * mass_kg * k * temperature / hbar**2)
        b_factor = 8 * np.pi**2 * msd / 3
        return debye_waller, b_factor

    def _extract_cross_sections(self, engine: NjoyEngine, integ: IntegralData, **kwargs) -> None:
        """Extract available thermal-scattering MF3 cross sections.

        THERMR writes the inelastic cross section at MT=221 and, when
        present, its elastic contribution at MT=222 for TARA's analysis
        deck. MF7/MT2 LTHR identifies whether MT=222 is coherent (1),
        incoherent (2), or mixed (3). A missing elastic section is not
        silently treated as zero when MF7 metadata expects elastic data.

        ``xs_total`` is the sum of *thermal scattering* contributions,
        not an evaluated all-reaction or absorption-inclusive total.

        Parameters
        ----------
        engine : NjoyEngine
            In-memory NJOY tape trees and material information.
        integ : IntegralData
            Destination for the available integrated cross sections.
        **kwargs : Any
            ``energy_grid`` and cross-section calculation flags.
        """
        if not engine.thermr_tree:
            return

        target_egrid = np.asarray(
            kwargs.get('energy_grid', np.geomspace(1e-5, 10.0, 1000)),
            dtype=float,
        )
        mf3 = engine.thermr_tree.get(3, {})
        inelastic = self._thermr_xs(mf3, 221, target_egrid)
        elastic = self._thermr_xs(mf3, 222, target_egrid)
        elastic_type = self._thermal_elastic_type(engine)

        if kwargs.get('calc_xs_inelastic', True):
            integ.xs_inelastic = inelastic

        if kwargs.get('calc_xs_elastic', True) and elastic is not None:
            self._store_elastic_channel(integ, elastic, elastic_type)

        if kwargs.get('calc_xs_total', True):
            total = self._thermal_scattering_total(inelastic, elastic, elastic_type)
            if total is not None:
                integ.xs_total = total

    @staticmethod
    def _store_elastic_channel(integ, elastic, elastic_type):
        """Store the elastic curve when its scattering type is resolved.

        Parameters
        ----------
        integ : IntegralData
            Destination for the resolved elastic channel.
        elastic : pandas.Series
            THERMR MT=222 cross section in barns.
        elastic_type : int
            LEAPR elastic type from MF7/MT2.
        """
        # A single MT=222 curve for LTHR=3 combines both elastic components.
        if elastic_type == 1:
            integ.xs_coherent_elastic = elastic
        elif elastic_type == 2:
            integ.xs_incoherent_elastic = elastic

    @staticmethod
    def _thermal_scattering_total(inelastic, elastic, elastic_type):
        """Combine THERMR contributions when the available curves define a total.

        Parameters
        ----------
        inelastic, elastic : pandas.Series or None
            THERMR inelastic and elastic cross sections in barns.
        elastic_type : int
            LEAPR elastic type, or zero when MF7/MT2 is absent.

        Returns
        -------
        pandas.Series or None
            Total thermal scattering cross section when available.
        """
        if inelastic is None:
            return None
        if elastic is not None:
            return inelastic + elastic
        if elastic_type == 0:
            # An absent MF7/MT2 marks a model with no elastic contribution.
            return inelastic.copy()
        return None

    @staticmethod
    def _thermr_xs(mf3, mt: int, energy_grid: np.ndarray):
        """Interpolate one available THERMR reaction on the analysis grid.

        Parameters
        ----------
        mf3 : dict
            THERMR MF3 sections, keyed by MT number.
        mt : int
            Reaction number to retrieve.
        energy_grid : numpy.ndarray
            Incident-neutron energy grid in eV.

        Returns
        -------
        pandas.Series or None
            Interpolated cross section in barns, or None if absent.
        """
        if mt not in mf3:
            return None
        section = mf3[mt]['xstable']
        source_energy = np.asarray(section['E'], dtype=float)
        source_xs = np.asarray(section['xs'], dtype=float)
        values = np.interp(
            energy_grid, source_energy, source_xs, left=0.0, right=0.0
        )
        return pd.Series(values, index=energy_grid)

    @staticmethod
    def _thermal_elastic_type(engine: NjoyEngine) -> int:
        """Read MF7/MT2 LTHR (0=none, 1=coherent, 2=incoherent, 3=mixed).

        Parameters
        ----------
        engine : NjoyEngine
            NJOY engine with its original LEAPR MF7 tape.

        Returns
        -------
        int
            LEAPR elastic-scattering type.

        Raises
        ------
        ValueError
            If an elastic section lacks a supported LTHR value.
        """
        elastic = (engine.leapr_tree or {}).get(7, {}).get(2)
        if elastic is None:
            return 0
        try:
            lthr = int(elastic['LTHR'])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError('MF7/MT2 has no valid LTHR metadata.') from exc
        if lthr not in (1, 2, 3):
            raise ValueError(f'Unsupported MF7/MT2 LTHR={lthr}.')
        return lthr

    def _extract_kinematics(self, engine: NjoyEngine, diff: DifferentialData, integ: IntegralData, **kwargs) -> None:
        """Decode MF7 S(alpha,beta) using LASYM/LLN before shared kinematics.

        Parameters
        ----------
        engine : NjoyEngine
            In-memory backend state from which the requested data are extracted.
        diff : DifferentialData
            Container receiving differential scattering data.
        integ : IntegralData
            Container receiving integrated scattering quantities.
        **kwargs : Any
            Additional options accepted by the current interface.

        Raises
        ------
        ValueError
            If the input or required state violates the operation contract.
        """
        if not engine.leapr_tree:
            return
        if not (kwargs.get('calc_sab', False) or any(self._requested_outputs(kwargs).values())):
            return

        section = engine.leapr_tree.get(7, {}).get(4, {})
        table = self._decode_thermal_law(section)
        if table is not None:
            # The decoder has already expanded the law to full signed beta.
            self._apply_universal_kinematics(
                table, diff, integ, engine.temperature, engine.mass_amu, **kwargs,
            )

    @classmethod
    def _decode_thermal_law(cls, section):
        """Decode the available MF7/MT4 records into a physical scattering law.

        Parameters
        ----------
        section : dict
            Parsed MF7/MT4 section from the LEAPR output tape.

        Returns
        -------
        pandas.DataFrame or None
            Full scattering law with alpha rows and signed beta columns.
        """
        beta_table = section.get('beta', {})
        records = section.get('S_table', {})
        # Pair beta coordinates and scattering-law records by ENDF record index.
        record_keys = sorted(set(beta_table) & set(records))
        if not record_keys:
            return None

        lasym, lln = cls._thermal_law_flags(section)
        beta = np.array([beta_table[key] for key in record_keys], dtype=float)
        alpha, values = cls._stack_thermal_law_columns(records, record_keys)
        table = pd.DataFrame(
            values, index=pd.Index(alpha, name='alpha'), columns=pd.Index(beta, name='beta'),
        )
        return decode_njoy_mf7_sab(table, lasym=lasym, lln=lln)

    @staticmethod
    def _thermal_law_flags(section):
        """Read the symmetry and logarithmic-storage flags from MF7/MT4.

        Parameters
        ----------
        section : dict
            Parsed MF7/MT4 section.

        Returns
        -------
        tuple of int
            LASYM and LLN representation flags.
        """
        # The output tape records the actual representation used by LEAPR.
        try:
            lasym = int(section['LASYM'])
            lln = int(section['LLN'])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError('NJOY MF7/MT4 requires valid LASYM and LLN metadata.') from exc
        if lasym not in (0, 1) or lln not in (0, 1):
            raise ValueError('NJOY MF7/MT4 LASYM and LLN must be 0 or 1.')
        return lasym, lln

    @staticmethod
    def _stack_thermal_law_columns(records, record_keys):
        """Collect beta records that share a single alpha axis.

        Parameters
        ----------
        records : dict
            Scattering-law records keyed by their ENDF indices.
        record_keys : sequence of int
            Record keys in beta-column order.

        Returns
        -------
        tuple of numpy.ndarray
            Common alpha axis and scattering-law matrix.
        """
        alpha = np.array(records[record_keys[0]]['alpha'], dtype=float)
        columns = []
        for key in record_keys:
            coordinates = np.array(records[key]['alpha'], dtype=float)
            values = np.array(records[key]['S'], dtype=float)
            if coordinates.shape != alpha.shape or not np.allclose(coordinates, alpha):
                raise ValueError("Inconsistent alpha grid across NJOY MF7/MT4 beta records.")
            require_same_shape(values, alpha, message="NJOY MF7/MT4 S(alpha,beta) column does not match alpha grid.")
            columns.append(values)
        return alpha, np.column_stack(columns)
