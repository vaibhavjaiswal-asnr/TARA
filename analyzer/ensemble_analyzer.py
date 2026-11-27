"""
Backend-agnostic ensemble orchestrator and statistical aggregator.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

from parallel_execution import run_parallel

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

from common.validation import require_finite_array
from core.data.input_pdos import PhononSpectra
from core.data.material_ensemble import MaterialEnsemble
from core.data.material_scattering import MaterialScatteringModel
from core.data.spectra_ensemble import SpectraEnsemble

from analyzer.core_interfaces import BaseParser, BaseRunner
from analyzer.physics.data_models import PhysicsData


# The reference spectrum is not a stochastic sample. A negative identifier is
# used only while passing it through the existing single-sample backend API.
_REFERENCE_SAMPLE_INDEX = -1


def _isolated_worker(
    index: int,
    spectrum: PhononSpectra | MaterialScatteringModel,
    runner: BaseRunner,
    parser: BaseParser,
    runner_kwargs: Dict[str, Any],
    parser_kwargs: Dict[str, Any],
) -> PhysicsData:
    """
    Execute and parse one spectrum with guaranteed backend cleanup.

    Parameters
    ----------
    index : int
        Stable ensemble sample index used by the runner and parser.
    spectrum : PhononSpectra or MaterialScatteringModel
        Spectrum or full material realization to evaluate.
    runner : BaseRunner
        Backend execution adapter.
    parser : BaseParser
        Backend extraction adapter.
    runner_kwargs : dict
        Keyword arguments forwarded to ``runner.execute``.
    parser_kwargs : dict
        Keyword arguments forwarded to ``parser.parse``.

    Returns
    -------
    PhysicsData
        Parsed physical evaluation for the requested sample.
    """
    # Keep execution and parsing in the same worker so cleanup follows parsing even on
    # failure.
    runner_output = None
    try:
        runner_output = runner.execute(index, spectrum, **runner_kwargs)
        return parser.parse(index, runner_output, **parser_kwargs)
    finally:
        if runner_output is not None:
            runner.cleanup(runner_output)


class EnsembleAnalyzer:
    """
    Analyze vibrational or material realizations through one backend pipeline.

    The analyzer preserves the original stochastic sample indices throughout
    execution, aggregation, and reduction. The baseline spectrum stored in
    ``ensemble.reference`` is evaluated separately and is never inserted into
    the stochastic result tables.

    Parameters
    ----------
    ensemble : SpectraEnsemble or MaterialEnsemble
        Stochastic spectra or complete liquid materials produced by TARA.
    runner : BaseRunner
        Backend execution adapter.
    parser : BaseParser
        Backend extraction adapter.
    sample_indices : sequence of int, optional
        Original ensemble indices to analyze. If omitted, all stochastic
        samples are selected. This argument is primarily used when creating a
        reduced analyzer while retaining the original ensemble.

    Raises
    ------
    TypeError
        If the supplied ensemble has an unsupported type.
    ValueError
        If ``sample_indices`` contains duplicates or indices outside the
        ensemble range.
    """

    def __init__(
        self,
        ensemble: SpectraEnsemble | MaterialEnsemble,
        runner: BaseRunner,
        parser: BaseParser,
        sample_indices: Optional[Sequence[int]] = None,
    ) -> None:
        """Initialize the analyzer with an ensemble and backend adapters.

        Parameters
        ----------
        ensemble : SpectraEnsemble or MaterialEnsemble
            Stochastic spectra or material realizations with a separate reference.
        runner : BaseRunner
            Backend adapter used to evaluate one spectrum.
        parser : BaseParser
            Adapter that extracts physics results from an in-memory backend state.
        sample_indices : sequence of int or None, optional
            Original stochastic sample indices to include in the analysis.

        Raises
        ------
        TypeError
            If the ensemble is neither a SpectraEnsemble nor a MaterialEnsemble.
        ValueError
            If explicit sample indices are empty, repeated, out of range,
            or have more than one dimension.
        """
        if not isinstance(ensemble, (SpectraEnsemble, MaterialEnsemble)):
            raise TypeError("ensemble must be a SpectraEnsemble instance or a MaterialEnsemble.")

        self.ensemble = ensemble
        self.runner = runner
        self.parser = parser
        # Retain the original sample identifiers when selecting or reducing the ensemble.
        self.sample_indices = self._validate_sample_indices(sample_indices)

        # Stochastic results and baseline results are intentionally separated so
        # Sample_Index remains aligned with ensemble.pdos and sample_info.
        self.raw_results: List[PhysicsData] = []
        self.reference_result: Optional[PhysicsData] = None

        self.thermo_df: Optional[pd.DataFrame] = None
        self.xs_total_df: Optional[pd.DataFrame] = None
        self.xs_inelastic_df: Optional[pd.DataFrame] = None

    def _validate_sample_indices(
        self,
        sample_indices: Optional[Sequence[int]],
    ) -> np.ndarray:
        """
        Validate and normalize the selected stochastic sample indices.

        Parameters
        ----------
        sample_indices : sequence of int or None
            Requested original ensemble indices. ``None`` selects every sample.

        Returns
        -------
        numpy.ndarray
            One-dimensional integer array of stable ensemble indices.

        Raises
        ------
        ValueError
            If explicit indices are empty, repeated, out of range, or not one-dimensional.
        """
        if sample_indices is None:
            return np.arange(len(self.ensemble), dtype=np.int64)

        indices = np.asarray(sample_indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("sample_indices must be a non-empty one-dimensional sequence.")
        if np.any(indices < 0) or np.any(indices >= len(self.ensemble)):
            raise ValueError("sample_indices contains an index outside the ensemble range.")
        if np.unique(indices).size != indices.size:
            raise ValueError("sample_indices must not contain duplicate indices.")

        return indices.copy()

    def compute(
        self,
        parallel: bool = True,
        runner_kwargs: Optional[Dict[str, Any]] = None,
        parser_kwargs: Optional[Dict[str, Any]] = None,
        compute_reference: bool = True,
    ) -> None:
        """
        Execute backend calculations for the selected stochastic samples.

        Parameters
        ----------
        parallel : bool, default=True
            If True, evaluate stochastic samples with joblib using the available
            worker allocation determined by :mod:`parallel_execution`.
        runner_kwargs : dict, optional
            Keyword arguments forwarded to every runner execution.
        parser_kwargs : dict, optional
            Keyword arguments forwarded to every parser call.
        compute_reference : bool, default=True
            If True and ``ensemble.reference`` is available, evaluate the
            baseline separately and store it in ``reference_result``.

        Notes
        -----
        This method updates `raw_results`, `reference_result`, and the
        available aggregate attributes in place. Stochastic rows retain
        their original ensemble indices. The reference is stored separately.
        """
        runner_kwargs = dict(runner_kwargs or {})
        parser_kwargs = parser_kwargs or {}
        # Resolve backend requirements once before dispatching samples to workers.
        if isinstance(self.runner, BaseRunner):
            runner_kwargs = self.runner.prepare_analysis(runner_kwargs, parser_kwargs)

        self.raw_results = self._compute_stochastic_results(
            parallel=parallel,
            runner_kwargs=runner_kwargs,
            parser_kwargs=parser_kwargs,
        )

        self.reference_result = None
        if compute_reference:
            self.reference_result = self._compute_reference_result(runner_kwargs, parser_kwargs)

        self._aggregate_dataframes()

    def _compute_reference_result(
        self, runner_kwargs: Dict[str, Any], parser_kwargs: Dict[str, Any],
    ) -> Optional[PhysicsData]:
        """Evaluate the unperturbed reference through the prepared backend.

        Parameters
        ----------
        runner_kwargs : dict
            Prepared execution options for the backend runner.
        parser_kwargs : dict
            Observable options for the backend parser.

        Returns
        -------
        PhysicsData or None
            Reference evaluation, or None if the ensemble has no reference.
        """
        reference = self.ensemble.reference
        if reference is None:
            return None
        return _isolated_worker(
            _REFERENCE_SAMPLE_INDEX, reference, self.runner, self.parser,
            runner_kwargs, parser_kwargs,
        )

    @staticmethod
    def _prepare_series_for_alignment(series: pd.Series) -> pd.Series:
        """
        Normalize one kinematic Series for ensemble-level alignment.

        Duplicate coordinates are collapsed by averaging their values. The
        coordinate order of first occurrence is retained so the selected
        canonical grid preserves the orientation produced by the physics
        calculation.

        Parameters
        ----------
        series : pandas.Series
            One-dimensional kinematic result indexed by numeric coordinates.

        Returns
        -------
        pandas.Series
            Numeric, finite Series with a unique coordinate index.

        Raises
        ------
        ValueError
            If coordinates or values are non-numeric, non-finite, or have
            inconsistent lengths.
        """
        try:
            coordinates = np.asarray(series.index, dtype=np.float64)
            values = np.asarray(series.values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Kinematic result coordinates and values must be numeric."
            ) from exc

        if coordinates.size != values.size:
            raise ValueError(
                "Kinematic result coordinates and values must have matching lengths."
            )
        require_finite_array(coordinates, message="Kinematic result coordinates must be finite.")
        require_finite_array(values, message="Kinematic result values must be finite.")

        prepared = pd.Series(values, index=pd.Index(coordinates))
        if not prepared.index.is_unique:
            prepared = prepared.groupby(level=0, sort=False).mean()

        return prepared

    @staticmethod
    def _align_series_collection(
        series_by_sample: Sequence[tuple[int, pd.Series]],
    ) -> pd.DataFrame:
        """
        Align sample-dependent one-dimensional results onto one existing grid.

        Each input Series is first normalized to a unique coordinate axis by
        averaging exact duplicate coordinates. The longest cleaned grid already
        present in the collection is then used as the canonical coordinate
        system. Other series are linearly interpolated onto that grid, with zero
        outside their original support.

        Parameters
        ----------
        series_by_sample : sequence of tuple of int and pandas.Series
            Pairs of original ensemble sample index and one-dimensional
            kinematic result.

        Returns
        -------
        pandas.DataFrame
            Sample-indexed matrix with a unique canonical coordinate axis.

        Raises
        ------
        ValueError
            If no series are supplied or a series cannot be normalized to a
            finite numeric coordinate/value representation.
        """
        if not series_by_sample:
            raise ValueError("At least one series is required for alignment.")

        prepared_series = [
            (
                int(sample_index),
                EnsembleAnalyzer._prepare_series_for_alignment(series),
            )
            for sample_index, series in series_by_sample
        ]

        # Use the existing coordinate set with the most points as the target grid.
        canonical_series = max(
            (series for _, series in prepared_series),
            key=lambda series: series.size,
        )
        canonical_index = canonical_series.index.copy()
        return EnsembleAnalyzer._stack_interpolated_series(prepared_series, canonical_index)

    @staticmethod
    def _stack_interpolated_series(
        series_by_sample: Sequence[tuple[int, pd.Series]], target_index: pd.Index,
    ) -> pd.DataFrame:
        """Interpolate cleaned series and stack them on the target grid.

        Parameters
        ----------
        series_by_sample : sequence of tuple of int and pandas.Series
            Original sample indices paired with cleaned coordinate series.
        target_index : pandas.Index
            Target coordinates in the order used for the output columns.

        Returns
        -------
        pandas.DataFrame
            Rows in input sample order, with zero outside each sample's
            original coordinate support.
        """
        target_grid = target_index.to_numpy(dtype=np.float64, copy=False)
        matrix = np.empty((len(series_by_sample), target_grid.size), dtype=np.float64)
        for row, (_, series) in enumerate(series_by_sample):
            matrix[row] = EnsembleAnalyzer._interpolate_clean_series(series, target_grid)

        return pd.DataFrame(
            matrix,
            index=pd.Index([index for index, _ in series_by_sample], name="Sample_Index"),
            columns=target_index,
        )

    @staticmethod
    def _interpolate_clean_series(series: pd.Series, target_grid: np.ndarray) -> np.ndarray:
        """Interpolate one cleaned series without extrapolating its support.

        Parameters
        ----------
        series : pandas.Series
            Series with unique coordinates and numerical values.
        target_grid : numpy.ndarray
            Destination numeric coordinate grid.

        Returns
        -------
        numpy.ndarray
            Interpolated values, zero outside the original support.
        """
        source_grid = series.index.to_numpy(dtype=np.float64, copy=False)
        if source_grid.size == 0:
            return np.zeros(target_grid.size, dtype=np.float64)
        order = np.argsort(source_grid, kind="stable")
        return np.interp(
            target_grid,
            source_grid[order],
            series.to_numpy(dtype=np.float64, copy=False)[order],
            left=0.0, right=0.0,
        )

    def _compute_stochastic_results(
        self,
        *,
        parallel: bool,
        runner_kwargs: Dict[str, Any],
        parser_kwargs: Dict[str, Any],
    ) -> List[PhysicsData]:
        """Evaluate selected stochastic samples through the shared runner.

        Parameters
        ----------
        parallel : bool
            Use the worker count selected by the parallel executor when True.
        runner_kwargs : dict
            Arguments for the backend runner.
        parser_kwargs : dict
            Arguments for the backend parser.

        Returns
        -------
        list of PhysicsData
            Results ordered by the original sample selection.
        """
        return run_parallel(
            _isolated_worker,
            (
                ((int(index), self.ensemble[int(index)], self.runner,
                  self.parser, runner_kwargs, parser_kwargs), {})
                for index in self.sample_indices
            ),
            n_jobs=-1 if parallel else 1,
        )

    def _aggregate_dataframes(self) -> None:
        """Build ensemble tables and per-energy mappings from stochastic results.

        Notes
        -----
        Updates aggregate attributes from `raw_results`. An empty result
        list leaves the existing attributes untouched.
        """
        if not self.raw_results:
            return

        self._aggregate_thermodynamics()
        self._aggregate_integral_1d()
        self._aggregate_dict_multi_ein()
        self._aggregate_dataframe_multi_ein()

    def _aggregate_thermodynamics(self) -> None:
        """Build the thermodynamics table from the current stochastic results.

        Notes
        -----
        Sets `thermo_df` with one row per original sample index. A scope
        is attached only when every result reports the same scope.
        """
        thermo_list = []
        for result in self.raw_results:
            thermo = result.thermo
            thermo_list.append(
                {
                    "Sample_Index": result.sample_index,
                    "teff": thermo.teff,
                    "msd": thermo.msd,
                    "cv": thermo.cv,
                    "entropy": thermo.entropy,
                    "zero_point_energy": thermo.zero_point_energy,
                    "free_energy": thermo.free_energy,
                }
            )

        self.thermo_df = pd.DataFrame(thermo_list).set_index("Sample_Index")
        scopes = {(result.metadata or {}).get("thermodynamics_scope")
                  for result in self.raw_results}
        if len(scopes) == 1 and None not in scopes:
            self.thermo_df.attrs["scope"] = scopes.pop()

    def _aggregate_integral_1d(self) -> None:
        """Stack available cross sections and scattering ratios by sample index.

        Notes
        -----
        Stores each available observable in its corresponding ``*_df``
        attribute, using the incident-energy grid supplied by the parser.
        """
        integral_1d_attributes = [
            "xs_total",
            "xs_inelastic",
            "xs_coherent_elastic",
            "xs_incoherent_elastic",
            "xs_absorption",
            "upscattering_ratio",
            "downscattering_ratio",
            "elastic_ratio",
        ]

        for attr in integral_1d_attributes:
            series = self._integral_outputs(attr)
            if series:
                setattr(self, f"{attr}_df", self._stack_shared_grid(series))

    def _integral_outputs(self, attribute: str) -> list[tuple[int, Any]]:
        """Collect the available values of one integral observable.

        Parameters
        ----------
        attribute : str
            Name of the field on each sample's integral result.

        Returns
        -------
        list of tuple
            Original sample indices paired with available values in result order.
        """
        outputs = []
        for result in self.raw_results:
            value = getattr(result.integ, attribute)
            if value is not None:
                outputs.append((result.sample_index, value))
        return outputs

    @staticmethod
    def _stack_shared_grid(series: Sequence[tuple[int, pd.Series]]) -> pd.DataFrame:
        """Stack one-dimensional outputs evaluated on a common grid.

        Parameters
        ----------
        series : sequence of tuple
            Nonempty sequence of sample indices paired with their Series.

        Returns
        -------
        pandas.DataFrame
            Sample-indexed values on the first Series' coordinate grid.
        """
        indices, values = zip(*series)
        # These observables already share a grid, so stack without interpolation.
        return pd.DataFrame(
            np.array([value.values for value in values]),
            index=pd.Index(indices, name="Sample_Index"),
            columns=values[0].index,
        )

    def _aggregate_dict_multi_ein(self) -> None:
        """Align scattering kernels and angular distributions by incident energy.

        Notes
        -----
        Populates `scattering_kernel_dict` and `angular_distribution_dict`
        for the observables present in the stochastic results.
        """
        for attribute in ("scattering_kernel", "angular_distribution"):
            self._aggregate_by_incident_energy(attribute, from_frame=False)

    def _aggregate_dataframe_multi_ein(self) -> None:
        """Align cosine and energy distributions by incident energy.

        Notes
        -----
        Populates `cosine_distribution_dict` and `energy_distribution_dict`
        for the observables present in the stochastic results.
        """
        for attribute in ("cosine_distribution", "energy_distribution"):
            self._aggregate_by_incident_energy(attribute, from_frame=True)

    def _aggregate_by_incident_energy(self, attribute: str, *, from_frame: bool) -> None:
        """Build one sample-aligned matrix per available incident energy.

        Parameters
        ----------
        attribute : str
            Integral observable to aggregate.
        from_frame : bool
            Read incident energies from DataFrame rows when True, or mapping
            keys otherwise.

        Notes
        -----
        Stores a mapping from incident energy to a sample-indexed DataFrame
        in the corresponding ``*_dict`` attribute. Samples contribute only
        at incident energies present in their result.
        """
        outputs = self._integral_outputs(attribute)
        if not outputs:
            return

        by_energy: dict[float, list[tuple[int, pd.Series]]] = {}
        for sample_index, data in outputs:
            if from_frame:
                # An outer join leaves NaN where a sample has no native support.
                rows = ((energy, data.loc[energy].dropna()) for energy in dict.fromkeys(data.index))
            else:
                rows = data.items()
            for energy, series in rows:
                by_energy.setdefault(energy, []).append((sample_index, series))

        setattr(self, f"{attribute}_dict", {
            energy: self._align_series_collection(by_energy[energy])
            for energy in sorted(by_energy)
        })

    def extract(self, keys: List[str]) -> Any:
        """
        Extract explicitly requested aggregated properties.

        Parameters
        ----------
        keys : list of str
            Property names such as "thermo", "xs_total", or "scattering_kernel".

        Returns
        -------
        EnsemblePhysicsData
            Available requested properties and any sample or reference metadata.

        Notes
        -----
        Missing, empty, or unrecognized properties are reported through the
        logger and omitted from the result.
        """
        from .physics.data_models import EnsemblePhysicsData

        ensemble_data = EnsemblePhysicsData(metadata=self._result_metadata())
        for key in keys:
            output = self._get_aggregated_output(key)
            if output is not None:
                setattr(ensemble_data, key, output)
        return ensemble_data

    def _result_metadata(self) -> dict | None:
        """Collect available metadata while keeping the reference separate.

        Returns
        -------
        dict or None
            Metadata indexed by stochastic sample plus a separate reference
            entry, or None if neither source has metadata.
        """
        samples = {
            result.sample_index: result.metadata
            for result in self.raw_results
            if result.metadata is not None
        }
        reference = None if self.reference_result is None else self.reference_result.metadata
        if samples or reference is not None:
            return {"samples": samples, "reference": reference}
        return None

    def _get_aggregated_output(self, key: str) -> pd.DataFrame | dict | None:
        """Resolve one requested property and report unavailable results.

        Parameters
        ----------
        key : str
            Property name without the internal storage suffix.

        Returns
        -------
        pandas.DataFrame, dict, or None
            Available result, with DataFrame storage taking precedence when
            present. Returns None for empty or unrecognized results.
        """
        frame_name = f"{key}_df"
        mapping_name = f"{key}_dict"
        if hasattr(self, frame_name):
            frame = getattr(self, frame_name)
            if frame is not None and not frame.empty:
                return frame
        elif hasattr(self, mapping_name):
            mapping = getattr(self, mapping_name)
            if mapping:
                return mapping
        else:
            logging.warning("Key '%s' is not a recognized computed attribute.", key)
            return None

        logging.warning("Key '%s' was requested but is empty.", key)
        return None

    def reduce(
        self,
        sample_indices: Sequence[int],
    ) -> "EnsembleAnalyzer":
        """
        Return an analyzer restricted to selected stochastic samples.

        Original ensemble indices and provenance are preserved.
        Previously computed results are reused when available.

        Parameters
        ----------
        sample_indices : sequence of int
            Original ensemble indices to retain, in the requested order.
            Every index must belong to the current analyzer selection.

        Returns
        -------
        EnsembleAnalyzer
            Analyzer sharing the source ensemble, backend adapters, and
            reference result, with stochastic results in the requested order.

        Raises
        ------
        ValueError
            If indices are empty, repeated, out of range, not one-dimensional,
            or outside the current analyzer selection.
        """
        indices = self._validate_sample_indices(sample_indices)

        if not np.all(np.isin(indices, self.sample_indices)):
            raise ValueError(
                "Selected indices must belong to the current analyzer."
            )

        reduced = EnsembleAnalyzer(
            ensemble=self.ensemble,
            runner=self.runner,
            parser=self.parser,
            sample_indices=indices,
        )

        if self.raw_results:
            reduced.raw_results = self._select_cached_results(indices)
            reduced._aggregate_dataframes()

        # Reference is independent of stochastic sample selection.
        reduced.reference_result = self.reference_result
        return reduced

    def _select_cached_results(self, indices: Sequence[int]) -> list[PhysicsData]:
        """Select completed calculations in the requested sample order.

        Parameters
        ----------
        indices : sequence of int
            Original ensemble indices to retain.

        Returns
        -------
        list of PhysicsData
            Cached calculations for the requested samples that were evaluated.
        """
        results = {result.sample_index: result for result in self.raw_results}
        return [results[int(index)] for index in indices if int(index) in results]
