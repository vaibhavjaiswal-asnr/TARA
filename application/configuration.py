"""Read and validate application-owned ACE experiment configurations.

TOML and file-system conventions live in this outer application layer. Core
sampling and Processing continue to accept their existing Python objects.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib
from typing import Any

from common.validation import require_existing_file


class ApplicationConfigError(ValueError):
    """Raised when an application TOML file has invalid settings."""


_SCHEMA: dict[str, dict[str, type]] = {
    "workflow": {"backend": str},
    "material": {
        "base_endf_file": str,
        "temperature_k": float,
        "state": str,
        "leapr_file": str,
        "ncmat_file": str,
        "ncrystal_builtin": str,
        "atom_label": str,
        "elastic_mode": str,
    },
    "output": {"directory": str},
    "ensemble": {
        "n_samples": int,
        "include_reference": bool,
        "random_seed": int,
        "batch_size": int,
        "sampling_method": str,
    },
    "execution": {"n_jobs": int, "njoy_executable": str},
    "sampling": {
        "max_energy_shift_mev": float,
        "max_amplitude_shift_pct": float,
        "grid_partition_method": str,
        "automatic_partition_prominence_pct": float,
        "manual_energy_ranges": list,
        "domain_splitting_strategy": str,
        "correlation_strategy": str,
        "variable_coupling": str,
        "shared_grid_structure": bool,
        "freeze_origin": bool,
        "fast_amplitude": bool,
        "amplitude_method": str,
        "spatial_method": str,
        "auto_normalize": bool,
    },
    "diffusion_sampling": {
        "enabled": bool,
        "mode": str,
        "max_diffusivity_shift_pct": float,
        "max_weight_shift_pct": float,
        "weight_compensation": str,
    },
    "ace": {
        "max_energy_ev": float,
        "suffix": str,
        "nbin": int,
        "tolerance": float,
        "save_xsdir": bool,
        "mti": int,
        "mte": int,
    },
}
_REQUIRED = {
    "workflow": {"backend"},
    "material": {"base_endf_file", "temperature_k"},
    "output": {"directory"},
    "ensemble": {"n_samples", "include_reference", "random_seed", "batch_size"},
    "execution": {"n_jobs", "njoy_executable"},
    "sampling": set(_SCHEMA["sampling"]) - {
        "manual_energy_ranges", "automatic_partition_prominence_pct"
    },
    "diffusion_sampling": set(),
    "ace": set(_SCHEMA["ace"]) - {"mti", "mte"},
}


@dataclass(frozen=True)
class AceApplicationConfig:
    """Validated application input for one ACE generation run.

    Parameters
    ----------
    file_path : pathlib.Path
        Absolute path to the input TOML file.
    sections : dict of str to dict
        Validated configuration tables keyed by section name.

    Attributes
    ----------
    backend
        ACE generation backend selected in the workflow table.
    """

    file_path: Path
    sections: dict[str, dict[str, Any]]

    def section(self, name: str) -> dict[str, Any]:
        """Return a shallow copy of one validated configuration section.

        Parameters
        ----------
        name : str
            TOML section name.

        Returns
        -------
        dict
            Section values keyed by setting name.

        Raises
        ------
        KeyError
            If the section is absent.
        """
        return dict(self.sections[name])

    @property
    def backend(self) -> str:
        """Return the requested ACE generation backend.

        Returns
        -------
        str
            Backend name, either "njoy" or "ncrystal".
        """
        return self.sections["workflow"]["backend"]

    def path(self, section: str, key: str) -> Path:
        """Resolve a file-system setting relative to the TOML file.

        Parameters
        ----------
        section, key : str
            Section and setting identifying the path.

        Returns
        -------
        pathlib.Path
            Absolute, resolved path.
        """
        raw = self.sections[section][key]
        return (self.file_path.parent / raw).resolve()


def _check_type(section: str, key: str, value: Any, expected: type) -> None:
    """Check a setting against its declared TOML type.

    Parameters
    ----------
    section, key : str
        TOML section and setting used in error messages.
    value : object
        Parsed TOML value.
    expected : type
        Required type. Float settings also accept finite integers.

    Raises
    ------
    ApplicationConfigError
        If the value has the wrong type or a numeric setting is nonfinite.
    """
    if expected is float:
        good = type(value) in (int, float) and math.isfinite(value)
    else:
        good = type(value) is expected
    if not good:
        raise ApplicationConfigError(
            f"[{section}].{key} must be {expected.__name__}; got {value!r}."
        )


def _number(value: Any, location: str, *, positive: bool = False) -> None:
    """Validate a finite number with an optional strict lower bound.

    Parameters
    ----------
    value : object
        Parsed value to validate.
    location : str
        Setting name used in error messages.
    positive : bool, optional
        Require a number greater than zero when True.

    Raises
    ------
    ApplicationConfigError
        If the value is nonnumeric, nonfinite, or violates the lower bound.
    """
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ApplicationConfigError(f"{location} must be a finite number.")
    if positive and value <= 0:
        raise ApplicationConfigError(f"{location} must be strictly positive.")


def load_ace_config(filename: str | Path) -> AceApplicationConfig:
    """Load one ACE TOML file, validating keys, types, and combinations.

    Parameters
    ----------
    filename : str or pathlib.Path
        TOML input path. Relative file paths within the TOML are resolved
        from the directory containing that file.

    Returns
    -------
    AceApplicationConfig
        Validated application settings.

    Raises
    ------
    FileNotFoundError
        The TOML document does not exist.
    ApplicationConfigError
        The file has invalid keys, types, or combinations.
    """
    source = Path(filename).expanduser().resolve()
    sections = _validate_schema(_read_toml(source))
    diffusion_enabled = _validate_diffusion(sections)

    # Keep the historical prominence for configurations that omit the setting.
    sampling = sections["sampling"]
    sampling.setdefault("automatic_partition_prominence_pct", 1.0)
    _validate_material(sections)
    _validate_positive_controls(sections)
    _validate_perturbation_bounds(sampling, diffusion_enabled)
    _validate_execution_controls(sections)
    _validate_sampling_methods(sampling)
    _validate_manual_ranges(sampling)
    _validate_path_settings(sections)
    return AceApplicationConfig(file_path=source, sections=sections)


def _read_toml(source: Path) -> dict[str, Any]:
    """Read a TOML configuration from disk.

    Parameters
    ----------
    source : pathlib.Path
        Resolved configuration path.

    Returns
    -------
    dict
        Parsed TOML document.

    Raises
    ------
    FileNotFoundError
        If the source is missing or is not a file.
    ApplicationConfigError
        If the TOML syntax is invalid.
    """
    require_existing_file(source, message=f"ACE configuration not found: {source}")
    try:
        with source.open("rb") as stream:
            raw = tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise ApplicationConfigError(f"Malformed TOML in {source}: {exc}") from exc
    return raw


def _validate_schema(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate section names, required keys, and setting types.

    Parameters
    ----------
    raw : dict
        Parsed TOML document.

    Returns
    -------
    dict of str to dict
        Validated shallow copies of the supplied tables.

    Raises
    ------
    ApplicationConfigError
        If a section or setting has an invalid name, shape, or type.
    """
    unknown_sections = set(raw) - set(_SCHEMA)
    missing_sections = (set(_SCHEMA) - {"diffusion_sampling"}) - set(raw)
    if unknown_sections or missing_sections:
        raise ApplicationConfigError(
            f"Invalid TOML sections: unknown={sorted(unknown_sections)}, "
            f"missing={sorted(missing_sections)}."
        )

    # Validate TOML keys and types before checking relationships between settings.
    sections: dict[str, dict[str, Any]] = {}
    for group, fields in _SCHEMA.items():
        if group == "diffusion_sampling" and group not in raw:
            continue
        values = raw[group]
        if not isinstance(values, dict):
            raise ApplicationConfigError(f"[{group}] must be a TOML table.")
        unknown = set(values) - set(fields)
        missing = _REQUIRED[group] - set(values)
        if unknown or missing:
            raise ApplicationConfigError(
                f"Invalid [{group}] settings: unknown={sorted(unknown)}, "
                f"missing={sorted(missing)}."
            )
        for key, value in values.items():
            _check_type(group, key, value, fields[key])
        sections[group] = dict(values)
    return sections


def _validate_diffusion(sections: dict[str, dict[str, Any]]) -> bool:
    """Validate optional diffusion settings and their material requirements.

    Parameters
    ----------
    sections : dict of str to dict
        Schema-validated workflow, material, and sampling settings.

    Returns
    -------
    bool
        Whether diffusion sampling is enabled.

    Raises
    ------
    ApplicationConfigError
        If the diffusion settings conflict with the selected material or sampling.
    """
    # Optional diffusion controls are entirely absent from historical files.
    from core.config import DiffusionSamplingConfig, TaraConfigError
    raw_diffusion = sections.get('diffusion_sampling', {})
    diffusion_values = {
        'enabled': raw_diffusion.get('enabled', False),
        'mode': raw_diffusion.get('mode', 'diffusivity_only'),
        'max_diffusivity_shift_pct': raw_diffusion.get('max_diffusivity_shift_pct', 0.0),
        'max_weight_shift_pct': raw_diffusion.get('max_weight_shift_pct', 0.0),
        'weight_compensation': raw_diffusion.get('weight_compensation'),
    }
    try:
        diffusion = DiffusionSamplingConfig(**diffusion_values)
    except TaraConfigError as exc:
        raise ApplicationConfigError(f"[diffusion_sampling] {exc}") from exc
    if diffusion.enabled and (sections['material'].get('state', 'solid') != 'liquid'
                              or sections['workflow']['backend'] != 'njoy'):
        raise ApplicationConfigError('Diffusion sampling requires NJOY liquid material.')
    # Joint sampling keeps the acoustic origin fixed when energy shifts are enabled.
    if (diffusion.enabled and sections['sampling']['max_energy_shift_mev'] > 0
            and not sections['sampling']['freeze_origin']):
        raise ApplicationConfigError(
            'Diffusion with vibrational energy shifts requires [sampling].freeze_origin = true.'
        )
    if 'diffusion_sampling' in sections:
        sections['diffusion_sampling'] = dict(raw_diffusion)
    return diffusion.enabled


def _validate_material(sections: dict[str, dict[str, Any]]) -> None:
    """Validate material sources and backend-specific reaction requirements.

    Parameters
    ----------
    sections : dict of str to dict
        Schema-validated configuration tables.

    Raises
    ------
    ApplicationConfigError
        If a setting or combination is invalid.
    """
    workflow, material, ensemble = (sections[name] for name in
                                    ("workflow", "material", "ensemble"))
    backend = workflow["backend"]
    if backend not in ("njoy", "ncrystal"):
        raise ApplicationConfigError("[workflow].backend must be 'njoy' or 'ncrystal'.")
    if material.get("state", "solid") not in ("solid", "liquid"):
        raise ApplicationConfigError("[material].state must be solid or liquid.")
    if material.get("state") == "liquid" and "mti" not in sections["ace"]:
        raise ApplicationConfigError("Liquid ACE requires explicit [ace].mti.")
    if material.get("state") == "liquid" and not sections["sampling"]["auto_normalize"]:
        raise ApplicationConfigError("Liquid spectrum sampling requires [sampling].auto_normalize = true.")
    for key in ("mti", "mte"):
        if key in sections["ace"] and (sections["ace"][key] <= 0 if key == "mti" else sections["ace"][key] < 0):
            raise ApplicationConfigError(f"[ace].{key} is not a valid reaction identifier.")
    if backend == "njoy":
        if not material.get("leapr_file"):
            raise ApplicationConfigError("NJOY requires [material].leapr_file.")
        for invalid in ("ncmat_file", "ncrystal_builtin", "atom_label", "elastic_mode"):
            if invalid in material:
                raise ApplicationConfigError(f"NJOY does not accept [material].{invalid}.")
        if not ensemble["include_reference"]:
            raise ApplicationConfigError("NJOY always generates sample_000; include_reference must be true.")
    else:
        sources = sum(bool(material.get(k)) for k in ("ncmat_file", "ncrystal_builtin"))
        if sources != 1:
            raise ApplicationConfigError(
                "NCrystal requires exactly one of [material].ncmat_file and ncrystal_builtin."
            )
        if "leapr_file" in material:
            raise ApplicationConfigError("NCrystal does not accept [material].leapr_file.")
        if not material.get("atom_label") or not material.get("elastic_mode"):
            raise ApplicationConfigError("NCrystal requires atom_label and elastic_mode.")


def _validate_positive_controls(sections: dict[str, dict[str, Any]]) -> None:
    """Validate positive temperature and ACE controls.

    Parameters
    ----------
    sections : dict of str to dict
        Schema-validated configuration tables.

    Raises
    ------
    ApplicationConfigError
        If a setting or combination is invalid.
    """
    for group, key in (("material", "temperature_k"), ("ace", "max_energy_ev"),
                       ("ace", "tolerance")):
        _number(sections[group][key], f"[{group}].{key}", positive=True)


def _validate_perturbation_bounds(sampling: dict[str, Any], diffusion_enabled: bool) -> None:
    """Validate vibrational limits and automatic partition prominence.

    Parameters
    ----------
    sampling : dict
        Vibrational sampling settings.
    diffusion_enabled : bool
        Whether diffusion supplies an active perturbation.

    Raises
    ------
    ApplicationConfigError
        If a setting or combination is invalid.
    """
    for key in ("max_energy_shift_mev", "max_amplitude_shift_pct"):
        value = sampling[key]
        _number(value, f"[sampling].{key}")
        if value < 0:
            raise ApplicationConfigError(f"[sampling].{key} must be nonnegative.")
    if not any(sampling[k] > 0 for k in
               ("max_energy_shift_mev", "max_amplitude_shift_pct")) and not diffusion_enabled:
        raise ApplicationConfigError("At least one sampling perturbation bound must be positive.")
    prominence_pct = sampling["automatic_partition_prominence_pct"]
    if not 0.0 < prominence_pct <= 100.0:
        raise ApplicationConfigError(
            "[sampling].automatic_partition_prominence_pct must be in (0, 100]."
        )


def _validate_execution_controls(sections: dict[str, dict[str, Any]]) -> None:
    """Apply the default design and validate ensemble and execution controls.

    Parameters
    ----------
    sections : dict of str to dict
        Schema-validated configuration tables.

    Raises
    ------
    ApplicationConfigError
        If a setting or combination is invalid.
    """
    ensemble = sections["ensemble"]
    ensemble.setdefault("sampling_method", "latin_hypercube")
    if ensemble["sampling_method"] not in ("latin_hypercube", "monte_carlo"):
        raise ApplicationConfigError(
            "[ensemble].sampling_method must be latin_hypercube or monte_carlo."
        )
    for key in ("n_samples", "batch_size"):
        if ensemble[key] <= 0:
            raise ApplicationConfigError(f"[ensemble].{key} must be positive.")
    if ensemble["random_seed"] < 0 or ensemble["random_seed"] >= 2**32:
        raise ApplicationConfigError("[ensemble].random_seed must be in [0, 2**32).")
    if sections["execution"]["n_jobs"] == 0 or sections["execution"]["n_jobs"] < -1:
        raise ApplicationConfigError("[execution].n_jobs must be -1 or a positive integer.")
    if sections["ace"]["nbin"] <= 0:
        raise ApplicationConfigError("[ace].nbin must be positive.")
    if not sections["ace"]["suffix"].strip():
        raise ApplicationConfigError("[ace].suffix cannot be empty.")


def _validate_sampling_methods(sampling: dict[str, Any]) -> None:
    """Validate the choice of vibrational sampling strategies.

    Parameters
    ----------
    sampling : dict
        Vibrational sampling settings.

    Raises
    ------
    ApplicationConfigError
        If a setting or combination is invalid.
    """
    if sampling["amplitude_method"] not in ("fast", "precise"):
        raise ApplicationConfigError("[sampling].amplitude_method must be fast or precise.")
    if sampling["spatial_method"] not in ("diffeomorphic", "crossfade"):
        raise ApplicationConfigError("[sampling].spatial_method must be diffeomorphic or crossfade.")
    for key, allowed in {
        'grid_partition_method': ('automatic', 'manual'),
        'domain_splitting_strategy': ('fixed', 'deterministic', 'random'),
        'correlation_strategy': ('independent', 'random_group'),
        'variable_coupling': ('independent', 'coupled'),
    }.items():
        if sampling[key] not in allowed:
            raise ApplicationConfigError(
                f"[sampling].{key} must be one of {allowed}."
            )


def _validate_manual_ranges(sampling: dict[str, Any]) -> None:
    """Validate manual energy intervals and their partition mode.

    Parameters
    ----------
    sampling : dict
        Vibrational sampling settings.

    Raises
    ------
    ApplicationConfigError
        If a setting or combination is invalid.
    """
    ranges = sampling.get("manual_energy_ranges", [])
    for idx, item in enumerate(ranges):
        if not (isinstance(item, list) and len(item) == 2):
            raise ApplicationConfigError(f"[sampling].manual_energy_ranges[{idx}] must be [start, end].")
        a, b = item
        _number(a, f"manual_energy_ranges[{idx}][0]")
        _number(b, f"manual_energy_ranges[{idx}][1]")
        if a < 0 or b <= a:
            raise ApplicationConfigError(f"manual_energy_ranges[{idx}] must satisfy 0 <= start < end.")
    if sampling["grid_partition_method"] == "manual" and not ranges:
        raise ApplicationConfigError("Manual grid partition requires manual_energy_ranges.")
    if sampling["grid_partition_method"] != "manual" and ranges:
        raise ApplicationConfigError("manual_energy_ranges requires grid_partition_method='manual'.")


def _validate_path_settings(sections: dict[str, dict[str, Any]]) -> None:
    """Reject blank input, output, and executable path settings.

    Parameters
    ----------
    sections : dict of str to dict
        Schema-validated configuration tables.

    Raises
    ------
    ApplicationConfigError
        If a setting or combination is invalid.
    """
    material = sections["material"]
    for label in ("base_endf_file", "leapr_file", "ncmat_file"):
        if label in material and not material[label].strip():
            raise ApplicationConfigError(f"[material].{label} cannot be empty.")
    if not sections["output"]["directory"].strip():
        raise ApplicationConfigError("[output].directory cannot be empty.")
    if not sections["execution"]["njoy_executable"].strip():
        raise ApplicationConfigError("[execution].njoy_executable cannot be empty.")
