"""Fresh-process native extraction; invoked only by the Processing backend."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys

import NCrystal as NC
import numpy as np

from input_output.ncrystal.liquids import parse_custom_liquids
from processing.backends.liquid_capture.payload import read_sabtap_arrays
from processing.backends.ncmat_isolation import isolate_ncmat_dyninfo


def capture(request_path):
    """Capture kernels and native cross sections for one isolated material.

    Parameters
    ----------
    request_path : path-like
        JSON request in the sample sandbox.
    """
    root, request, text, settings, cfg = _read_capture_inputs(request_path)
    _verify_native_library(request['library_path'])
    # Native dynamics hold weak references to Info, so keep it for the capture.
    info = NC.createInfo(cfg)
    dynamics, active = _active_dynamics(info, settings, request)
    energies = np.loadtxt(root / "energies.txt", ndmin=1)
    mat, arrays, components, sum_xs, selected_xs = _capture_liquid_components(
        root, cfg, settings, dynamics, request['temperature'], energies, active,
    )
    _include_free_gas(
        root, text, dynamics.keys() - settings.keys(), request['temperature'],
        energies, active, sum_xs, selected_xs,
    )
    _finalize_cross_sections(mat, arrays, sum_xs, selected_xs)
    target = dynamics[request['target']]
    arrays['backend_energy_grid'] = target.vdos_egrid_expanded
    arrays['backend_pdos'] = target.vdos_density
    _write_capture_result(root, arrays, components, target, text, active, request)


def _read_capture_inputs(request_path):
    """Read the capture request and its source material.

    Parameters
    ----------
    request_path : path-like
        JSON request in the sample sandbox.

    Returns
    -------
    tuple
        Sandbox path, request, source NCMAT text, liquid settings by label,
        and native configuration string.

    Raises
    ------
    RuntimeError
        If the installed NCrystal version is incompatible with capture.
    """
    request_path = Path(request_path)
    root = request_path.parent
    request = json.loads(request_path.read_text())
    if NC.__version__ != "4.4.6":
        raise RuntimeError("Unverified NCrystal ABI; expected 4.4.6")
    material = root / "material.ncmat"
    text = material.read_text()
    settings = {item.atom: item for item in parse_custom_liquids(text)}
    cfg = f"{material};temp={request['temperature']:.17g}K;vdoslux={request['vdoslux']}"
    return root, request, text, settings, cfg


def _capture_liquid_components(root, cfg, settings, dynamics, temperature, energies, active):
    """Load the liquid model and read the components written by the capture hook.

    Parameters
    ----------
    root : pathlib.Path
        Sample sandbox containing the captured files.
    cfg : str
        NCrystal configuration for the full material.
    settings, dynamics : dict
        Custom liquid settings and native dynamics keyed by species label.
    temperature : float
        Requested temperature in kelvin.
    energies : numpy.ndarray
        Incident energies in eV.
    active : set of str
        Species included in the selected scattering cross section.

    Returns
    -------
    tuple
        Loaded material, output arrays, component metadata, and the full
        and selected liquid scattering totals in barns.

    Raises
    ------
    RuntimeError
        If the native model or number of captured kernels is unexpected.
    """
    mat = NC.load(cfg)
    if "LiquidsModel" not in mat.scatter.getName():
        raise RuntimeError("LiquidsModel was not selected")
    # Disable the dump hook before evaluating any isolated free-gas species.
    os.environ.pop("NCRYSTAL_SAB_DUMP_DIR", None)

    files = sorted(root.glob("sab_*.json"))
    if len(files) != len(settings):
        raise RuntimeError(f"Expected {len(settings)} kernels, captured {len(files)}")
    arrays = {"xs_energy": energies, "xs_absorption": mat.absorption.xsect(energies)}
    components, sum_xs, selected_xs = _read_captured_kernels(
        files, arrays, settings, dynamics, temperature, energies, active,
    )
    return mat, arrays, components, sum_xs, selected_xs


def _finalize_cross_sections(mat, arrays, sum_xs, selected_xs):
    """Validate the component sum and store the selected scattering values.

    Parameters
    ----------
    mat : object
        Loaded NCrystal material exposing its native scattering calculation.
    arrays : dict
        Output arrays with incident energies and absorption cross sections.
    sum_xs, selected_xs : numpy.ndarray
        Full and selected scattering totals in barns per original material atom.

    Raises
    ------
    RuntimeError
        If the component sum differs from native material scattering.
    ValueError
        If a reported scattering or absorption value is negative or nonfinite.
    """
    native_xs = mat.scatter.xsect(arrays['xs_energy'])
    if not np.allclose(sum_xs, native_xs, rtol=1e-9, atol=1e-10):
        raise RuntimeError("Captured components do not reconstruct native material scattering")
    arrays['xs_scattering'] = selected_xs
    for key in ('xs_scattering', 'xs_absorption'):
        if not np.isfinite(arrays[key]).all() or np.any(arrays[key] < 0):
            raise ValueError(f"Invalid {key}")


def _verify_native_library(expected_path):
    """Check the shared library loaded in the child process.

    Parameters
    ----------
    expected_path : str
        Expected absolute NCrystal shared library path.
    """
    # Detect loader overrides that would mix two C++ library installations.
    from NCrystal._locatelib import get_libpath_and_namespace
    loaded_path, namespace = get_libpath_and_namespace()
    if namespace or Path(loaded_path).resolve() != Path(expected_path).resolve():
        raise RuntimeError('Child loaded a different NCrystal library')
    mapped = {Path(line.split(maxsplit=5)[-1].strip()).resolve()
              for line in Path('/proc/self/maps').read_text().splitlines()
              if '/' in line and 'libNCrystal' in line}
    if mapped != {Path(expected_path).resolve()}:
        raise RuntimeError('Mixed or unexpected NCrystal shared libraries in child')


def _active_dynamics(info, settings, request):
    """Validate the requested species against the native dynamics.

    Parameters
    ----------
    info : NCrystal.Info
        Loaded material owning the native dynamics.
    settings : dict
        Parsed custom liquid settings by species.
    request : dict
        Capture configuration containing the active species and target.

    Returns
    -------
    tuple of dict and set
        Dynamics by label and the selected species labels.
    """
    dynamics = {di.atomData.displayLabel(): di for di in info.dyninfos}
    if len(dynamics) != len(info.dyninfos):
        raise ValueError("Repeated dynamics labels need an explicit component mapping")
    if not settings.keys() <= dynamics.keys():
        raise ValueError("Custom liquid label absent from dynamics")
    for label, di in dynamics.items():
        expected = NC.Info.DI_VDOS if label in settings else NC.Info.DI_FreeGas
        if not isinstance(di, expected):
            raise ValueError(f"Unsupported liquid dynamics for {label}: {type(di).__name__}")
    active = set(request['active_species'] or dynamics)
    if not active <= dynamics.keys() or request['target'] not in active:
        raise ValueError("Unknown active species or inactive analysis target")
    if request['target'] not in settings:
        raise ValueError("Analysis target must have a captured liquid VDOS kernel")
    return dynamics, active


def _match_capture_species(record, dynamics, candidates):
    """Identify a kernel by mass and bound cross section.

    Parameters
    ----------
    record : dict
        Parsed native kernel record.
    dynamics : dict
        Native dynamics by label.
    candidates : iterable of str
        Unmatched species with custom liquid kernels.

    Returns
    -------
    str
        Unique matching species label.
    """
    if record.get('layout') != 'sab[beta_index][alpha_index]':
        raise ValueError("Unknown capture layout")
    matches = [
        label for label in candidates
        if np.isclose(dynamics[label].atomData.averageMassAMU(),
                      record['element_mass_amu'], rtol=1e-10, atol=0)
        and np.isclose(dynamics[label].atomData.scatteringXS(),
                       record['bound_xs_barn'], rtol=1e-10, atol=0)
    ]
    if len(matches) != 1:
        raise ValueError("Ambiguous capture identity; refuse to infer label from order")
    return matches[0]


def _read_captured_kernels(files, arrays, settings, dynamics, temperature, energies, active):
    """Decode kernels and accumulate fraction-weighted cross sections.

    Parameters
    ----------
    files : sequence of pathlib.Path
        Captured JSON paths in component-index order.
    arrays : dict
        Output arrays, updated with the decoded kernels.
    settings, dynamics : dict
        Custom liquid settings and native dynamics by label.
    temperature : float
        Requested temperature in kelvin.
    energies : numpy.ndarray
        Incident energies in eV.
    active : set of str
        Labels included in the selected scattering cross section.

    Returns
    -------
    tuple
        Component metadata, all liquid scattering, and selected liquid scattering.
    """
    components = {}
    sum_xs, selected_xs = np.zeros_like(energies), np.zeros_like(energies)
    for index, filename in enumerate(files):
        record = json.loads(filename.read_text())
        unmatched = (label for label in settings if label not in components)
        label = _match_capture_species(record, dynamics, unmatched)
        component, weighted_xs = _decode_captured_component(
            filename, record, prefix=f"component_{index}", dynamics=dynamics[label],
            temperature=temperature, energy_shape=energies.shape, arrays=arrays,
        )
        # Fractions refer to the original material even when only a subset is selected.
        sum_xs += weighted_xs
        if label in active:
            selected_xs += weighted_xs
        components[label] = component
    return components, sum_xs, selected_xs


def _decode_captured_component(
    filename, record, *, prefix, dynamics, temperature, energy_shape, arrays,
):
    """Decode one kernel and read its fraction-weighted scattering values.

    Parameters
    ----------
    filename : pathlib.Path
        JSON kernel path used to locate the companion cross-section table.
    record : dict
        Parsed kernel record in beta-major layout.
    prefix : str
        Component key prefix in the binary handoff.
    dynamics : NCrystal.Info.DI_VDOS
        Native dynamics for the matched species.
    temperature : float
        Requested temperature in kelvin.
    energy_shape : tuple of int
        Shape of the incident energy grid.
    arrays : dict
        Output arrays, updated with this component's alpha, beta, and S values.

    Returns
    -------
    tuple of (dict, numpy.ndarray)
        Component metadata and weighted scattering cross sections in barns
        per original material atom.

    Raises
    ------
    ValueError
        If the captured temperature or cross-section table is inconsistent
        with the request.
    """
    if not np.isclose(record['temperature_K'], temperature, rtol=1e-12):
        raise ValueError("Captured temperature differs from request")
    alpha, beta, sab = read_sabtap_arrays(filename, record)
    arrays[f"{prefix}_alpha"] = alpha
    arrays[f"{prefix}_beta"] = beta
    arrays[f"{prefix}_sab"] = sab

    xs_path = filename.with_name(filename.name.replace('sab_', 'xs_')).with_suffix('.txt')
    component_xs = np.loadtxt(xs_path, ndmin=1)
    if component_xs.shape != energy_shape:
        raise ValueError("Incomplete component cross sections")
    weighted_xs = component_xs * dynamics.fraction
    component = {
        "prefix": prefix, "mass_amu": record['element_mass_amu'],
        "temperature": record['temperature_K'], "bound_xs": record['bound_xs_barn'],
        "fraction": dynamics.fraction, "suggested_emax": record['suggested_emax_eV'],
    }
    return component, weighted_xs


def _include_free_gas(root, text, labels, temperature, energies, active, sum_xs, selected_xs):
    """Add standard free-gas cross sections to the running totals.

    Parameters
    ----------
    root : pathlib.Path
        Capture directory for temporary isolated NCMAT files.
    text : str
        Original NCMAT content.
    labels : set of str
        Species handled by the standard free-gas backend.
    temperature : float
        Material temperature in kelvin.
    energies : numpy.ndarray
        Incident energies in eV.
    active : set of str
        Labels included in the selected scattering cross section.
    sum_xs, selected_xs : numpy.ndarray
        Material and selected scattering totals in barns, updated in place.
    """
    # The plugin treats all non-custom species as free gas. Evaluate these via
    # the standard backend, retaining their original material fractions.
    for label in labels:
        free_text = re.sub(r'(?ms)^@CUSTOM_LIQUIDS\s*\n.*?(?=^@|\Z)', '', text)
        free_text = isolate_ncmat_dyninfo(free_text, {label})
        free_path = root / f"freegas_{label}.ncmat"
        free_path.write_text(free_text)
        free_xs = NC.createScatter(f"{free_path};temp={temperature:.17g}K").xsect(energies)
        sum_xs += free_xs
        if label in active:
            selected_xs += free_xs


def _hash_dynamic_plugin_files(plugins):
    """Compute SHA-256 fingerprints for file-backed dynamic plugins.

    Parameters
    ----------
    plugins : iterable of tuple
        NCrystal plugin names, locations, and registration types.

    Returns
    -------
    dict of str to str
        SHA-256 digests keyed by plugin file path.
    """
    hashes = {}
    for _, location, kind in plugins:
        if not location or kind != "dynamic":
            continue
        path = Path(location)
        # NCrystal also registers data-directory plugins as dynamic.
        if path.is_file():
            hashes[location] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _write_capture_result(root, arrays, components, target, text, active, request):
    """Write the binary arrays and their provenance to the parent handoff.

    Parameters
    ----------
    root : pathlib.Path
        Capture directory.
    arrays : dict
        Evaluated cross sections, backend VDOS, and kernels.
    components : dict
        Kernel metadata keyed by species.
    target : NCrystal.Info.DI_VDOS
        Analysis target whose VDOS is used for thermodynamics.
    text : str
        Source NCMAT content for the material checksum.
    active : set of str
        Labels included in the selected scattering cross section.
    request : dict
        Capture request containing the target label and VDOS luxury level.
    """
    # A bounded, binary handoff; text captures are discarded with the sandbox.
    np.savez(root / 'result.npz', **arrays)
    vdos_analysis = target.analyseVDOS()
    plugins = NC.plugins.browsePlugins()
    metadata = {
        "components": components, "vdos_analysis": vdos_analysis,
        "provenance": {
            "ncrystal_version": NC.__version__, "plugins": plugins,
            "plugin_hashes": _hash_dynamic_plugin_files(plugins),
            "material_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "active_species": sorted(active), "target": request['target'],
            "vdoslux": request['vdoslux'], "capture": "SABScatter constructor",
            "xs_scope": "selected species per original material atom",
            "absorption_scope": "whole material per atom",
            "sab_scope": "target species per atom",
            "raw_alpha_convention": "neutron mass",
            "analyzer_alpha_convention": "target atom mass",
            "beta_convention": "(Eout-Ein)/(k_B*T)",
            "components": {label: {key: value for key, value in component.items()
                                    if key != 'prefix'}
                           for label, component in components.items()},
            "thermodynamics_scope": "target solid-like VDOS only",
            "kinematic_normalization": "per unit target free scattering cross section",
        },
    }
    (root / 'result.json').write_text(json.dumps(metadata))


if __name__ == '__main__':
    capture(sys.argv[1])
