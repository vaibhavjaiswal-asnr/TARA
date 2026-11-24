"""Verify both SABData payload layouts at the child-process handoff."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from processing.backends.liquid_capture.child import _read_captured_kernels
from processing.backends.liquid_capture.payload import read_sabtap_arrays
from processing.backends.ncrystal_liquid_endf import load_sabtap_json


@pytest.fixture(params=["inline", "binary"])
def captured_component(tmp_path, request):
    """Write one native-style kernel record and its companion cross-section file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for the capture files.
    request : pytest.FixtureRequest
        Selects storage as inline JSON or native float64 binary.

    Returns
    -------
    tuple
        Kernel record path and the expected alpha, beta, and S arrays.
    """
    alpha = np.array([0.0, 0.5])
    beta = np.array([-1.0, 0.0, 1.0])
    sab = np.arange(1.0, 7.0).reshape(beta.size, alpha.size)
    record = {
        "layout": "sab[beta_index][alpha_index]",
        "temperature_K": 295.0,
        "element_mass_amu": 1.0,
        "bound_xs_barn": 80.0,
        "suggested_emax_eV": 0.1,
    }
    if request.param == "binary":
        binary = tmp_path / "sab_0000.bin"
        np.concatenate((alpha, beta, sab.ravel(order="C"))).tofile(binary)
        record.update(binary_file=binary.name, n_alpha=alpha.size, n_beta=beta.size)
    else:
        record.update(alpha=alpha.tolist(), beta=beta.tolist(), sab=sab.tolist())

    filename = tmp_path / "sab_0000.json"
    filename.write_text(json.dumps(record), encoding="utf-8")
    np.savetxt(tmp_path / "xs_0000.txt", [10.0, 14.0])
    return filename, alpha, beta, sab


def test_child_reads_capture_payload(captured_component):
    """Pass both storage formats through the child reader used by liquid MF7.

    Parameters
    ----------
    captured_component : tuple
        Capture paths and arrays for one native-style component.
    """
    filename, alpha, beta, sab = captured_component
    atom_data = SimpleNamespace(averageMassAMU=lambda: 1.0, scatteringXS=lambda: 80.0)
    dynamics = {"H": SimpleNamespace(atomData=atom_data, fraction=0.5)}
    arrays = {}

    components, full_xs, selected_xs = _read_captured_kernels(
        [filename], arrays, {"H": object()}, dynamics, 295.0,
        np.array([0.01, 0.0253]), {"H"},
    )

    prefix = components["H"]["prefix"]
    np.testing.assert_array_equal(arrays[f"{prefix}_alpha"], alpha)
    np.testing.assert_array_equal(arrays[f"{prefix}_beta"], beta)
    np.testing.assert_array_equal(arrays[f"{prefix}_sab"], sab)
    np.testing.assert_array_equal(full_xs, [5.0, 7.0])
    np.testing.assert_array_equal(selected_xs, full_xs)


def test_export_reader_preserves_beta_major_order(captured_component):
    """Keep the standalone exporter reader's flat beta-major representation.

    Parameters
    ----------
    captured_component : tuple
        Capture paths and arrays for one native-style component.
    """
    filename, alpha, beta, sab = captured_component

    capture = load_sabtap_json(filename)

    np.testing.assert_array_equal(capture["alpha"], alpha)
    np.testing.assert_array_equal(capture["beta"], beta)
    np.testing.assert_array_equal(capture["sab"], sab.ravel(order="C"))


def test_reject_truncated_binary_payload(tmp_path):
    """Reject a companion binary file whose size disagrees with its grids.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for the captured payload.
    """
    binary = tmp_path / "sab_0000.bin"
    np.zeros(10, dtype=np.float64).tofile(binary)
    record = {"binary_file": binary.name, "n_alpha": 2, "n_beta": 3}

    with pytest.raises(ValueError, match="binary payload size mismatch"):
        read_sabtap_arrays(tmp_path / "sab_0000.json", record)
