"""Regression coverage for native capture with data-directory plugins."""

import hashlib
import json
from types import SimpleNamespace

from processing.backends.liquid_capture import child


def test_capture_records_data_plugins_without_hashing_directories(tmp_path, monkeypatch):
    """Preserve a WaterData-style plugin in provenance alongside a library hash.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary capture directory and plugin paths.
    monkeypatch : pytest.MonkeyPatch
        Supplies the NCrystal plugin registry for the metadata writer.
    """
    library = tmp_path / "libNCPlugin_Liquids.so"
    library.write_bytes(b"liquid plugin fixture")
    data_directory = tmp_path / "WaterData" / "data"
    data_directory.mkdir(parents=True)
    plugins = [
        ("Liquids", str(library), "dynamic"),
        ("WaterData", str(data_directory), "dynamic"),
    ]
    monkeypatch.setattr(child.NC.plugins, "browsePlugins", lambda: plugins)
    target = SimpleNamespace(analyseVDOS=lambda: {"teff": 295.0})

    child._write_capture_result(
        tmp_path, {}, {}, target, "fixture NCMAT", {"H"},
        {"target": "H", "vdoslux": 0},
    )

    metadata = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    provenance = metadata["provenance"]
    assert provenance["plugins"] == [list(entry) for entry in plugins]
    assert provenance["plugin_hashes"] == {
        str(library): hashlib.sha256(library.read_bytes()).hexdigest(),
    }
