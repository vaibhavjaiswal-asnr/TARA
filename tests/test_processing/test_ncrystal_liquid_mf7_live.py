"""Run the notebook's liquid MF7 batch with the locally built adapter."""

import os
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.getenv("TARA_RUN_LIQUID_PLUGIN_SMOKE") != "1",
    reason="Requires the compiled Liquids plugin and a matching NCrystal C++ toolchain",
)
def test_live_liquid_batch_builds_adapter_and_exports_mf7(tmp_path):
    """Require the default adapter to produce a reference and sampled MF7 tape.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary batch destination for the validated ENDF outputs.
    """
    from core.config import TaraConfig
    from core.data.input_pdos import PhononSpectra
    from core.sampler.concrete_samplers import TaraSampler
    from input_output.ncrystal.core import NCrystalFile
    from processing.orchestrators.ncrystal_batch import NcrystalBatchOrchestrator

    source = Path(__file__).parents[1] / "fixtures" / "liquid_water_295K_illustrative.ncmat"
    template = NCrystalFile.from_file(str(source))
    reference = PhononSpectra(template.pdos_for_atom("H"))
    config = TaraConfig(
        max_energy_shift_mev=0.2,
        max_amplitude_shift_pct=1.0,
        n_samples=1,
        random_seed=17,
    )
    sampler = TaraSampler(config, reference)
    ensemble = sampler.generate_ensemble()

    # Let the batch compile this checkout's adapter, as the notebook does by default.
    batch = NcrystalBatchOrchestrator(n_jobs=1)
    result = batch.generate_ensemble_sab_tapes(
        ensemble,
        output_dir=str(tmp_path),
        base_ncmat_file=str(source),
        temperature=295.0,
        atom_label="H",
    )

    assert set(result) == {"sample_000", "sample_001"}
    assert all(Path(path).stat().st_size > 0 for path in result.values())
