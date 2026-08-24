from pathlib import Path

import pytest

import navcim_m4.search as search
from navcim_m4.simulators import prescribed_hardware_sweep_groups


def test_prescribed_hardware_sweep_groups_are_independent_full_factorials():
    groups = prescribed_hardware_sweep_groups()

    assert {name: len(configs) for name, configs in groups.items()} == {
        "adc_cell": 9,
        "sa_pe_tile": 27,
        "num_col_muxed": 3,
    }
    assert {(config.adc_bits, config.cell_bits) for config in groups["adc_cell"]} == {
        (adc, cell) for adc in (4, 6, 8) for cell in (1, 2, 4)
    }
    assert all((config.adc_bits, config.cell_bits, config.num_col_muxed) == (8, 2, 8) for config in groups["sa_pe_tile"])
    assert all((config.adc_bits, config.cell_bits, config.sa_row, config.sa_col, config.pe, config.tile) == (8, 2, 128, 128, 4, 8) for config in groups["num_col_muxed"])


def test_booksim_mode_controls_predictor_loading(monkeypatch):
    loaded = []

    class FakePredictor:
        def __init__(self, path):
            loaded.append(path)

    monkeypatch.setattr(search, "PredictorBundle", FakePredictor)

    assert search._predictor_for_mode("simulator", Path("unused.json"), None) is None
    assert loaded == []
    assert isinstance(search._predictor_for_mode("predictor", Path("bundle.json"), None), FakePredictor)
    assert loaded == [Path("bundle.json")]
    with pytest.raises(ValueError, match="Meta bundle requires predictor"):
        search._predictor_for_mode("simulator", None, Path("meta.json"))


def test_candidate_cache_requires_exact_input_contract(tmp_path):
    path = tmp_path / "result.json"
    inputs = {"config": {"tile": 8}, "network_csv_sha256": "a"}
    path.write_text(__import__("json").dumps({
        "schema": "navcim.search-candidate/v1",
        "inputs": inputs,
        "result": {"config": {"tile": 8}},
    }))

    assert search._load_cached_candidate(path, inputs) == {"config": {"tile": 8}}
    assert search._load_cached_candidate(path, {**inputs, "network_csv_sha256": "b"}) is None
