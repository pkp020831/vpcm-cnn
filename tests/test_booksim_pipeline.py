import json

import pytest

from navcim_m4.booksim.features import AREA_FEATURES, DYNAMIC_FEATURES, TARGET_FEATURES
from navcim_m4.booksim.pipeline import create_samples_stage
from navcim_m4.booksim.ppa import combine_ppa, combine_predicted_layerwise_ppa
from navcim_m4.booksim.ranking import pareto_front, topsis_rank
from navcim_m4.simulators import BookSimResult, NeuroSimNoCConfig, NeuroSimResult


def test_static_and_dynamic_predictors_use_distinct_feature_contracts():
    assert TARGET_FEATURES["latency_cycles"] == DYNAMIC_FEATURES
    assert TARGET_FEATURES["dynamic_power_w"] == DYNAMIC_FEATURES
    assert TARGET_FEATURES["area_m2"] == AREA_FEATURES
    assert TARGET_FEATURES["leakage_power_w"] == AREA_FEATURES


def test_sample_generation_is_an_independent_manifest_stage(tmp_path):
    report = create_samples_stage(tmp_path, count=3, seed=117)

    assert report["stage"] == "samples"
    assert report["metrics"]["sample_count"] == 3
    assert (tmp_path / "samples.csv").read_text(encoding="ascii").count("\n") == 4
    assert json.loads((tmp_path / "samples.json").read_text())["schema"] == "navcim.booksim-pipeline/v1"


def test_combined_ppa_uses_explicit_unit_conversions():
    neurosim = NeuroSimResult(10.0, 20.0, 1000.0, 100.0)
    booksim = BookSimResult(5.0, 0.6, 0.1, 2.0)
    noc = NeuroSimNoCConfig(1e-3, 1e-4, 128, 2.0, 1e-7)

    result = combine_ppa(neurosim, booksim, noc)

    assert noc.clock_hz == 5e8
    assert result.booksim_latency_ns == 10.0
    assert result.booksim_dynamic_energy_pj == 6000.0
    assert result.latency_ns == 20.0
    assert result.dynamic_energy_pj == 6020.0
    assert result.dynamic_power_mw == 301.0
    assert result.leakage_power_mw == 101.0
    assert result.total_power_mw == 402.0
    assert result.area_um2 == pytest.approx(2_000_100.0)


def test_predictor_energy_sums_each_layer_before_composition():
    neurosim = NeuroSimResult(10.0, 20.0, 1000.0, 100.0)
    booksim = BookSimResult(10.0, 0.3, 0.1, 2.0)
    noc = NeuroSimNoCConfig(1e-3, 1e-4, 128, 2.0, 1e-7)

    class Layer:
        def __init__(self, layer, latency_ns, dynamic_energy_pj):
            self.layer = layer
            self.latency_ns = latency_ns
            self.dynamic_energy_pj = dynamic_energy_pj

    result = combine_predicted_layerwise_ppa(
        neurosim,
        booksim,
        noc,
        None,
        [Layer(2, 3.0, 4.0), Layer(3, 5.0, 6.0)],
        [
            {"layer": 2, "prediction": {"latency_cycles": 2.0, "dynamic_power_w": 0.1}},
            {"layer": 3, "prediction": {"latency_cycles": 8.0, "dynamic_power_w": 0.2}},
        ],
    )

    # Residual NeuroSim work is 2 ns/10 pJ. BookSim energy is 0.1*4 + 0.2*16 ns*mW.
    assert result.latency_ns == pytest.approx(30.0)
    assert result.dynamic_energy_pj == pytest.approx(3620.0)


def test_pareto_then_topsis_keeps_only_non_dominated_candidates():
    candidates = [
        {"id": "balanced", "latency": 2.0, "power": 2.0, "area": 2.0},
        {"id": "fast", "latency": 1.0, "power": 3.0, "area": 2.0},
        {"id": "dominated", "latency": 3.0, "power": 3.0, "area": 3.0},
    ]
    objectives = {"latency": "min", "power": "min", "area": "min"}

    front = pareto_front(candidates, objectives)
    ranked = topsis_rank(front, objectives, {name: 1.0 for name in objectives})

    assert {candidate["id"] for candidate in front} == {"balanced", "fast"}
    assert [candidate["topsis_rank"] for candidate in ranked] == [1, 2]
