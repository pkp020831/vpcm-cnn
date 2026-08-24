import json

import pytest

from navcim_m4.pipeline import create_search_space_stage, rank_candidates_stage, score_candidates


def test_search_space_stage_has_explicit_independent_configs(tmp_path):
    output = tmp_path / "space.json"

    report = create_search_space_stage(output, [64, 128], [4], [2, 8], [5], [2], [8])

    assert report["metrics"]["candidate_count"] == 2
    assert [item["key"] for item in report["configs"]] == [
        "sa64x64_pe4_tile8_adc5_cell2_mux8",
        "sa128x128_pe4_tile8_adc5_cell2_mux8",
    ]
    assert json.loads(output.read_text())["stage"] == "search-space"


def test_score_candidates_can_be_evaluated_without_simulators():
    candidates = [
        {
            "neurosim": {"latency_ns": 2.0, "dynamic_energy_pj": 4.0, "area_um2": 8.0},
            "booksim": {"latency_cycles": 16.0, "total_power_w": 32.0},
        },
        {
            "neurosim": {"latency_ns": 4.0, "dynamic_energy_pj": 8.0, "area_um2": 16.0},
            "booksim": {"latency_cycles": 32.0, "total_power_w": 64.0},
        },
    ]

    score_candidates(candidates)

    assert candidates[0]["score"] == 5.0
    assert candidates[1]["score"] == 10.0


def test_ranking_requires_at_least_one_completed_candidate(tmp_path):
    with pytest.raises(ValueError, match="At least one candidate"):
        rank_candidates_stage([], tmp_path / "ranking.json")
