from navcim_m4.booksim_predictor import FEATURES, TARGETS, _sample


def test_booksim_training_samples_follow_fixed_baseline_feature_space():
    sample = _sample(117)
    assert tuple(sample) == FEATURES
    assert 2 <= sample["network_size"] <= 17
    assert sample["num_start_node"] >= 1
    assert sample["num_dest_node"] >= 1
    assert sample["num_start_node"] == 1 or sample["num_dest_node"] == 1
    assert 1 <= sample["data_size_flits"] <= 15_000_000
    assert TARGETS == ("latency_cycles", "dynamic_power_w", "area_m2", "leakage_power_w")
