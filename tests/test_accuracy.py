import sys

import pytest
import torch

from navcim_m4.accuracy import CrossSimConfig, _crosssim_model, prescribed_sweep_conditions
from navcim_m4.models import create_model, sample_input


def test_crosssim_config_rejects_unimplemented_drift():
    with pytest.raises(ValueError, match="drift"):
        CrossSimConfig(drift_time=1).validate()


def test_crosssim_ideal_cpu_matches_torch():
    model = create_model()
    analog = _crosssim_model(model, CrossSimConfig(adc_bits=0, dac_bits=0, cell_bits=31), 10.0, [10.0] * 9)
    with torch.inference_mode():
        difference = (model(sample_input()) - analog(sample_input())).abs().max().item()
    assert difference < 1e-5
    assert "cupy" not in sys.modules


def test_prescribed_sweep_has_six_baselines_and_42_conditions():
    conditions = prescribed_sweep_conditions()
    assert len(conditions) == 42
    assert sum(runs == 1 for _, _, runs in conditions) == 6
    assert {config.adc_bits for _, config, _ in conditions} == {4, 5, 8}
    assert {(config.cell_bits, config.weight_slices) for _, config, _ in conditions} == {(7, 1), (1, 7)}
