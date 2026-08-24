import json

from navcim_m4.neurosim.analysis import parse_layer_metrics
from navcim_m4.neurosim.pipeline import register_search_space_stage


LAYER_OUTPUT = """
-------------------- Estimation of Layer 1 ----------------------
layer1's readLatency is: 100ns
layer1's readDynamicEnergy is: 200pJ
layer1's leakagePower is: 3uW
layer1's buffer latency is: 10ns
layer1's buffer readDynamicEnergy is: 20pJ
layer1's ic latency is: 30ns
layer1's ic readDynamicEnergy is: 40pJ
layer1's CIM Array Area is: 50um^2
layer1's Tile Area is: 60um^2
layer1's H-tree Area is: 70um^2
layer1's H-tree Latency is : 8ns
layer1's H-tree Energy is : 9pJ
----------- ADC (or S/As and precharger for SRAM) readLatency is : 11ns
----------- Accumulation Circuits (subarray level) readLatency is : 12ns
----------- Other Peripheries (buffers and IC) readLatency is : 13ns
----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : 14pJ
----------- Accumulation Circuits (subarray level) readDynamicEnergy is : 15pJ
----------- Other Peripheries (buffers and IC) readDynamicEnergy is : 16pJ
"""


def test_layerwise_parser_is_independent_from_simulator_execution():
    layers = parse_layer_metrics(LAYER_OUTPUT)

    assert len(layers) == 1
    assert layers[0].latency_ns == 100
    assert layers[0].dynamic_energy_pj == 200
    assert layers[0].adc_latency_ns == 11
    assert layers[0].other_energy_pj == 16


def test_search_space_registration_freezes_candidate_contract(tmp_path):
    source = tmp_path / "main-space.json"
    source.write_text(json.dumps({
        "schema": "navcim.pipeline/v1",
        "stage": "search-space",
        "configs": [{
            "key": "sa128x128_pe4_tile8_adc5_cell2_mux8",
            "values": {
                "sa_row": 128,
                "sa_col": 128,
                "pe": 4,
                "tile": 8,
                "adc_bits": 5,
                "cell_bits": 2,
                "num_col_muxed": 8,
            },
        }],
    }), encoding="utf-8")

    report = register_search_space_stage(source, tmp_path / "registered.json")

    assert report["stage"] == "search-space"
    assert report["metrics"]["candidate_count"] == 1
    assert report["configs"][0]["key"].endswith("mux8")
