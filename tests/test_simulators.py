from navcim_m4.simulators import BOOKSIM_FLIT_BITS, parse_booksim_output, parse_neurosim_noc_config, parse_neurosim_output


def test_parse_neurosim_output():
    output = """
ChipArea : 1234.5um^2
Chip layer-by-layer readLatency (per image) is: 42.5ns
Chip total readDynamicEnergy is: 91.25pJ
Chip total leakage Power is: 7.75uW
"""
    result = parse_neurosim_output(output)
    assert result.latency_ns == 42.5
    assert result.dynamic_energy_pj == 91.25
    assert result.leakage_power_uw == 7.75
    assert result.area_um2 == 1234.5


def test_parse_booksim_output():
    output = """
Time taken is 120 cycles
- Total Power: 0.5
- Total Area: 0.25
- Total leak Power: 0.01
"""
    result = parse_booksim_output(output)
    assert result.latency_cycles == 120
    assert result.total_power_w == 0.5
    assert result.area_m2 == 0.25
    assert result.leakage_power_w == 0.01


def test_parse_neurosim_noc_config():
    output = """
Tilewidth : 0.0002m
minDist0.00001m
busWidth256
NoC unitLatencyRep: 2e-10 unitLatencyWire: 1e-10
Chip clock period is: 2ns
"""
    result = parse_neurosim_noc_config(output)
    assert result.bus_width_bits == 256
    assert result.clock_hz == 5e8
    assert result.link_latency_cycles == 1


def test_booksim_flit_width_matches_legacy_navcim_baseline():
    assert BOOKSIM_FLIT_BITS == 128
