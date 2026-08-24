from __future__ import annotations

import csv
import re
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from ..simulators import FLOAT, parse_neurosim_noc_config, parse_neurosim_output
from .contracts import file_reference, read_manifest, verify_file, write_manifest


@dataclass(frozen=True)
class LayerMetrics:
    layer: int
    latency_ns: float
    dynamic_energy_pj: float
    leakage_power_uw: float
    buffer_latency_ns: float
    buffer_energy_pj: float
    interconnect_latency_ns: float
    interconnect_energy_pj: float
    cim_array_area_um2: float
    tile_area_um2: float
    htree_area_um2: float
    htree_latency_ns: float
    htree_energy_pj: float
    adc_latency_ns: float
    accumulation_latency_ns: float
    other_latency_ns: float
    adc_energy_pj: float
    accumulation_energy_pj: float
    other_energy_pj: float


def _metric(section: str, pattern: str, label: str) -> float:
    match = re.search(pattern, section)
    if not match:
        raise ValueError(f"Missing layer metric: {label}")
    return float(match.group(1))


def parse_layer_metrics(output: str) -> tuple[LayerMetrics, ...]:
    headers = list(re.finditer(r"Estimation of Layer\s+(\d+)", output))
    layers = []
    for index, header in enumerate(headers):
        layer = int(header.group(1))
        section = output[header.end():headers[index + 1].start() if index + 1 < len(headers) else len(output)]
        prefix = rf"layer{layer}'s"
        layers.append(LayerMetrics(
            layer=layer,
            latency_ns=_metric(section, rf"{prefix} readLatency is:\s*{FLOAT}ns", "latency"),
            dynamic_energy_pj=_metric(section, rf"{prefix} readDynamicEnergy is:\s*{FLOAT}pJ", "dynamic energy"),
            leakage_power_uw=_metric(section, rf"{prefix} leakagePower is:\s*{FLOAT}uW", "leakage power"),
            buffer_latency_ns=_metric(section, rf"{prefix} buffer latency is:\s*{FLOAT}ns", "buffer latency"),
            buffer_energy_pj=_metric(section, rf"{prefix} buffer readDynamicEnergy is:\s*{FLOAT}pJ", "buffer energy"),
            interconnect_latency_ns=_metric(section, rf"{prefix} ic latency is:\s*{FLOAT}ns", "interconnect latency"),
            interconnect_energy_pj=_metric(section, rf"{prefix} ic readDynamicEnergy is:\s*{FLOAT}pJ", "interconnect energy"),
            cim_array_area_um2=_metric(section, rf"{prefix} CIM Array Area is:\s*{FLOAT}um\^2", "CIM array area"),
            tile_area_um2=_metric(section, rf"{prefix} Tile Area is:\s*{FLOAT}um\^2", "tile area"),
            htree_area_um2=_metric(section, rf"{prefix} H-tree Area is:\s*{FLOAT}um\^2", "H-tree area"),
            htree_latency_ns=_metric(section, rf"{prefix} H-tree Latency is\s*:\s*{FLOAT}ns", "H-tree latency"),
            htree_energy_pj=_metric(section, rf"{prefix} H-tree Energy is\s*:\s*{FLOAT}pJ", "H-tree energy"),
            adc_latency_ns=_metric(section, rf"ADC .*? readLatency is\s*:\s*{FLOAT}ns", "ADC latency"),
            accumulation_latency_ns=_metric(section, rf"Accumulation Circuits .*? readLatency is\s*:\s*{FLOAT}ns", "accumulation latency"),
            other_latency_ns=_metric(section, rf"Other Peripheries .*? readLatency is\s*:\s*{FLOAT}ns", "other latency"),
            adc_energy_pj=_metric(section, rf"ADC .*? readDynamicEnergy is\s*:\s*{FLOAT}pJ", "ADC energy"),
            accumulation_energy_pj=_metric(section, rf"Accumulation Circuits .*? readDynamicEnergy is\s*:\s*{FLOAT}pJ", "accumulation energy"),
            other_energy_pj=_metric(section, rf"Other Peripheries .*? readDynamicEnergy is\s*:\s*{FLOAT}pJ", "other energy"),
        ))
    if not layers:
        raise ValueError("NeuroSim output contains no layer estimates")
    return tuple(layers)


def analyze_candidate_stage(candidate_manifest: Path, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    candidate = read_manifest(candidate_manifest, "candidate")
    log = verify_file(candidate["outputs"]["log"])
    output = log.read_text(encoding="utf-8")
    layers = parse_layer_metrics(output)
    chip = parse_neurosim_output(output)
    noc = parse_neurosim_noc_config(output)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_csv = output_dir / "layers.csv"
    with layer_csv.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(LayerMetrics)])
        writer.writeheader()
        writer.writerows(asdict(layer) for layer in layers)
    largest_latency = max(layers, key=lambda layer: layer.latency_ns)
    largest_energy = max(layers, key=lambda layer: layer.dynamic_energy_pj)
    return write_manifest(
        output_dir / "analysis.json",
        "analysis",
        {
            "inputs": {"candidate_manifest": file_reference(candidate_manifest)},
            "config": candidate["config"],
            "chip": asdict(chip),
            "noc": asdict(noc) | {"clock_hz": noc.clock_hz, "link_latency_cycles": noc.link_latency_cycles},
            "layers": [asdict(layer) for layer in layers],
            "outputs": {"layer_csv": file_reference(layer_csv)},
            "metrics": {
                "layer_count": len(layers),
                "largest_latency_layer": largest_latency.layer,
                "largest_latency_ns": largest_latency.latency_ns,
                "largest_energy_layer": largest_energy.layer,
                "largest_energy_pj": largest_energy.dynamic_energy_pj,
            },
        },
        started,
    )
