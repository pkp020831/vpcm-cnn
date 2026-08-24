from __future__ import annotations

import json
from dataclasses import asdict
from math import ceil
from pathlib import Path

from ..booksim_mapping import output_activation_bits, read_floorplan
from ..layers import LayerArtifacts
from ..simulators import (
    BOOKSIM_FLIT_BITS,
    BookSimResult,
    parse_neurosim_noc_config,
    parse_neurosim_output,
)
from .features import BookSimFeatures
from .models import PredictorBundle


class OutOfDistributionError(ValueError):
    pass


def candidate_features(
    neurosim_log: Path,
    floorplan_path: Path,
    artifacts: LayerArtifacts,
) -> tuple[BookSimFeatures, ...]:
    output = neurosim_log.read_text(encoding="utf-8")
    noc = parse_neurosim_noc_config(output)
    neurosim = parse_neurosim_output(output)
    floorplan = read_floorplan(floorplan_path)
    if len(floorplan) != len(artifacts.records):
        raise ValueError("NeuroSim floorplan and workload layers differ in length")
    images_per_second = 1e9 / neurosim.latency_ns
    features = []
    for index in range(1, len(floorplan)):
        source, destination = floorplan[index - 1], floorplan[index]
        activation_bits = output_activation_bits(artifacts.records[index - 1])
        bits_per_source = ceil(activation_bits / source.tile_count)
        features.append(BookSimFeatures(
            network_size=source.mesh_rows,
            latency_per_flit=noc.link_latency_cycles,
            tile_width_m=noc.tile_width_m,
            inject_rate=min(1.0, bits_per_source * images_per_second / (BOOKSIM_FLIT_BITS * noc.clock_hz)),
            data_size_flits=ceil(activation_bits / BOOKSIM_FLIT_BITS),
            num_start_node=source.tile_count,
            num_dest_node=destination.tile_count,
        ))
    if not features:
        raise ValueError("BookSim prediction requires at least two workload layers")
    return tuple(features)


def predict_candidate(
    bundle: PredictorBundle,
    neurosim_log: Path,
    floorplan_path: Path,
    artifacts: LayerArtifacts,
    output_dir: Path,
    reject_out_of_distribution: bool = True,
) -> BookSimResult:
    layer_features = candidate_features(neurosim_log, floorplan_path, artifacts)
    layers = []
    violations = []
    for index, features in enumerate(layer_features, start=2):
        layer_violations = bundle.out_of_distribution(features)
        violations.extend(f"layer{index}:{violation}" for violation in layer_violations)
        layers.append({
            "layer": index,
            "features": asdict(features),
            "prediction": bundle.predict(features),
        })
    if violations and reject_out_of_distribution:
        raise OutOfDistributionError("; ".join(violations))
    dynamic_power = max(layer["prediction"]["dynamic_power_w"] for layer in layers)
    leakage_power = max(layer["prediction"]["leakage_power_w"] for layer in layers)
    result = BookSimResult(
        latency_cycles=sum(layer["prediction"]["latency_cycles"] for layer in layers),
        total_power_w=dynamic_power,
        leakage_power_w=leakage_power,
        area_m2=max(layer["prediction"]["area_m2"] for layer in layers),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "booksim_prediction.json").write_text(json.dumps({
        "schema": "navcim.booksim-prediction/v1",
        "bundle_manifest": str(bundle.path),
        "layers": layers,
        "out_of_distribution": sorted(set(violations)),
        "aggregate": asdict(result),
    }, indent=2, sort_keys=True), encoding="utf-8")
    return result
