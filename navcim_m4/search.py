from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

from .graph import validate_with_tvm
from .layers import create_layer_artifacts, write_network_csv
from .models import create_model, export_onnx
from .simulators import (
    HardwareConfig,
    build_booksim,
    build_neurosim,
    result_dict,
    run_booksim,
    run_neurosim,
)


def _score(results: list[dict]) -> None:
    fields = (
        ("neurosim", "latency_ns"),
        ("neurosim", "dynamic_energy_pj"),
        ("neurosim", "area_um2"),
        ("booksim", "latency_cycles"),
        ("booksim", "total_power_w"),
    )
    minima = {field: min(item[field[0]][field[1]] for item in results) for field in fields}
    for item in results:
        item["score"] = sum(
            item[group][name] / minima[(group, name)] if minima[(group, name)] > 0 else 0.0
            for group, name in fields
        )


def run_search(
    root: Path,
    output_dir: Path,
    sa_values: list[int],
    pe_values: list[int],
    tile_values: list[int],
    workers: int = 4,
    jobs: int = 8,
) -> dict:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = create_model()
    onnx_path = export_onnx(model, output_dir / "vgg11_cifar10.onnx")
    graph = validate_with_tvm(onnx_path)
    artifacts = create_layer_artifacts(model, output_dir / "layers")
    network_csv = write_network_csv(artifacts.records, output_dir / "vgg11_cifar10.csv")
    neurosim_binary = build_neurosim(root, jobs)
    booksim_binary = build_booksim(root, jobs)
    configs = [
        HardwareConfig(sa, sa, pe, tile)
        for sa, pe, tile in product(sa_values, pe_values, tile_values)
        if tile >= pe
    ]
    if not configs:
        raise ValueError("Search space did not produce a valid configuration")

    results: list[dict] = []
    failures: list[dict] = []

    def simulate(config: HardwareConfig) -> dict:
        candidate_dir = output_dir / "candidates" / config.key
        neurosim = run_neurosim(neurosim_binary, network_csv, artifacts, config, candidate_dir)
        booksim = run_booksim(booksim_binary, root, config, candidate_dir)
        return result_dict(config, neurosim, booksim)

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as executor:
        futures = {executor.submit(simulate, config): config for config in configs}
        for future in as_completed(futures):
            config = futures[future]
            try:
                results.append(future.result())
            except (RuntimeError, ValueError, OSError) as error:
                failures.append({"config": config.key, "error": str(error)})
    if not results:
        details = "; ".join(item["config"] for item in failures)
        raise RuntimeError(f"All simulator candidates failed: {details}")
    _score(results)
    results.sort(key=lambda item: item["score"])
    report = {
        "model": "VGG11_CIFAR10",
        "input_shape": [1, 3, 32, 32],
        "graph": {
            "conv_layers": graph.conv_layers,
            "linear_layers": graph.linear_layers,
            "relax_functions": graph.relax_functions,
        },
        "workers": max(1, min(workers, 4)),
        "best": results[0],
        "candidates": results,
        "failures": failures,
    }
    (output_dir / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report
