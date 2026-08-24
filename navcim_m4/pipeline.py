from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any

from .layers import LayerArtifacts, LayerRecord, create_layer_artifacts, write_network_csv
from .models import ModelInfo, create_model, export_onnx, load_checkpoint
from .simulators import (
    BookSimResult,
    HardwareConfig,
    NeuroSimResult,
    build_booksim,
    build_neurosim,
    result_dict,
    run_booksim,
    run_neurosim,
)


SCHEMA = "navcim.pipeline/v1"


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_reference(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": _sha256(path)}


def _verify_file(reference: dict[str, str]) -> Path:
    path = Path(reference["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sha256(path) != reference["sha256"]:
        raise ValueError(f"Pipeline input changed after registration: {path}")
    return path


def _write_manifest(path: Path, stage: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": SCHEMA, "stage": stage, **payload}
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _read_manifest(path: Path, stage: str) -> dict[str, Any]:
    path = path.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("stage") != stage:
        raise ValueError(f"Expected {SCHEMA} {stage!r} manifest: {path}")
    return manifest


def export_model_stage(
    output_dir: Path,
    checkpoint: Path | None = None,
    seed: int = 117,
) -> dict[str, Any]:
    """Freeze model identity and export ONNX for downstream stages."""
    started = time.perf_counter()
    output_dir = output_dir.resolve()
    if checkpoint is None:
        model = create_model(seed)
        source = {"kind": "seed", "seed": seed}
    else:
        checkpoint = checkpoint.resolve()
        model = load_checkpoint(checkpoint)[0]
        source = {
            "kind": "checkpoint",
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
        }
    onnx_path = export_onnx(model, output_dir / "model.onnx").resolve()
    info = ModelInfo()
    return _write_manifest(
        output_dir / "model.json",
        "model-export",
        {
            "model": {
                "name": info.name,
                "input_shape": list(info.input_shape),
                "source": source,
            },
            "outputs": {"onnx": str(onnx_path), "onnx_sha256": _sha256(onnx_path)},
            "performance": {"wall_seconds": time.perf_counter() - started},
        },
    )


def _model_from_manifest(path: Path):
    manifest = _read_manifest(path, "model-export")
    source = manifest["model"]["source"]
    if source["kind"] == "seed":
        return create_model(int(source["seed"])), manifest
    checkpoint = Path(source["path"])
    if _sha256(checkpoint) != source["sha256"]:
        raise ValueError(f"Checkpoint changed after model export: {checkpoint}")
    return load_checkpoint(checkpoint)[0], manifest


def validate_graph_stage(model_manifest: Path, output: Path) -> dict[str, Any]:
    """Validate an exported graph without regenerating the model."""
    from .graph import validate_with_tvm

    started = time.perf_counter()
    model_data = _read_manifest(model_manifest, "model-export")
    onnx_path = Path(model_data["outputs"]["onnx"])
    if _sha256(onnx_path) != model_data["outputs"]["onnx_sha256"]:
        raise ValueError(f"ONNX model changed after export: {onnx_path}")
    report = validate_with_tvm(onnx_path)
    return _write_manifest(
        output,
        "graph-validation",
        {
            "inputs": {"model_manifest": _file_reference(model_manifest)},
            "graph": asdict(report) | {"weight_layers": report.weight_layers},
            "performance": {"wall_seconds": time.perf_counter() - started},
        },
    )


def create_workload_stage(model_manifest: Path, output_dir: Path) -> dict[str, Any]:
    """Materialize all NeuroSim layer inputs as a reusable workload."""
    started = time.perf_counter()
    model, _ = _model_from_manifest(model_manifest)
    output_dir = output_dir.resolve()
    artifacts = create_layer_artifacts(model, output_dir / "layers")
    network_csv = write_network_csv(artifacts.records, output_dir / "network.csv")
    layers = [
        {
            "record": asdict(record),
            "weight_file": str(weight_file),
            "input_file": str(input_file),
        }
        for record, weight_file, input_file in zip(
            artifacts.records, artifacts.weight_files, artifacts.input_files, strict=True
        )
    ]
    return _write_manifest(
        output_dir / "workload.json",
        "workload-generation",
        {
            "inputs": {"model_manifest": _file_reference(model_manifest)},
            "outputs": {
                "network_csv": _file_reference(network_csv),
                "layers": [{
                    "record": layer["record"],
                    "weight_file": _file_reference(Path(layer["weight_file"])),
                    "input_file": _file_reference(Path(layer["input_file"])),
                } for layer in layers],
            },
            "metrics": {"layer_count": len(layers)},
            "performance": {"wall_seconds": time.perf_counter() - started},
        },
    )


def _artifacts_from_manifest(path: Path) -> tuple[LayerArtifacts, Path]:
    manifest = _read_manifest(path, "workload-generation")
    layers = manifest["outputs"]["layers"]
    artifacts = LayerArtifacts(
        tuple(LayerRecord(**layer["record"]) for layer in layers),
        tuple(_verify_file(layer["weight_file"]) for layer in layers),
        tuple(_verify_file(layer["input_file"]) for layer in layers),
    )
    network_csv = _verify_file(manifest["outputs"]["network_csv"])
    return artifacts, network_csv


def build_simulators_stage(root: Path, output: Path, jobs: int = 8) -> dict[str, Any]:
    """Build simulator binaries once and publish their explicit paths."""
    started = time.perf_counter()
    root = root.resolve()
    neurosim_binary = build_neurosim(root, jobs).resolve()
    booksim_binary = build_booksim(root, jobs).resolve()
    return _write_manifest(
        output,
        "simulator-build",
        {
            "root": str(root),
            "outputs": {
                "neurosim_binary": _file_reference(neurosim_binary),
                "booksim_binary": _file_reference(booksim_binary),
            },
            "performance": {"wall_seconds": time.perf_counter() - started},
        },
    )


def create_search_space_stage(
    output: Path,
    sa_values: list[int],
    pe_values: list[int],
    tile_values: list[int],
    adc_values: list[int],
    cell_values: list[int],
    mux_values: list[int],
) -> dict[str, Any]:
    """Expand and persist the hardware configurations independently."""
    started = time.perf_counter()
    configs = list(
        dict.fromkeys(
            HardwareConfig(sa, sa, pe, tile, adc, cell, mux)
            for sa, pe, tile, adc, cell, mux in product(
                sa_values, pe_values, tile_values, adc_values, cell_values, mux_values
            )
            if tile >= pe
        )
    )
    if not configs:
        raise ValueError("Search space did not produce a valid configuration")
    return _write_manifest(
        output,
        "search-space",
        {
            "configs": [{"key": config.key, "values": asdict(config)} for config in configs],
            "metrics": {"candidate_count": len(configs)},
            "performance": {"wall_seconds": time.perf_counter() - started},
        },
    )


def _config_from_manifest(path: Path, key: str) -> HardwareConfig:
    manifest = _read_manifest(path, "search-space")
    matches = [entry for entry in manifest["configs"] if entry["key"] == key]
    if len(matches) != 1:
        raise ValueError(f"Configuration key is not unique in search space: {key}")
    return HardwareConfig(**matches[0]["values"])


def run_neurosim_stage(
    workload_manifest: Path,
    build_manifest: Path,
    search_space_manifest: Path,
    config_key: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate one hardware configuration with NeuroSim."""
    started = time.perf_counter()
    artifacts, network_csv = _artifacts_from_manifest(workload_manifest)
    build_data = _read_manifest(build_manifest, "simulator-build")
    config = _config_from_manifest(search_space_manifest, config_key)
    output_dir = output_dir.resolve()
    result = run_neurosim(
        _verify_file(build_data["outputs"]["neurosim_binary"]),
        network_csv,
        artifacts,
        config,
        output_dir,
    )
    return _write_manifest(
        output_dir / "neurosim.json",
        "neurosim",
        {
            "inputs": {
                "workload_manifest": _file_reference(workload_manifest),
                "build_manifest": _file_reference(build_manifest),
                "search_space_manifest": _file_reference(search_space_manifest),
            },
            "config": {"key": config.key, "values": asdict(config)},
            "outputs": {
                "log": _file_reference(output_dir / "neurosim.log"),
                "floorplan": _file_reference(output_dir / "neurosim_floorplan.csv"),
            },
            "metrics": asdict(result),
            "performance": {"wall_seconds": time.perf_counter() - started},
        },
    )


def run_booksim_stage(
    root: Path,
    workload_manifest: Path,
    build_manifest: Path,
    neurosim_manifest: Path,
) -> dict[str, Any]:
    """Evaluate NoC behavior from one completed NeuroSim artifact."""
    started = time.perf_counter()
    artifacts, _ = _artifacts_from_manifest(workload_manifest)
    build_data = _read_manifest(build_manifest, "simulator-build")
    neurosim_data = _read_manifest(neurosim_manifest, "neurosim")
    expected_inputs = neurosim_data["inputs"]
    if expected_inputs["workload_manifest"] != _file_reference(workload_manifest):
        raise ValueError("BookSim workload does not match the NeuroSim workload")
    if expected_inputs["build_manifest"] != _file_reference(build_manifest):
        raise ValueError("BookSim build does not match the NeuroSim build")
    config = HardwareConfig(**neurosim_data["config"]["values"])
    output_dir = neurosim_manifest.resolve().parent
    result = run_booksim(
        _verify_file(build_data["outputs"]["booksim_binary"]),
        root.resolve(),
        config,
        output_dir,
        artifacts,
    )
    return _write_manifest(
        output_dir / "booksim.json",
        "booksim",
        {
            "inputs": {
                "workload_manifest": _file_reference(workload_manifest),
                "build_manifest": _file_reference(build_manifest),
                "neurosim_manifest": _file_reference(neurosim_manifest),
            },
            "config": {"key": config.key, "values": asdict(config)},
            "outputs": {"log": _file_reference(output_dir / "booksim.log")},
            "metrics": asdict(result),
            "performance": {"wall_seconds": time.perf_counter() - started},
        },
    )


def score_candidates(results: list[dict[str, Any]]) -> None:
    """Add the legacy normalized PPA/NoC score to candidate records."""
    if not results:
        raise ValueError("At least one candidate is required for ranking")
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


def rank_candidates_stage(candidate_manifests: list[Path], output: Path) -> dict[str, Any]:
    """Join completed candidate artifacts and rank them without rerunning simulators."""
    started = time.perf_counter()
    candidates = []
    for path in candidate_manifests:
        booksim_data = _read_manifest(path, "booksim")
        neurosim_path = _verify_file(booksim_data["inputs"]["neurosim_manifest"])
        neurosim_data = _read_manifest(neurosim_path, "neurosim")
        if booksim_data["config"] != neurosim_data["config"]:
            raise ValueError(f"Candidate configuration changed between simulators: {path}")
        config = HardwareConfig(**booksim_data["config"]["values"])
        candidates.append(
            result_dict(
                config,
                NeuroSimResult(**neurosim_data["metrics"]),
                BookSimResult(**booksim_data["metrics"]),
            )
        )
    score_candidates(candidates)
    candidates.sort(key=lambda item: item["score"])
    return _write_manifest(
        output,
        "ranking",
        {
            "inputs": {"candidate_manifests": [_file_reference(path) for path in candidate_manifests]},
            "best": candidates[0],
            "candidates": candidates,
            "metrics": {"candidate_count": len(candidates)},
            "performance": {"wall_seconds": time.perf_counter() - started},
        },
    )


def _values(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one independently measurable NavCim pipeline stage")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    export_parser = subparsers.add_parser("export-model")
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.add_argument("--checkpoint", type=Path)
    export_parser.add_argument("--seed", type=int, default=117)

    graph_parser = subparsers.add_parser("validate-graph")
    graph_parser.add_argument("--model-manifest", type=Path, required=True)
    graph_parser.add_argument("--output", type=Path, required=True)

    workload_parser = subparsers.add_parser("create-workload")
    workload_parser.add_argument("--model-manifest", type=Path, required=True)
    workload_parser.add_argument("--output-dir", type=Path, required=True)

    build_parser = subparsers.add_parser("build-simulators")
    build_parser.add_argument("--root", type=Path, default=_root())
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--jobs", type=int, default=8)

    space_parser = subparsers.add_parser("create-search-space")
    space_parser.add_argument("--output", type=Path, required=True)
    space_parser.add_argument("--sa", default="128")
    space_parser.add_argument("--pe", default="4")
    space_parser.add_argument("--tile", default="8")
    space_parser.add_argument("--adc", default="5")
    space_parser.add_argument("--cell", default="2")
    space_parser.add_argument("--mux", default="8")

    neurosim_parser = subparsers.add_parser("run-neurosim")
    neurosim_parser.add_argument("--workload-manifest", type=Path, required=True)
    neurosim_parser.add_argument("--build-manifest", type=Path, required=True)
    neurosim_parser.add_argument("--search-space-manifest", type=Path, required=True)
    neurosim_parser.add_argument("--config-key", required=True)
    neurosim_parser.add_argument("--output-dir", type=Path, required=True)

    booksim_parser = subparsers.add_parser("run-booksim")
    booksim_parser.add_argument("--root", type=Path, default=_root())
    booksim_parser.add_argument("--workload-manifest", type=Path, required=True)
    booksim_parser.add_argument("--build-manifest", type=Path, required=True)
    booksim_parser.add_argument("--neurosim-manifest", type=Path, required=True)

    rank_parser = subparsers.add_parser("rank")
    rank_parser.add_argument("--candidate-manifest", type=Path, action="append", required=True)
    rank_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.stage == "export-model":
        report = export_model_stage(args.output_dir, args.checkpoint, args.seed)
    elif args.stage == "validate-graph":
        report = validate_graph_stage(args.model_manifest, args.output)
    elif args.stage == "create-workload":
        report = create_workload_stage(args.model_manifest, args.output_dir)
    elif args.stage == "build-simulators":
        report = build_simulators_stage(args.root, args.output, args.jobs)
    elif args.stage == "create-search-space":
        report = create_search_space_stage(
            args.output,
            _values(args.sa),
            _values(args.pe),
            _values(args.tile),
            _values(args.adc),
            _values(args.cell),
            _values(args.mux),
        )
    elif args.stage == "run-neurosim":
        report = run_neurosim_stage(
            args.workload_manifest,
            args.build_manifest,
            args.search_space_manifest,
            args.config_key,
            args.output_dir,
        )
    elif args.stage == "run-booksim":
        report = run_booksim_stage(
            args.root, args.workload_manifest, args.build_manifest, args.neurosim_manifest
        )
    elif args.stage == "rank":
        report = rank_candidates_stage(args.candidate_manifest, args.output)
    else:
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
