from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..layers import LayerArtifacts, LayerRecord
from ..simulators import HardwareConfig, run_neurosim
from ..booksim.ranking import pareto_front, topsis_rank
from .analysis import analyze_candidate_stage
from .contracts import file_reference, read_manifest, verify_file, write_manifest


MAIN_SCHEMA = "navcim.pipeline/v1"


def _read_main_manifest(path: Path, stage: str) -> dict[str, Any]:
    manifest = json.loads(path.resolve().read_text(encoding="utf-8"))
    if manifest.get("schema") != MAIN_SCHEMA or manifest.get("stage") != stage:
        raise ValueError(f"Expected {MAIN_SCHEMA} {stage!r} manifest: {path}")
    return manifest


def _path(value: str | dict[str, str]) -> Path:
    return Path(value["path"] if isinstance(value, dict) else value)


def register_workload_stage(workload_manifest: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    workload = _read_main_manifest(workload_manifest, "workload-generation")
    layers = []
    for layer in workload["outputs"]["layers"]:
        layers.append({
            "record": layer["record"],
            "weight_file": file_reference(_path(layer["weight_file"])),
            "input_file": file_reference(_path(layer["input_file"])),
        })
    return write_manifest(
        output,
        "workload",
        {
            "inputs": {"source_manifest": str(workload_manifest.resolve())},
            "outputs": {
                "network_csv": file_reference(_path(workload["outputs"]["network_csv"])),
                "layers": layers,
            },
            "metrics": {"layer_count": len(layers)},
        },
        started,
    )


def register_search_space_stage(search_space_manifest: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    search_space = _read_main_manifest(search_space_manifest, "search-space")
    configs = search_space["configs"]
    if not configs:
        raise ValueError("Search space contains no candidates")
    return write_manifest(
        output,
        "search-space",
        {
            "inputs": {"source_manifest": str(search_space_manifest.resolve())},
            "configs": configs,
            "metrics": {"candidate_count": len(configs)},
        },
        started,
    )


def prepare_simulator_stage(binary: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    binary = binary.resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    return write_manifest(
        output,
        "simulator",
        {
            "outputs": {"binary": file_reference(binary)},
            "constants": {
                "weight_bits": 8,
                "activation_bits": 8,
                "model": "VGG11_CIFAR10",
            },
        },
        started,
    )


def load_registered_workload(path: Path) -> tuple[LayerArtifacts, Path]:
    manifest = read_manifest(path, "workload")
    layers = manifest["outputs"]["layers"]
    artifacts = LayerArtifacts(
        tuple(LayerRecord(**layer["record"]) for layer in layers),
        tuple(verify_file(layer["weight_file"]) for layer in layers),
        tuple(verify_file(layer["input_file"]) for layer in layers),
    )
    return artifacts, verify_file(manifest["outputs"]["network_csv"])


_workload = load_registered_workload


def _config(path: Path, key: str) -> HardwareConfig:
    search_space = read_manifest(path, "search-space")
    matches = [entry for entry in search_space["configs"] if entry["key"] == key]
    if len(matches) != 1:
        raise ValueError(f"Configuration key is not unique: {key}")
    return HardwareConfig(**matches[0]["values"])


def run_candidate_stage(
    workload_manifest: Path,
    search_space_manifest: Path,
    simulator_manifest: Path,
    config_key: str,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    artifacts, network_csv = _workload(workload_manifest)
    config = _config(search_space_manifest, config_key)
    simulator = read_manifest(simulator_manifest, "simulator")
    binary = verify_file(simulator["outputs"]["binary"])
    output_dir = output_dir.resolve()
    result = run_neurosim(binary, network_csv, artifacts, config, output_dir)
    return write_manifest(
        output_dir / "candidate.json",
        "candidate",
        {
            "inputs": {
                "workload_manifest": file_reference(workload_manifest),
                "search_space_manifest": file_reference(search_space_manifest),
                "simulator_manifest": file_reference(simulator_manifest),
            },
            "config": {"key": config.key, "values": asdict(config)},
            "chip": asdict(result),
            "outputs": {
                "log": file_reference(output_dir / "neurosim.log"),
                "floorplan": file_reference(output_dir / "neurosim_floorplan.csv"),
            },
        },
        started,
    )


def _candidate_valid(path: Path, expected_inputs: dict[str, dict[str, str]]) -> bool:
    try:
        candidate = read_manifest(path, "candidate")
        if candidate["inputs"] != expected_inputs:
            return False
        verify_file(candidate["outputs"]["log"])
        verify_file(candidate["outputs"]["floorplan"])
        return True
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return False


def run_pending_stage(
    workload_manifest: Path,
    search_space_manifest: Path,
    simulator_manifest: Path,
    output_dir: Path,
    workers: int = 2,
    force: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    search_space = read_manifest(search_space_manifest, "search-space")
    output_dir = output_dir.resolve()
    pending, reused = [], []
    expected_inputs = {
        "workload_manifest": file_reference(workload_manifest),
        "search_space_manifest": file_reference(search_space_manifest),
        "simulator_manifest": file_reference(simulator_manifest),
    }
    for entry in search_space["configs"]:
        manifest = output_dir / "runs" / entry["key"] / "candidate.json"
        if not force and manifest.is_file() and _candidate_valid(manifest, expected_inputs):
            reused.append(entry["key"])
        else:
            pending.append(entry["key"])
    completed, failures = [], []

    def run(key: str) -> None:
        run_candidate_stage(
            workload_manifest,
            search_space_manifest,
            simulator_manifest,
            key,
            output_dir / "runs" / key,
        )

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as executor:
        futures = {executor.submit(run, key): key for key in pending}
        for future in as_completed(futures):
            key = futures[future]
            try:
                future.result()
                completed.append(key)
            except (OSError, RuntimeError, ValueError) as error:
                failures.append({"config_key": key, "error": str(error)})
    return write_manifest(
        output_dir / "simulation.json",
        "simulation-batch",
        {
            "inputs": {
                "workload_manifest": str(workload_manifest.resolve()),
                "search_space_manifest": str(search_space_manifest.resolve()),
                "simulator_manifest": str(simulator_manifest.resolve()),
            },
            "outputs": {"runs_dir": str((output_dir / "runs").resolve())},
            "config_keys": [entry["key"] for entry in search_space["configs"]],
            "metrics": {
                "requested": len(search_space["configs"]),
                "completed": len(completed),
                "reused": len(reused),
                "failed": len(failures),
                "workers": max(1, min(workers, 4)),
            },
            "failures": failures,
        },
        started,
    )


def _analysis_valid(path: Path, candidate_path: Path) -> bool:
    try:
        analysis = read_manifest(path, "analysis")
        if analysis["inputs"]["candidate_manifest"] != file_reference(candidate_path):
            return False
        verify_file(analysis["outputs"]["layer_csv"])
        return True
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return False


def analyze_pending_stage(simulation_manifest: Path, output_dir: Path, force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    simulation = read_manifest(simulation_manifest, "simulation-batch")
    runs_dir = Path(simulation["outputs"]["runs_dir"])
    config_keys = set(simulation["config_keys"])
    output_dir = output_dir.resolve()
    completed, reused, failures = [], [], []
    for candidate_path in sorted(runs_dir.glob("*/candidate.json")):
        key = candidate_path.parent.name
        if key not in config_keys:
            continue
        analysis_path = output_dir / key / "analysis.json"
        if not force and analysis_path.is_file() and _analysis_valid(analysis_path, candidate_path):
            reused.append(key)
            continue
        try:
            analyze_candidate_stage(candidate_path, output_dir / key)
            completed.append(key)
        except (OSError, ValueError) as error:
            failures.append({"config_key": key, "error": str(error)})
    return write_manifest(
        output_dir / "analysis.json",
        "analysis-batch",
        {
            "inputs": {"simulation_manifest": str(simulation_manifest.resolve())},
            "outputs": {"analyses_dir": str(output_dir)},
            "config_keys": sorted(config_keys),
            "metrics": {
                "completed": len(completed),
                "reused": len(reused),
                "failed": len(failures),
            },
            "failures": failures,
        },
        started,
    )


def collect_results_stage(
    simulation_manifest: Path,
    analysis_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    simulation = read_manifest(simulation_manifest, "simulation-batch")
    analysis_batch = read_manifest(analysis_manifest, "analysis-batch")
    runs_dir = Path(simulation["outputs"]["runs_dir"])
    analyses_dir = Path(analysis_batch["outputs"]["analyses_dir"])
    config_keys = set(simulation["config_keys"])
    if set(analysis_batch["config_keys"]) != config_keys:
        raise ValueError("NeuroSim analysis does not cover the simulation config set")
    candidates = []
    for candidate_path in sorted(runs_dir.glob("*/candidate.json")):
        key = candidate_path.parent.name
        if key not in config_keys:
            continue
        analysis_path = analyses_dir / key / "analysis.json"
        if not analysis_path.is_file():
            continue
        candidate = read_manifest(candidate_path, "candidate")
        analysis = read_manifest(analysis_path, "analysis")
        chip = candidate["chip"]
        total_power_mw = chip["dynamic_energy_pj"] / chip["latency_ns"] + chip["leakage_power_uw"] / 1000
        candidates.append({
            "config_key": key,
            **candidate["config"]["values"],
            **chip,
            "total_power_mw": total_power_mw,
            "largest_latency_layer": analysis["metrics"]["largest_latency_layer"],
            "largest_energy_layer": analysis["metrics"]["largest_energy_layer"],
            "simulation_wall_seconds": candidate["performance"]["wall_seconds"],
            "analysis_wall_seconds": analysis["performance"]["wall_seconds"],
            "candidate_manifest": str(candidate_path.resolve()),
            "analysis_manifest": str(analysis_path.resolve()),
        })
    if not candidates:
        raise ValueError("No analyzed NeuroSim candidates are available")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_csv = output_dir / "results.csv"
    with result_csv.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    return write_manifest(
        output_dir / "results.json",
        "results",
        {
            "inputs": {
                "simulation_manifest": str(simulation_manifest.resolve()),
                "analysis_manifest": str(analysis_manifest.resolve()),
            },
            "outputs": {"results_csv": file_reference(result_csv)},
            "candidates": candidates,
            "metrics": {"candidate_count": len(candidates)},
        },
        started,
    )


def rank_results_stage(
    results_manifest: Path,
    output: Path,
    latency_weight: float = 1.0,
    power_weight: float = 1.0,
    area_weight: float = 1.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    results = read_manifest(results_manifest, "results")
    objectives = {"latency_ns": "min", "total_power_mw": "min", "area_um2": "min"}
    weights = {
        "latency_ns": latency_weight,
        "total_power_mw": power_weight,
        "area_um2": area_weight,
    }
    front = pareto_front(results["candidates"], objectives)
    ranked = topsis_rank(front, objectives, weights)
    return write_manifest(
        output,
        "ranking",
        {
            "inputs": {"results_manifest": str(results_manifest.resolve())},
            "parameters": {"objectives": objectives, "weights": weights},
            "best": ranked[0],
            "pareto_candidates": ranked,
            "metrics": {
                "candidate_count": len(results["candidates"]),
                "pareto_count": len(ranked),
            },
        },
        started,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one isolated NeuroSim pipeline stage")
    stages = parser.add_subparsers(dest="stage", required=True)

    workload = stages.add_parser("register-workload")
    workload.add_argument("--workload-manifest", type=Path, required=True)
    workload.add_argument("--output", type=Path, required=True)

    space = stages.add_parser("register-search-space")
    space.add_argument("--search-space-manifest", type=Path, required=True)
    space.add_argument("--output", type=Path, required=True)

    simulator = stages.add_parser("prepare-simulator")
    simulator.add_argument("--binary", type=Path, required=True)
    simulator.add_argument("--output", type=Path, required=True)

    candidate = stages.add_parser("run-candidate")
    candidate.add_argument("--workload-manifest", type=Path, required=True)
    candidate.add_argument("--search-space-manifest", type=Path, required=True)
    candidate.add_argument("--simulator-manifest", type=Path, required=True)
    candidate.add_argument("--config-key", required=True)
    candidate.add_argument("--output-dir", type=Path, required=True)

    pending = stages.add_parser("run-pending")
    pending.add_argument("--workload-manifest", type=Path, required=True)
    pending.add_argument("--search-space-manifest", type=Path, required=True)
    pending.add_argument("--simulator-manifest", type=Path, required=True)
    pending.add_argument("--output-dir", type=Path, required=True)
    pending.add_argument("--workers", type=int, default=2)
    pending.add_argument("--force", action="store_true")

    analyze = stages.add_parser("analyze-candidate")
    analyze.add_argument("--candidate-manifest", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)

    analyze_pending = stages.add_parser("analyze-pending")
    analyze_pending.add_argument("--simulation-manifest", type=Path, required=True)
    analyze_pending.add_argument("--output-dir", type=Path, required=True)
    analyze_pending.add_argument("--force", action="store_true")

    collect = stages.add_parser("collect-results")
    collect.add_argument("--simulation-manifest", type=Path, required=True)
    collect.add_argument("--analysis-manifest", type=Path, required=True)
    collect.add_argument("--output-dir", type=Path, required=True)

    rank = stages.add_parser("rank-results")
    rank.add_argument("--results-manifest", type=Path, required=True)
    rank.add_argument("--output", type=Path, required=True)
    rank.add_argument("--latency-weight", type=float, default=1.0)
    rank.add_argument("--power-weight", type=float, default=1.0)
    rank.add_argument("--area-weight", type=float, default=1.0)

    args = parser.parse_args()
    if args.stage == "register-workload":
        report = register_workload_stage(args.workload_manifest, args.output)
    elif args.stage == "register-search-space":
        report = register_search_space_stage(args.search_space_manifest, args.output)
    elif args.stage == "prepare-simulator":
        report = prepare_simulator_stage(args.binary, args.output)
    elif args.stage == "run-candidate":
        report = run_candidate_stage(args.workload_manifest, args.search_space_manifest, args.simulator_manifest, args.config_key, args.output_dir)
    elif args.stage == "run-pending":
        report = run_pending_stage(args.workload_manifest, args.search_space_manifest, args.simulator_manifest, args.output_dir, args.workers, args.force)
    elif args.stage == "analyze-candidate":
        report = analyze_candidate_stage(args.candidate_manifest, args.output_dir)
    elif args.stage == "analyze-pending":
        report = analyze_pending_stage(args.simulation_manifest, args.output_dir, args.force)
    elif args.stage == "collect-results":
        report = collect_results_stage(args.simulation_manifest, args.analysis_manifest, args.output_dir)
    elif args.stage == "rank-results":
        report = rank_results_stage(args.results_manifest, args.output, args.latency_weight, args.power_weight, args.area_weight)
    else:
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
