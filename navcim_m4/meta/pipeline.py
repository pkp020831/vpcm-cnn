from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..booksim.inference import candidate_features
from ..booksim.models import PredictorBundle
from ..neurosim.contracts import read_manifest as read_neurosim_manifest
from ..neurosim.contracts import verify_file as verify_neurosim_file
from ..neurosim.pipeline import load_registered_workload
from ..simulators import (
    HardwareConfig,
    parse_neurosim_noc_config,
    run_booksim_layers,
)
from .contracts import file_reference, read_manifest, verify_file, write_manifest
from .models import (
    assemble_bundle_stage,
    create_split_stage,
    evaluate_bundle_stage,
    compare_candidates_stage,
    train_energy_stage,
    train_latency_stage,
)


def register_upstream_stage(
    neurosim_simulation_manifest: Path,
    neurosim_analysis_manifest: Path,
    neurosim_workload_manifest: Path,
    booksim_predictor_bundle: Path,
    booksim_binary: Path,
    root: Path,
    output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    simulation = read_neurosim_manifest(neurosim_simulation_manifest, "simulation-batch")
    analysis = read_neurosim_manifest(neurosim_analysis_manifest, "analysis-batch")
    read_neurosim_manifest(neurosim_workload_manifest, "workload")
    PredictorBundle(booksim_predictor_bundle)
    booksim_binary = booksim_binary.resolve()
    root = root.resolve()
    if not booksim_binary.is_file():
        raise FileNotFoundError(booksim_binary)
    template = root / "booksim2" / "src" / "examples" / "mesh88_lat"
    tech_file = root / "booksim2" / "src" / "power" / "techfile.txt"
    if not template.is_file() or not tech_file.is_file():
        raise FileNotFoundError("BookSim template or technology file is missing")
    return write_manifest(
        output,
        "upstream",
        {
            "inputs": {
                "neurosim_simulation_manifest": file_reference(neurosim_simulation_manifest),
                "neurosim_analysis_manifest": file_reference(neurosim_analysis_manifest),
                "neurosim_workload_manifest": file_reference(neurosim_workload_manifest),
                "booksim_predictor_bundle": file_reference(booksim_predictor_bundle),
                "booksim_binary": file_reference(booksim_binary),
                "booksim_template": file_reference(template),
                "booksim_tech_file": file_reference(tech_file),
            },
            "root": str(root),
            "outputs": {
                "neurosim_runs_dir": simulation["outputs"]["runs_dir"],
                "neurosim_analyses_dir": analysis["outputs"]["analyses_dir"],
            },
            "metrics": {
                "neurosim_candidates": simulation["metrics"]["completed"] + simulation["metrics"]["reused"],
            },
        },
        started,
    )


def run_pair_stage(upstream_manifest: Path, config_key: str, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    upstream = read_manifest(upstream_manifest, "upstream")
    candidate_path = Path(upstream["outputs"]["neurosim_runs_dir"]) / config_key / "candidate.json"
    analysis_path = Path(upstream["outputs"]["neurosim_analyses_dir"]) / config_key / "analysis.json"
    candidate = read_neurosim_manifest(candidate_path, "candidate")
    analysis = read_neurosim_manifest(analysis_path, "analysis")
    neurosim_log = verify_neurosim_file(candidate["outputs"]["log"])
    floorplan = verify_neurosim_file(candidate["outputs"]["floorplan"])
    workload_manifest = verify_file(upstream["inputs"]["neurosim_workload_manifest"])
    artifacts, _ = load_registered_workload(workload_manifest)
    predictor = PredictorBundle(verify_file(upstream["inputs"]["booksim_predictor_bundle"]))
    binary = verify_file(upstream["inputs"]["booksim_binary"])
    config = HardwareConfig(**candidate["config"]["values"])
    output_dir = output_dir.resolve()
    actual_layers = run_booksim_layers(
        binary,
        Path(upstream["root"]),
        config,
        output_dir,
        artifacts,
        candidate_path.parent,
    )
    features = candidate_features(neurosim_log, floorplan, artifacts)
    predicted_layers = predictor.predict_many(list(features))
    if len(actual_layers) != len(features):
        raise ValueError("Actual and predicted BookSim layer counts differ")
    neurosim_layers = {int(layer["layer"]): layer for layer in analysis["layers"]}
    noc = parse_neurosim_noc_config(neurosim_log.read_text(encoding="utf-8"))
    rows = []
    ood = []
    for (layer, actual), layer_features, predicted in zip(actual_layers, features, predicted_layers, strict=True):
        if layer not in neurosim_layers:
            raise ValueError(f"NeuroSim analysis is missing layer {layer}")
        neuro = neurosim_layers[layer]
        violations = predictor.out_of_distribution(layer_features)
        ood.extend(f"layer{layer}:{violation}" for violation in violations)
        actual_booksim_latency_ns = actual.latency_cycles / noc.clock_hz * 1e9
        predicted_booksim_latency_ns = predicted["latency_cycles"] / noc.clock_hz * 1e9
        actual_booksim_energy_pj = actual.total_power_w * actual_booksim_latency_ns * 1000
        rows.append({
            "config_key": config_key,
            "layer": layer,
            **{f"hardware_{name}": value for name, value in candidate["config"]["values"].items()},
            **{f"booksim_feature_{name}": value for name, value in asdict(layer_features).items()},
            "neurosim_latency_ns": neuro["latency_ns"],
            "neurosim_dynamic_energy_pj": neuro["dynamic_energy_pj"],
            "actual_booksim_latency_cycles": actual.latency_cycles,
            "actual_booksim_latency_ns": actual_booksim_latency_ns,
            "actual_booksim_dynamic_power_w": actual.total_power_w,
            "actual_booksim_leakage_power_w": actual.leakage_power_w,
            "actual_booksim_area_m2": actual.area_m2,
            "predicted_booksim_latency_cycles": predicted["latency_cycles"],
            "predicted_booksim_latency_ns": predicted_booksim_latency_ns,
            "predicted_booksim_dynamic_power_w": predicted["dynamic_power_w"],
            "predicted_booksim_leakage_power_w": predicted["leakage_power_w"],
            "predicted_booksim_area_m2": predicted["area_m2"],
            "actual_overall_latency_ns": neuro["latency_ns"] + actual_booksim_latency_ns,
            "actual_booksim_dynamic_energy_pj": actual_booksim_energy_pj,
            "actual_overall_dynamic_energy_pj": neuro["dynamic_energy_pj"] + actual_booksim_energy_pj,
        })
    return write_manifest(
        output_dir / "pair.json",
        "pair",
        {
            "inputs": {
                "upstream_manifest": file_reference(upstream_manifest),
                "neurosim_candidate_manifest": file_reference(candidate_path),
                "neurosim_analysis_manifest": file_reference(analysis_path),
            },
            "config": candidate["config"],
            "chip": analysis["chip"],
            "noc": analysis["noc"],
            "rows": rows,
            "out_of_distribution": sorted(set(ood)),
            "outputs": {
                "booksim_log": file_reference(output_dir / "booksim.log"),
                "booksim_layers": file_reference(output_dir / "booksim_layers.json"),
            },
            "metrics": {"paired_layers": len(rows), "ood_features": len(set(ood))},
        },
        started,
    )


def _pair_valid(path: Path, upstream_manifest: Path, candidate_path: Path, analysis_path: Path) -> bool:
    try:
        pair = read_manifest(path, "pair")
        expected = {
            "upstream_manifest": file_reference(upstream_manifest),
            "neurosim_candidate_manifest": file_reference(candidate_path),
            "neurosim_analysis_manifest": file_reference(analysis_path),
        }
        if pair["inputs"] != expected:
            return False
        verify_file(pair["outputs"]["booksim_log"])
        verify_file(pair["outputs"]["booksim_layers"])
        return True
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return False


def _pair_artifacts_valid(path: Path, candidate_path: Path, analysis_path: Path) -> bool:
    try:
        pair = read_manifest(path, "pair")
        if pair["inputs"]["neurosim_candidate_manifest"] != file_reference(candidate_path):
            return False
        if pair["inputs"]["neurosim_analysis_manifest"] != file_reference(analysis_path):
            return False
        verify_file(pair["outputs"]["booksim_log"])
        verify_file(pair["outputs"]["booksim_layers"])
        return True
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return False


def run_pending_stage(upstream_manifest: Path, output_dir: Path, workers: int = 2, force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    upstream = read_manifest(upstream_manifest, "upstream")
    runs_dir = Path(upstream["outputs"]["neurosim_runs_dir"])
    analyses_dir = Path(upstream["outputs"]["neurosim_analyses_dir"])
    output_dir = output_dir.resolve()
    pending, reused = [], []
    for candidate_path in sorted(runs_dir.glob("*/candidate.json")):
        key = candidate_path.parent.name
        analysis_path = analyses_dir / key / "analysis.json"
        pair_path = output_dir / "pairs" / key / "pair.json"
        if not analysis_path.is_file():
            continue
        if not force and pair_path.is_file() and _pair_valid(pair_path, upstream_manifest, candidate_path, analysis_path):
            reused.append(key)
        else:
            pending.append(key)
    completed, failures = [], []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as executor:
        futures = {
            executor.submit(run_pair_stage, upstream_manifest, key, output_dir / "pairs" / key): key
            for key in pending
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                future.result()
                completed.append(key)
            except (OSError, RuntimeError, ValueError) as error:
                failures.append({"config_key": key, "error": str(error)})
    return write_manifest(
        output_dir / "pairing.json",
        "pairing-batch",
        {
            "inputs": {"upstream_manifest": file_reference(upstream_manifest)},
            "outputs": {"pairs_dir": str((output_dir / "pairs").resolve())},
            "metrics": {
                "requested": len(pending) + len(reused),
                "completed": len(completed),
                "reused": len(reused),
                "failed": len(failures),
                "workers": max(1, min(workers, 4)),
            },
            "failures": failures,
        },
        started,
    )


def summarize_pairing_stage(upstream_manifest: Path, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    upstream = read_manifest(upstream_manifest, "upstream")
    runs_dir = Path(upstream["outputs"]["neurosim_runs_dir"])
    analyses_dir = Path(upstream["outputs"]["neurosim_analyses_dir"])
    output_dir = output_dir.resolve()
    completed, incomplete = [], []
    for candidate_path in sorted(runs_dir.glob("*/candidate.json")):
        key = candidate_path.parent.name
        analysis_path = analyses_dir / key / "analysis.json"
        pair_path = output_dir / "pairs" / key / "pair.json"
        if analysis_path.is_file() and pair_path.is_file() and _pair_valid(pair_path, upstream_manifest, candidate_path, analysis_path):
            completed.append(key)
        else:
            incomplete.append(key)
    return write_manifest(
        output_dir / "pairing.json",
        "pairing-batch",
        {
            "inputs": {"upstream_manifest": file_reference(upstream_manifest)},
            "outputs": {"pairs_dir": str((output_dir / "pairs").resolve())},
            "metrics": {
                "requested": len(completed) + len(incomplete),
                "completed": len(completed),
                "reused": 0,
                "failed": len(incomplete),
                "workers": 0,
            },
            "completed_config_keys": completed,
            "failures": [{"config_key": key, "error": "pair output is missing or incomplete"} for key in incomplete],
        },
        started,
    )


def collect_dataset_stage(pairing_manifest: Path, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    pairing = read_manifest(pairing_manifest, "pairing-batch")
    expected_upstream = pairing["inputs"]["upstream_manifest"]
    verify_file(expected_upstream)
    pair_paths = [
        Path(pairing["outputs"]["pairs_dir"]) / key / "pair.json"
        for key in pairing.get("completed_config_keys", [])
    ]
    rows = []
    configs = []
    for path in pair_paths:
        pair = read_manifest(path, "pair")
        if pair["inputs"]["upstream_manifest"] != expected_upstream:
            raise ValueError(f"Pair was generated from a different upstream manifest: {path}")
        rows.extend({
            **row,
            "neurosim_chip_area_um2": pair["chip"]["area_um2"],
            "neurosim_chip_leakage_power_uw": pair["chip"]["leakage_power_uw"],
            "neurosim_chip_latency_ns": pair["chip"]["latency_ns"],
            "neurosim_chip_dynamic_energy_pj": pair["chip"]["dynamic_energy_pj"],
        } for row in pair["rows"])
        configs.append(pair["config"]["key"])
    if not rows:
        raise ValueError("No paired meta rows are available")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = output_dir / "dataset.csv"
    with dataset.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return write_manifest(
        output_dir / "dataset.json",
        "dataset",
        {
            "inputs": {"pairing_manifest": file_reference(pairing_manifest)},
            "outputs": {"dataset": file_reference(dataset)},
            "metrics": {"configs": len(set(configs)), "rows": len(rows)},
            "config_keys": sorted(set(configs)),
        },
        started,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one NeuroSim+BookSim meta-learner pipeline stage")
    stages = parser.add_subparsers(dest="stage", required=True)

    upstream = stages.add_parser("register-upstream")
    upstream.add_argument("--neurosim-simulation-manifest", type=Path, required=True)
    upstream.add_argument("--neurosim-analysis-manifest", type=Path, required=True)
    upstream.add_argument("--neurosim-workload-manifest", type=Path, required=True)
    upstream.add_argument("--booksim-predictor-bundle", type=Path, required=True)
    upstream.add_argument("--booksim-binary", type=Path, required=True)
    upstream.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    upstream.add_argument("--output", type=Path, required=True)

    pair = stages.add_parser("run-pair")
    pair.add_argument("--upstream-manifest", type=Path, required=True)
    pair.add_argument("--config-key", required=True)
    pair.add_argument("--output-dir", type=Path, required=True)

    pending = stages.add_parser("run-pending")
    pending.add_argument("--upstream-manifest", type=Path, required=True)
    pending.add_argument("--output-dir", type=Path, required=True)
    pending.add_argument("--workers", type=int, default=2)
    pending.add_argument("--force", action="store_true")

    summarize = stages.add_parser("summarize-pairs")
    summarize.add_argument("--upstream-manifest", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)

    dataset = stages.add_parser("collect-dataset")
    dataset.add_argument("--pairing-manifest", type=Path, required=True)
    dataset.add_argument("--output-dir", type=Path, required=True)

    split = stages.add_parser("create-split")
    split.add_argument("--dataset-manifest", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=117)
    split.add_argument("--test-ratio", type=float, default=0.25)

    latency = stages.add_parser("train-latency")
    latency.add_argument("--dataset-manifest", type=Path, required=True)
    latency.add_argument("--split-manifest", type=Path, required=True)
    latency.add_argument("--output-dir", type=Path, required=True)

    energy = stages.add_parser("train-energy")
    energy.add_argument("--dataset-manifest", type=Path, required=True)
    energy.add_argument("--split-manifest", type=Path, required=True)
    energy.add_argument("--latency-model-manifest", type=Path, required=True)
    energy.add_argument("--output-dir", type=Path, required=True)
    energy.add_argument("--seed", type=int, default=117)
    energy.add_argument("--model-family", choices=("auto", "mlp", "polynomial", "extra-trees"), default="auto")

    bundle = stages.add_parser("assemble-bundle")
    bundle.add_argument("--latency-model-manifest", type=Path, required=True)
    bundle.add_argument("--energy-model-manifest", type=Path, required=True)
    bundle.add_argument("--output", type=Path, required=True)

    evaluate = stages.add_parser("evaluate-bundle")
    evaluate.add_argument("--bundle-manifest", type=Path, required=True)
    evaluate.add_argument("--dataset-manifest", type=Path, required=True)
    evaluate.add_argument("--split-manifest", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    compare = stages.add_parser("compare-candidates")
    compare.add_argument("--bundle-manifest", type=Path, required=True)
    compare.add_argument("--dataset-manifest", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()
    if args.stage == "register-upstream":
        report = register_upstream_stage(args.neurosim_simulation_manifest, args.neurosim_analysis_manifest, args.neurosim_workload_manifest, args.booksim_predictor_bundle, args.booksim_binary, args.root, args.output)
    elif args.stage == "run-pair":
        report = run_pair_stage(args.upstream_manifest, args.config_key, args.output_dir)
    elif args.stage == "run-pending":
        report = run_pending_stage(args.upstream_manifest, args.output_dir, args.workers, args.force)
    elif args.stage == "summarize-pairs":
        report = summarize_pairing_stage(args.upstream_manifest, args.output_dir)
    elif args.stage == "collect-dataset":
        report = collect_dataset_stage(args.pairing_manifest, args.output_dir)
    elif args.stage == "create-split":
        report = create_split_stage(args.dataset_manifest, args.output, args.seed, args.test_ratio)
    elif args.stage == "train-latency":
        report = train_latency_stage(args.dataset_manifest, args.split_manifest, args.output_dir)
    elif args.stage == "train-energy":
        report = train_energy_stage(args.dataset_manifest, args.split_manifest, args.latency_model_manifest, args.output_dir, args.seed, args.model_family)
    elif args.stage == "assemble-bundle":
        report = assemble_bundle_stage(args.latency_model_manifest, args.energy_model_manifest, args.output)
    elif args.stage == "evaluate-bundle":
        report = evaluate_bundle_stage(args.bundle_manifest, args.dataset_manifest, args.split_manifest, args.output)
    elif args.stage == "compare-candidates":
        report = compare_candidates_stage(args.bundle_manifest, args.dataset_manifest, args.output, args.top_k)
    else:
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
