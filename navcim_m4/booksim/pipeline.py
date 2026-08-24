from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..simulators import BOOKSIM_FLIT_BITS, _run, parse_booksim_output
from .contracts import file_reference, read_manifest, verify_file, write_manifest
from .features import FEATURES, TARGETS, sample_features
from .models import (
    assemble_bundle_stage,
    compare_top_k_stage,
    create_split_stage,
    evaluate_bundle_stage,
    train_target_stage,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def create_samples_stage(output_dir: Path, count: int, seed: int = 117) -> dict[str, Any]:
    started = time.perf_counter()
    if count < 1:
        raise ValueError("count must be positive")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.csv"
    with samples_path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "seed", *FEATURES))
        writer.writeheader()
        for index in range(count):
            writer.writerow({
                "sample_id": f"sample-{index:06d}",
                "seed": seed + index,
                **asdict(sample_features(seed + index)),
            })
    return write_manifest(
        output_dir / "samples.json",
        "samples",
        {
            "parameters": {"count": count, "seed": seed},
            "outputs": {"samples": file_reference(samples_path)},
            "metrics": {"sample_count": count},
        },
        started,
    )


def prepare_simulator_stage(root: Path, booksim_binary: Path, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    root, booksim_binary, output_dir = root.resolve(), booksim_binary.resolve(), output_dir.resolve()
    if not booksim_binary.is_file():
        raise FileNotFoundError(booksim_binary)
    template = root / "booksim2" / "src" / "examples" / "mesh88_lat"
    tech_file = root / "booksim2" / "src" / "power" / "techfile.txt"
    if not template.is_file() or not tech_file.is_file():
        raise FileNotFoundError("BookSim template or technology file is missing")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = output_dir / "booksim_predictor.cfg"
    text = re.sub(
        r"^tech_file\s*=.*;$",
        f"tech_file = {tech_file};",
        template.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    config.write_text(text, encoding="utf-8")
    return write_manifest(
        output_dir / "simulator.json",
        "simulator",
        {
            "root": str(root),
            "inputs": {
                "booksim_binary": file_reference(booksim_binary),
                "template": file_reference(template),
                "tech_file": file_reference(tech_file),
            },
            "outputs": {"config": file_reference(config)},
            "constants": {"flit_bits": BOOKSIM_FLIT_BITS},
        },
        started,
    )


def _sample_row(samples_manifest: Path, sample_id: str) -> dict[str, str]:
    samples = read_manifest(samples_manifest, "samples")
    path = verify_file(samples["outputs"]["samples"])
    with path.open(newline="", encoding="ascii") as stream:
        matches = [row for row in csv.DictReader(stream) if row["sample_id"] == sample_id]
    if len(matches) != 1:
        raise ValueError(f"Sample id is not unique in the manifest: {sample_id}")
    return matches[0]


def simulate_sample_stage(
    samples_manifest: Path,
    simulator_manifest: Path,
    sample_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    row = _sample_row(samples_manifest, sample_id)
    simulator = read_manifest(simulator_manifest, "simulator")
    binary = verify_file(simulator["inputs"]["booksim_binary"])
    config = verify_file(simulator["outputs"]["config"])
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary), str(config), row["network_size"], row["latency_per_flit"],
        row["tile_width_m"], row["inject_rate"], row["data_size_flits"],
        "0", row["num_start_node"], row["num_start_node"], row["num_dest_node"], "-1",
    ]
    output = _run(command, Path(simulator["root"]) / "booksim2" / "src", timeout=120)
    log = output_dir / "booksim.log"
    log.write_text(output, encoding="utf-8")
    result = parse_booksim_output(output)
    dynamic_power = result.total_power_w
    return write_manifest(
        output_dir / "result.json",
        "sample-result",
        {
            "inputs": {
                "samples_manifest": file_reference(samples_manifest),
                "simulator_manifest": file_reference(simulator_manifest),
                "sample_id": sample_id,
            },
            "features": {name: float(row[name]) for name in FEATURES},
            "metrics": {
                "latency_cycles": result.latency_cycles,
                "dynamic_power_w": dynamic_power,
                "area_m2": result.area_m2,
                "leakage_power_w": result.leakage_power_w,
                "total_power_w": result.total_power_w,
            },
            "outputs": {"log": file_reference(log)},
        },
        started,
    )


def _sample_result_valid(path: Path, samples_manifest: Path, simulator_manifest: Path, sample_id: str) -> bool:
    try:
        result = read_manifest(path, "sample-result")
        if result["inputs"] != {
            "samples_manifest": file_reference(samples_manifest),
            "simulator_manifest": file_reference(simulator_manifest),
            "sample_id": sample_id,
        }:
            return False
        verify_file(result["outputs"]["log"])
        return True
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return False


def simulate_pending_stage(
    samples_manifest: Path,
    simulator_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    samples = read_manifest(samples_manifest, "samples")
    samples_path = verify_file(samples["outputs"]["samples"])
    with samples_path.open(newline="", encoding="ascii") as stream:
        sample_ids = [row["sample_id"] for row in csv.DictReader(stream)]
    completed, reused, failures = [], [], []
    for sample_id in sample_ids:
        sample_dir = output_dir.resolve() / "samples" / sample_id
        result_path = sample_dir / "result.json"
        if result_path.is_file() and _sample_result_valid(result_path, samples_manifest, simulator_manifest, sample_id):
            reused.append(sample_id)
            continue
        try:
            simulate_sample_stage(samples_manifest, simulator_manifest, sample_id, sample_dir)
            completed.append(sample_id)
        except (OSError, RuntimeError, ValueError) as error:
            failures.append({"sample_id": sample_id, "error": str(error)})
    return write_manifest(
        output_dir.resolve() / "simulation.json",
        "simulation-batch",
        {
            "inputs": {
                "samples_manifest": file_reference(samples_manifest),
                "simulator_manifest": file_reference(simulator_manifest),
            },
            "outputs": {"sample_results_dir": str((output_dir.resolve() / "samples"))},
            "metrics": {
                "requested": len(sample_ids),
                "completed": len(completed),
                "reused": len(reused),
                "failed": len(failures),
            },
            "failures": failures,
        },
        started,
    )


def collect_dataset_stage(
    samples_manifest: Path,
    simulation_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    simulation = read_manifest(simulation_manifest, "simulation-batch")
    if simulation["inputs"]["samples_manifest"] != file_reference(samples_manifest):
        raise ValueError("Sample and simulation manifests do not match")
    samples = read_manifest(samples_manifest, "samples")
    samples_path = verify_file(samples["outputs"]["samples"])
    with samples_path.open(newline="", encoding="ascii") as stream:
        sample_rows = {row["sample_id"]: row for row in csv.DictReader(stream)}
    result_dir = Path(simulation["outputs"]["sample_results_dir"])
    rows, missing = [], []
    for sample_id, sample in sample_rows.items():
        result_path = result_dir / sample_id / "result.json"
        if not result_path.is_file():
            missing.append(sample_id)
            continue
        result = read_manifest(result_path, "sample-result")
        if not _sample_result_valid(result_path, samples_manifest, Path(simulation["inputs"]["simulator_manifest"]["path"]), sample_id):
            missing.append(sample_id)
            continue
        rows.append({
            "sample_id": sample_id,
            **{name: sample[name] for name in FEATURES},
            **{name: result["metrics"][name] for name in TARGETS},
        })
    if not rows:
        raise ValueError("No completed BookSim samples are available")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = output_dir / "dataset.csv"
    with dataset.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", *FEATURES, *TARGETS))
        writer.writeheader()
        writer.writerows(rows)
    return write_manifest(
        output_dir / "dataset.json",
        "dataset",
        {
            "inputs": {
                "samples_manifest": file_reference(samples_manifest),
                "simulation_manifest": file_reference(simulation_manifest),
            },
            "outputs": {"dataset": file_reference(dataset)},
            "metrics": {"completed_samples": len(rows), "missing_samples": len(missing)},
            "missing_sample_ids": missing,
        },
        started,
    )


def import_dataset_stage(input_csv: Path, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    input_csv = input_csv.resolve()
    with input_csv.open(newline="", encoding="ascii") as stream:
        source_rows = list(csv.DictReader(stream))
    required = set(FEATURES) | {"latency_cycles", "area_m2", "leakage_power_w"}
    if not source_rows or required - set(source_rows[0]):
        raise ValueError(f"Imported dataset is empty or missing columns: {sorted(required - set(source_rows[0] if source_rows else ())) }")
    if "dynamic_power_w" not in source_rows[0] and "total_power_w" not in source_rows[0]:
        raise ValueError("Imported dataset requires dynamic_power_w or total_power_w")
    rows = []
    for index, source in enumerate(source_rows):
        dynamic_power = float(source["dynamic_power_w"]) if source.get("dynamic_power_w") else float(source["total_power_w"])
        if dynamic_power < 0:
            raise ValueError(f"Imported sample {index} has negative dynamic power")
        rows.append({
            "sample_id": source.get("sample_id") or f"imported-{index:06d}",
            **{name: source[name] for name in FEATURES},
            "latency_cycles": source["latency_cycles"],
            "dynamic_power_w": dynamic_power,
            "area_m2": source["area_m2"],
            "leakage_power_w": source["leakage_power_w"],
        })
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = output_dir / "dataset.csv"
    with dataset.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", *FEATURES, *TARGETS))
        writer.writeheader()
        writer.writerows(rows)
    return write_manifest(
        output_dir / "dataset.json",
        "dataset",
        {
            "inputs": {"imported_dataset": file_reference(input_csv)},
            "outputs": {"dataset": file_reference(dataset)},
            "metrics": {"completed_samples": len(rows), "missing_samples": 0},
            "transformations": ["BookSim Total Power is imported as dynamic_power_w; leakage is reported separately"],
        },
        started,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one isolated BookSim prediction stage")
    stages = parser.add_subparsers(dest="stage", required=True)

    samples = stages.add_parser("create-samples")
    samples.add_argument("--output-dir", type=Path, required=True)
    samples.add_argument("--count", type=int, required=True)
    samples.add_argument("--seed", type=int, default=117)

    simulator = stages.add_parser("prepare-simulator")
    simulator.add_argument("--root", type=Path, default=_root())
    simulator.add_argument("--booksim-binary", type=Path, required=True)
    simulator.add_argument("--output-dir", type=Path, required=True)

    one = stages.add_parser("simulate-sample")
    one.add_argument("--samples-manifest", type=Path, required=True)
    one.add_argument("--simulator-manifest", type=Path, required=True)
    one.add_argument("--sample-id", required=True)
    one.add_argument("--output-dir", type=Path, required=True)

    pending = stages.add_parser("simulate-pending")
    pending.add_argument("--samples-manifest", type=Path, required=True)
    pending.add_argument("--simulator-manifest", type=Path, required=True)
    pending.add_argument("--output-dir", type=Path, required=True)

    dataset = stages.add_parser("collect-dataset")
    dataset.add_argument("--samples-manifest", type=Path, required=True)
    dataset.add_argument("--simulation-manifest", type=Path, required=True)
    dataset.add_argument("--output-dir", type=Path, required=True)

    imported = stages.add_parser("import-dataset")
    imported.add_argument("--input-csv", type=Path, required=True)
    imported.add_argument("--output-dir", type=Path, required=True)

    split = stages.add_parser("create-split")
    split.add_argument("--dataset-manifest", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=117)
    split.add_argument("--test-ratio", type=float, default=0.2)

    target = stages.add_parser("train-target")
    target.add_argument("--dataset-manifest", type=Path, required=True)
    target.add_argument("--split-manifest", type=Path, required=True)
    target.add_argument("--target", choices=TARGETS, required=True)
    target.add_argument("--output-dir", type=Path, required=True)
    target.add_argument("--seed", type=int, default=117)
    target.add_argument("--model-family", choices=("auto", "paper-mlp", "extra-trees", "mlp"), default="auto")

    bundle = stages.add_parser("assemble-bundle")
    bundle.add_argument("--model-manifest", type=Path, action="append", required=True)
    bundle.add_argument("--output-dir", type=Path, required=True)

    evaluate = stages.add_parser("evaluate-bundle")
    evaluate.add_argument("--bundle-manifest", type=Path, required=True)
    evaluate.add_argument("--dataset-manifest", type=Path, required=True)
    evaluate.add_argument("--split-manifest", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    compare = stages.add_parser("compare-top-k")
    compare.add_argument("--bundle-manifest", type=Path, required=True)
    compare.add_argument("--dataset-manifest", type=Path, required=True)
    compare.add_argument("--split-manifest", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--top-k", type=int, default=5)
    compare.add_argument("--validation-pool", type=int, default=100)
    compare.add_argument("--latency-weight", type=float, default=1.0)
    compare.add_argument("--power-weight", type=float, default=1.0)
    compare.add_argument("--area-weight", type=float, default=1.0)


    args = parser.parse_args()
    if args.stage == "create-samples":
        report = create_samples_stage(args.output_dir, args.count, args.seed)
    elif args.stage == "prepare-simulator":
        report = prepare_simulator_stage(args.root, args.booksim_binary, args.output_dir)
    elif args.stage == "simulate-sample":
        report = simulate_sample_stage(args.samples_manifest, args.simulator_manifest, args.sample_id, args.output_dir)
    elif args.stage == "simulate-pending":
        report = simulate_pending_stage(args.samples_manifest, args.simulator_manifest, args.output_dir)
    elif args.stage == "collect-dataset":
        report = collect_dataset_stage(args.samples_manifest, args.simulation_manifest, args.output_dir)
    elif args.stage == "import-dataset":
        report = import_dataset_stage(args.input_csv, args.output_dir)
    elif args.stage == "create-split":
        report = create_split_stage(args.dataset_manifest, args.output, args.seed, args.test_ratio)
    elif args.stage == "train-target":
        report = train_target_stage(args.dataset_manifest, args.split_manifest, args.target, args.output_dir, args.seed, args.model_family)
    elif args.stage == "assemble-bundle":
        report = assemble_bundle_stage(args.model_manifest, args.output_dir)
    elif args.stage == "evaluate-bundle":
        report = evaluate_bundle_stage(args.bundle_manifest, args.dataset_manifest, args.split_manifest, args.output)
    elif args.stage == "compare-top-k":
        report = compare_top_k_stage(
            args.bundle_manifest,
            args.dataset_manifest,
            args.split_manifest,
            args.output,
            args.top_k,
            {
                "latency_cycles": args.latency_weight,
                "dynamic_power_w": args.power_weight,
                "area_m2": args.area_weight,
            },
            args.validation_pool,
        )
    else:
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
