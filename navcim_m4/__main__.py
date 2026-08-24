from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from .accuracy import CrossSimConfig, evaluate_crosssim, run_prescribed_sweep, train_vgg11
from .graph import validate_with_tvm
from .models import create_model, export_onnx
from .search import run_prescribed_hardware_sweep, run_search
from .simulators import build_booksim, build_neurosim


def _values(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _crosssim_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--adc-bits", type=int, default=5)
    parser.add_argument("--dac-bits", type=int, default=8)
    parser.add_argument("--cell-bits", type=int, default=2)
    parser.add_argument("--subarray", type=int, default=128)
    parser.add_argument("--weight-slices", type=int, default=1)
    parser.add_argument("--input-slice-size", type=int, default=1)
    parser.add_argument("--programming-error", type=float, default=0.0)
    parser.add_argument("--read-noise", type=float, default=0.0)
    parser.add_argument("--lumped-read-noise", action="store_true")
    parser.add_argument("--drift-time", type=float, default=0.0)
    parser.add_argument("--wire-resistance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=117)


def _crosssim_config(args: argparse.Namespace) -> CrossSimConfig:
    return CrossSimConfig(args.adc_bits, args.dac_bits, args.cell_bits, args.subarray, args.subarray, args.weight_slices, args.input_slice_size, args.programming_error, args.read_noise, args.lumped_read_noise, args.drift_time, args.wire_resistance, args.seed)


def doctor() -> int:
    import numpy
    import onnx
    import torch
    import tvm

    report = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            "numpy": numpy.__version__,
            "onnx": onnx.__version__,
            "torch": torch.__version__,
            "tvm": tvm.__version__,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if platform.system() == "Darwin" and platform.machine() != "arm64":
        print("Expected native ARM64 Python on macOS", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="NavCim ARM64 core runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--jobs", type=int, default=8)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--output", type=Path, default=Path("outputs/vgg11_cifar10.onnx"))
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, default=Path("outputs/vgg11-cifar10"))
    run_parser.add_argument("--sa", default="128")
    run_parser.add_argument("--pe", default="4")
    run_parser.add_argument("--tile", default="8")
    run_parser.add_argument("--adc", default="5")
    run_parser.add_argument("--cell", default="2")
    run_parser.add_argument("--mux", default="8")
    run_parser.add_argument("--workers", type=int, default=4)
    run_parser.add_argument("--jobs", type=int, default=8)
    run_parser.add_argument("--checkpoint", type=Path)
    run_parser.add_argument("--crosssim", action="store_true")
    run_parser.add_argument("--crosssim-samples", type=int, default=100)
    run_parser.add_argument("--crosssim-runs", type=int, default=1)
    run_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    run_parser.add_argument("--booksim-mode", choices=("simulator", "predictor"), default="simulator")
    run_parser.add_argument("--predictor-bundle", type=Path)
    run_parser.add_argument("--no-predictor-fallback", action="store_true")
    run_parser.add_argument("--ranking-mode", choices=("legacy", "pareto-topsis"), default="pareto-topsis")
    run_parser.add_argument("--latency-weight", type=float, default=1.0)
    run_parser.add_argument("--power-weight", type=float, default=1.0)
    run_parser.add_argument("--area-weight", type=float, default=1.0)
    run_parser.add_argument("--validation-top-k", type=int, default=100)
    run_parser.add_argument("--meta-bundle", type=Path)
    run_parser.add_argument("--allow-predictor-ood", action="store_true")
    _crosssim_arguments(run_parser)
    sweep_parser = subparsers.add_parser("sweep")
    sweep_parser.add_argument("--output-dir", type=Path, default=Path("outputs/vgg11-cifar10-sweep"))
    sweep_parser.add_argument("--workers", type=int, default=4)
    sweep_parser.add_argument("--jobs", type=int, default=8)
    sweep_parser.add_argument("--checkpoint", type=Path)
    sweep_parser.add_argument("--crosssim", action="store_true")
    sweep_parser.add_argument("--crosssim-samples", type=int, default=100)
    sweep_parser.add_argument("--crosssim-runs", type=int, default=1)
    sweep_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    _crosssim_arguments(sweep_parser)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--seed", type=int, default=117)
    train_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    train_parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/vgg11-cifar10.pt"))
    train_parser.add_argument("--train-samples", type=int)
    train_parser.add_argument("--checkpoint-interval", type=int, default=10)
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--force", action="store_true")
    crosssim_parser = subparsers.add_parser("crosssim")
    crosssim_parser.add_argument("--checkpoint", type=Path, required=True)
    crosssim_parser.add_argument("--samples", type=int, default=100)
    crosssim_parser.add_argument("--runs", type=int, default=1)
    crosssim_parser.add_argument("--batch-size", type=int, default=32)
    crosssim_parser.add_argument("--device", choices=("cpu",), default="cpu")
    crosssim_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    crosssim_parser.add_argument("--output", type=Path, default=Path("outputs/crosssim"))
    _crosssim_arguments(crosssim_parser)
    sweep_parser = subparsers.add_parser("crosssim-sweep")
    sweep_parser.add_argument("--checkpoint", type=Path, required=True)
    sweep_parser.add_argument("--samples", type=int, default=100)
    sweep_parser.add_argument("--batch-size", type=int, default=100)
    sweep_parser.add_argument("--seed", type=int, default=117)
    sweep_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    sweep_parser.add_argument("--output", type=Path, default=Path("outputs/crosssim-sweep"))
    args = parser.parse_args()

    if args.command == "doctor":
        return doctor()
    if args.command == "build":
        print(build_neurosim(_root(), args.jobs))
        print(build_booksim(_root(), args.jobs))
        return 0
    if args.command == "export":
        path = export_onnx(create_model(), args.output)
        print(json.dumps(validate_with_tvm(path).__dict__, indent=2))
        return 0
    if args.command == "train":
        print(json.dumps(train_vgg11(args.data_dir, args.checkpoint, args.device, args.epochs, args.batch_size, args.seed, not args.force and not args.resume, args.train_samples, args.checkpoint_interval, args.resume), indent=2, sort_keys=True))
        return 0
    if args.command == "crosssim":
        print(json.dumps(evaluate_crosssim(args.checkpoint, args.data_dir, args.samples, args.batch_size, args.runs, _crosssim_config(args), args.output), indent=2, sort_keys=True))
        return 0
    if args.command == "crosssim-sweep":
        report = run_prescribed_sweep(args.checkpoint, args.data_dir, args.output, args.samples, args.batch_size, args.seed)
        print(json.dumps({"conditions": len(report["conditions"]), "output": str(args.output)}, sort_keys=True))
        return 0
    if args.command == "run":
        report = run_search(
            _root(),
            args.output_dir,
            _values(args.sa),
            _values(args.pe),
            _values(args.tile),
            args.workers,
            args.jobs,
            args.checkpoint,
            _crosssim_config(args) if args.crosssim else None,
            args.crosssim_samples,
            args.crosssim_runs,
            args.data_dir,
            _values(args.adc),
            _values(args.cell),
            _values(args.mux),
            booksim_mode=args.booksim_mode,
            predictor_bundle=args.predictor_bundle,
            predictor_fallback=not args.no_predictor_fallback,
            ranking_mode=args.ranking_mode,
            ranking_weights={
                "latency_ns": args.latency_weight,
                "total_power_mw": args.power_weight,
                "area_um2": args.area_weight,
            },
            validation_top_k=args.validation_top_k,
            meta_bundle=args.meta_bundle,
            allow_predictor_ood=args.allow_predictor_ood,
        )
        print(json.dumps(report["best"], indent=2, sort_keys=True))
        return 0
    if args.command == "sweep":
        report = run_prescribed_hardware_sweep(
            _root(), args.output_dir, args.workers, args.jobs, args.checkpoint,
            _crosssim_config(args) if args.crosssim else None,
            args.crosssim_samples, args.crosssim_runs, args.data_dir,
        )
        print(json.dumps({"candidates": len(report["candidates"]), "best": report["best"]}, indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
