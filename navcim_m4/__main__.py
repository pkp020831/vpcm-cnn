from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from .graph import validate_with_tvm
from .models import create_model, export_onnx
from .search import run_search
from .simulators import build_booksim, build_neurosim


def _values(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


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
    run_parser.add_argument("--workers", type=int, default=4)
    run_parser.add_argument("--jobs", type=int, default=8)
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
    if args.command == "run":
        report = run_search(
            _root(),
            args.output_dir,
            _values(args.sa),
            _values(args.pe),
            _values(args.tile),
            args.workers,
            args.jobs,
        )
        print(json.dumps(report["best"], indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
