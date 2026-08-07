from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .layers import LayerArtifacts


FLOAT = r"([-+0-9.eE]+)"


@dataclass(frozen=True)
class HardwareConfig:
    sa_row: int = 128
    sa_col: int = 128
    pe: int = 4
    tile: int = 8
    adc_bits: int = 5
    cell_bits: int = 2

    @property
    def key(self) -> str:
        return (
            f"sa{self.sa_row}x{self.sa_col}_pe{self.pe}_tile{self.tile}_"
            f"adc{self.adc_bits}_cell{self.cell_bits}"
        )


@dataclass(frozen=True)
class NeuroSimResult:
    latency_ns: float
    dynamic_energy_pj: float
    leakage_power_uw: float
    area_um2: float


@dataclass(frozen=True)
class BookSimResult:
    latency_cycles: float
    total_power_w: float
    leakage_power_w: float
    area_m2: float


def _run(command: list[str], cwd: Path, timeout: int = 1800, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if completed.returncode:
        tail = "\n".join(completed.stdout.splitlines()[-40:])
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}\n{tail}"
        )
    return completed.stdout


def build_neurosim(root: Path, jobs: int = 8) -> Path:
    source = root / "Inference_pytorch" / "NeuroSIM"
    build = root / "build" / "neurosim"
    command = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if platform.system() == "Darwin":
        command.append("-DCMAKE_OSX_ARCHITECTURES=arm64")
    _run(command, root)
    _run(["cmake", "--build", str(build), "--parallel", str(max(1, min(jobs, 8)))], root)
    binary = build / "neurosim"
    if not binary.is_file():
        raise FileNotFoundError(binary)
    return binary


def build_booksim(root: Path, jobs: int = 8) -> Path:
    source = root / "booksim2" / "src"
    if not source.is_dir():
        raise FileNotFoundError("BookSim2 is missing; initialize the booksim2 submodule")
    environment = os.environ.copy()
    environment.update({"CC": "/usr/bin/clang", "CXX": "/usr/bin/clang++"})
    bison = Path("/opt/homebrew/opt/bison/bin")
    if bison.is_dir():
        environment["PATH"] = f"{bison}:{environment['PATH']}"
    _run(["make", "-j", str(max(1, min(jobs, 8)))], source, env=environment)
    binary = source / "booksim"
    if not binary.is_file():
        raise FileNotFoundError(binary)
    return binary


def _metric(output: str, pattern: str, label: str) -> float:
    match = re.search(pattern, output)
    if not match:
        raise ValueError(f"Missing {label} in simulator output")
    return float(match.group(1))


def parse_neurosim_output(output: str) -> NeuroSimResult:
    return NeuroSimResult(
        latency_ns=_metric(
            output,
            rf"Chip layer-by-layer readLatency \(per image\) is:\s*{FLOAT}ns",
            "NeuroSim latency",
        ),
        dynamic_energy_pj=_metric(
            output,
            rf"Chip total readDynamicEnergy is:\s*{FLOAT}pJ",
            "NeuroSim energy",
        ),
        leakage_power_uw=_metric(
            output,
            rf"Chip total leakage Power is:\s*{FLOAT}uW",
            "NeuroSim leakage",
        ),
        area_um2=_metric(output, rf"ChipArea\s*:\s*{FLOAT}um\^2", "NeuroSim area"),
    )


def parse_booksim_output(output: str) -> BookSimResult:
    return BookSimResult(
        latency_cycles=_metric(output, rf"Time taken is\s*{FLOAT}\s*cycles", "BookSim latency"),
        total_power_w=_metric(output, rf"- Total Power:\s*{FLOAT}", "BookSim power"),
        leakage_power_w=_metric(output, rf"- Total leak Power:\s*{FLOAT}", "BookSim leakage"),
        area_m2=_metric(output, rf"- Total Area:\s*{FLOAT}", "BookSim area"),
    )


def run_neurosim(
    binary: Path,
    network_csv: Path,
    artifacts: LayerArtifacts,
    config: HardwareConfig,
    output_dir: Path,
) -> NeuroSimResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [str(binary), str(network_csv), "8", "8"]
    for weight_file, input_file in zip(artifacts.weight_files, artifacts.input_files, strict=True):
        command.extend((str(weight_file), str(input_file)))
    command.extend(
        (
            str(config.adc_bits),
            str(config.cell_bits),
            "VGG11_CIFAR10",
            str(config.sa_row),
            str(config.sa_col),
            str(config.pe),
            str(config.tile),
        )
    )
    output = _run(command, output_dir)
    (output_dir / "neurosim.log").write_text(output, encoding="utf-8")
    return parse_neurosim_output(output)


def run_booksim(binary: Path, root: Path, config: HardwareConfig, output_dir: Path) -> BookSimResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_template = root / "booksim2" / "src" / "examples" / "mesh88_lat"
    tech_file = root / "booksim2" / "src" / "power" / "techfile.txt"
    mesh = output_dir / "booksim.cfg"
    mesh_text = mesh_template.read_text(encoding="utf-8")
    mesh_text = re.sub(
        r"^tech_file\s*=.*;$",
        f"tech_file = {tech_file.resolve()};",
        mesh_text,
        flags=re.MULTILINE,
    )
    mesh.write_text(mesh_text, encoding="utf-8")
    network_size = max(2, min(8, config.tile // 2))
    command = [
        str(binary),
        str(mesh),
        str(network_size),
        "1",
        "0.001",
        "0.01",
        "128",
        "0",
        "1",
        "1",
        "1",
        "-1",
    ]
    output = _run(command, output_dir)
    (output_dir / "booksim.log").write_text(output, encoding="utf-8")
    return parse_booksim_output(output)


def result_dict(config: HardwareConfig, neurosim: NeuroSimResult, booksim: BookSimResult) -> dict:
    return {
        "config": asdict(config),
        "neurosim": asdict(neurosim),
        "booksim": asdict(booksim),
    }
