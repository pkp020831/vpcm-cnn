from __future__ import annotations

import os
import platform
import re
import json
import subprocess
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

from .booksim_mapping import read_floorplan, write_booksim_records
from .layers import LayerArtifacts


FLOAT = r"([-+0-9.eE]+)"

# Fixed BookSim baseline used for all DSE candidates. These values match the
# legacy NavCim homogeneous path and keep the router power model invariant.
BOOKSIM_FLIT_BITS = 128


@dataclass(frozen=True)
class HardwareConfig:
    sa_row: int = 128
    sa_col: int = 128
    pe: int = 4
    tile: int = 8
    adc_bits: int = 5
    cell_bits: int = 2
    num_col_muxed: int = 8

    @property
    def key(self) -> str:
        return (
            f"sa{self.sa_row}x{self.sa_col}_pe{self.pe}_tile{self.tile}_"
            f"adc{self.adc_bits}_cell{self.cell_bits}_mux{self.num_col_muxed}"
        )


def prescribed_hardware_sweep_groups() -> dict[str, list[HardwareConfig]]:
    return {
        "adc_cell": [
            HardwareConfig(128, 128, 4, 8, adc, cell, 8)
            for adc, cell in product((4, 6, 8), (1, 2, 4))
        ],
        "sa_pe_tile": [
            HardwareConfig(sa, sa, pe, tile, 8, 2, 8)
            for sa, pe, tile in product((64, 128, 256), (2, 4, 8), (8, 16, 32))
        ],
        "num_col_muxed": [
            HardwareConfig(128, 128, 4, 8, 8, 2, mux)
            for mux in (4, 8, 16)
        ],
    }


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


@dataclass(frozen=True)
class NeuroSimNoCConfig:
    tile_width_m: float
    min_repeater_distance_m: float
    bus_width_bits: int
    clock_period_ns: float
    unit_repeater_latency_s_per_m: float

    @property
    def clock_hz(self) -> float:
        return 1e9 / self.clock_period_ns

    @property
    def link_latency_cycles(self) -> int:
        return max(1, round(self.unit_repeater_latency_s_per_m * self.tile_width_m * self.clock_hz))


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


def parse_neurosim_noc_config(output: str) -> NeuroSimNoCConfig:
    return NeuroSimNoCConfig(
        tile_width_m=_metric(output, rf"Tilewidth\s*:\s*{FLOAT}m", "NeuroSim tile width"),
        min_repeater_distance_m=_metric(output, rf"minDist\s*{FLOAT}m", "NeuroSim repeater distance"),
        bus_width_bits=round(_metric(output, rf"busWidth\s*{FLOAT}", "NeuroSim bus width")),
        clock_period_ns=_metric(output, rf"Chip clock period is:\s*{FLOAT}\s*ns", "NeuroSim clock period"),
        unit_repeater_latency_s_per_m=_metric(output, rf"NoC unitLatencyRep:\s*{FLOAT}", "NeuroSim repeater latency"),
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
            str(config.num_col_muxed),
        )
    )
    floorplan = output_dir / "neurosim_floorplan.csv"
    environment = os.environ.copy()
    environment["NAVCIM_NEUROSIM_FLOORPLAN"] = str(floorplan)
    output = _run(command, output_dir, env=environment)
    (output_dir / "neurosim.log").write_text(output, encoding="utf-8")
    if not floorplan.is_file():
        raise RuntimeError("NeuroSim did not produce its floorplan")
    return parse_neurosim_output(output)


def run_booksim(
    binary: Path,
    root: Path,
    config: HardwareConfig,
    output_dir: Path,
    artifacts: LayerArtifacts,
    neurosim_dir: Path | None = None,
) -> BookSimResult:
    results = run_booksim_layers(binary, root, config, output_dir, artifacts, neurosim_dir)
    return BookSimResult(
        latency_cycles=sum(result.latency_cycles for _, result in results),
        total_power_w=max(result.total_power_w for _, result in results),
        leakage_power_w=max(result.leakage_power_w for _, result in results),
        area_m2=max(result.area_m2 for _, result in results),
    )


def run_booksim_layers(
    binary: Path,
    root: Path,
    config: HardwareConfig,
    output_dir: Path,
    artifacts: LayerArtifacts,
    neurosim_dir: Path | None = None,
) -> tuple[tuple[int, BookSimResult], ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    neurosim_dir = (neurosim_dir or output_dir).resolve()
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
    output = (neurosim_dir / "neurosim.log").read_text(encoding="utf-8")
    noc = parse_neurosim_noc_config(output)
    floorplan = read_floorplan(neurosim_dir / "neurosim_floorplan.csv")
    cluster, element = write_booksim_records(
        floorplan, artifacts.records, output_dir, BOOKSIM_FLIT_BITS,
        1e9 / parse_neurosim_output(output).latency_ns, noc.clock_hz,
    )
    mesh_width = floorplan[0].mesh_rows
    results: list[tuple[int, BookSimResult]] = []
    outputs: list[str] = []
    for layer in floorplan[1:]:
        command = [
            str(binary), str(mesh), str(mesh_width), "1", str(noc.link_latency_cycles), str(noc.link_latency_cycles), str(cluster), str(element),
            str(noc.tile_width_m), str(noc.tile_width_m), str(BOOKSIM_FLIT_BITS), str(BOOKSIM_FLIT_BITS), "0", "unused", str(layer.layer + 1), "1",
        ]
        output = _run(command, output_dir)
        outputs.append(output)
        results.append((layer.layer + 1, parse_booksim_output(output)))
    (output_dir / "booksim.log").write_text("\n\n".join(outputs), encoding="utf-8")
    if not results:
        raise ValueError("VGG11 requires at least two layers for BookSim traffic")
    (output_dir / "booksim_layers.json").write_text(json.dumps([
        {"layer": layer, **asdict(result)} for layer, result in results
    ], indent=2, sort_keys=True), encoding="utf-8")
    return tuple(results)


def result_dict(config: HardwareConfig, neurosim: NeuroSimResult, booksim: BookSimResult) -> dict:
    return {
        "config": asdict(config),
        "neurosim": asdict(neurosim),
        "booksim": asdict(booksim),
    }
