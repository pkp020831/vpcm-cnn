#!/usr/bin/env python3
"""Generate tables and plots for the prescribed NeuroSim hardware sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


METRICS = (
    ("latency_ns", "Latency", "ms", 1e-6),
    ("dynamic_energy_pj", "Dynamic energy", "uJ", 1e-6),
    ("area_um2", "Area", "mm2", 1e-6),
)


def config_key(item: dict) -> tuple[int, ...]:
    config = item["config"]
    return (
        config["sa_row"], config["sa_col"], config["pe"], config["tile"],
        config["adc_bits"], config["cell_bits"], config["num_col_muxed"],
    )


def label(item: dict) -> str:
    config = item["config"]
    return f"SA{config['sa_row']}-PE{config['pe']}-T{config['tile']}"


def pareto_front(items: list[dict]) -> list[dict]:
    keys = ("latency_ns", "dynamic_energy_pj", "area_um2")
    front = []
    for candidate in items:
        values = candidate["neurosim"]
        dominated = any(
            all(other["neurosim"][key] <= values[key] for key in keys)
            and any(other["neurosim"][key] < values[key] for key in keys)
            for other in items
            if other is not candidate
        )
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda item: item["neurosim"]["latency_ns"])


def normalized_geomean(item: dict, baseline: dict) -> float:
    ratios = [item["neurosim"][key] / baseline["neurosim"][key] for key, _, _, _ in METRICS]
    return math.prod(ratios) ** (1 / len(ratios))


def write_csv(path: Path, items: list[dict], baseline: dict) -> None:
    fields = [
        "sa", "pe", "tile", "adc_bits", "cell_bits", "num_col_muxed",
        "latency_ms", "dynamic_energy_uJ", "area_mm2",
        "latency_vs_baseline", "energy_vs_baseline", "area_vs_baseline",
        "normalized_ppa_geomean",
    ]
    with path.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in items:
            config = item["config"]
            metrics = item["neurosim"]
            writer.writerow({
                "sa": config["sa_row"], "pe": config["pe"], "tile": config["tile"],
                "adc_bits": config["adc_bits"], "cell_bits": config["cell_bits"],
                "num_col_muxed": config["num_col_muxed"],
                "latency_ms": metrics["latency_ns"] * 1e-6,
                "dynamic_energy_uJ": metrics["dynamic_energy_pj"] * 1e-6,
                "area_mm2": metrics["area_um2"] * 1e-6,
                "latency_vs_baseline": metrics["latency_ns"] / baseline["neurosim"]["latency_ns"],
                "energy_vs_baseline": metrics["dynamic_energy_pj"] / baseline["neurosim"]["dynamic_energy_pj"],
                "area_vs_baseline": metrics["area_um2"] / baseline["neurosim"]["area_um2"],
                "normalized_ppa_geomean": normalized_geomean(item, baseline),
            })


def plot_adc_cell(items: list[dict], baseline: dict, output: Path) -> None:
    adcs, cells = (4, 6, 8), (1, 2, 4)
    lookup = {(item["config"]["adc_bits"], item["config"]["cell_bits"]): item for item in items}
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)
    for axis, (key, title, _, _) in zip(axes, METRICS):
        values = [[lookup[(adc, cell)]["neurosim"][key] / baseline["neurosim"][key] for cell in cells] for adc in adcs]
        image = axis.imshow(values, cmap="RdYlGn_r", vmin=min(0.25, min(map(min, values))), vmax=max(map(max, values)))
        for row, values_row in enumerate(values):
            for column, value in enumerate(values_row):
                axis.text(column, row, f"{value:.2f}x", ha="center", va="center", fontsize=10, fontweight="bold")
        axis.set_title(f"{title} / baseline")
        axis.set_xticks(range(len(cells)), cells)
        axis.set_yticks(range(len(adcs)), adcs)
        axis.set_xlabel("Cell bits")
        axis.set_ylabel("ADC bits")
        figure.colorbar(image, ax=axis, shrink=0.78, label="Normalized value (lower is better)")
    figure.suptitle("ADC x Cell Precision: NeuroSim PPA Relative to ADC8 / Cell2", fontsize=14, fontweight="bold")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_architecture(items: list[dict], output: Path) -> None:
    colors = {64: "#277da1", 128: "#f8961e", 256: "#43aa8b"}
    markers = {8: "o", 16: "s", 32: "^"}
    front = pareto_front(items)
    areas = [item["neurosim"]["area_um2"] * 1e-6 for item in items]
    minimum, maximum = min(areas), max(areas)

    def bubble_size(area: float) -> float:
        return 70 + 330 * (math.log(area) - math.log(minimum)) / (math.log(maximum) - math.log(minimum))

    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.8), sharex=True, sharey=True, constrained_layout=True)
    for axis, pe in zip(axes, (2, 4, 8)):
        panel = [item for item in items if item["config"]["pe"] == pe]
        for item in panel:
            config, metrics = item["config"], item["neurosim"]
            area = metrics["area_um2"] * 1e-6
            is_pareto = item in front
            axis.scatter(
                metrics["latency_ns"] * 1e-6, metrics["dynamic_energy_pj"] * 1e-6,
                s=bubble_size(area), color=colors[config["sa_row"]], marker=markers[config["tile"]],
                edgecolor="#c1121f" if is_pareto else "black",
                linewidth=2.3 if is_pareto else 0.55, alpha=0.82,
            )
            axis.annotate(
                f"SA{config['sa_row']}/T{config['tile']}\n{area:.0f} mm2",
                (metrics["latency_ns"] * 1e-6, metrics["dynamic_energy_pj"] * 1e-6),
                xytext=(4, 5), textcoords="offset points", fontsize=7,
            )
        axis.set_title(f"PE = {pe}", fontweight="bold")
        axis.set_xlabel("Latency (ms), lower is better")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Dynamic energy (uJ), lower is better")

    legend_handles = []
    for sa, color in colors.items():
        legend_handles.append(axes[0].scatter([], [], s=80, color=color, label=f"SA {sa}"))
    for tile, marker in markers.items():
        legend_handles.append(axes[0].scatter([], [], s=80, color="#888888", marker=marker, label=f"Tile {tile}"))
    area_levels = (min(areas), sorted(areas)[len(areas) // 2], max(areas))
    for area in area_levels:
        legend_handles.append(axes[0].scatter([], [], s=bubble_size(area), facecolors="none", edgecolors="black", label=f"Area {area:.0f} mm2"))
    legend_handles.append(axes[0].scatter([], [], s=100, facecolors="none", edgecolors="#c1121f", linewidth=2.3, label="3-metric Pareto"))
    figure.legend(handles=legend_handles, loc="outside lower center", ncol=5, fontsize=9)
    figure.suptitle("SA x PE x Tile Trade-off: PE Panels with Explicit Area Encoding", fontsize=15, fontweight="bold")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_mux(items: list[dict], baseline: dict, output: Path) -> None:
    items = sorted(items, key=lambda item: item["config"]["num_col_muxed"])
    muxes = [item["config"]["num_col_muxed"] for item in items]
    figure, axis = plt.subplots(figsize=(8.8, 5.1), constrained_layout=True)
    width = 0.23
    colors = ("#277da1", "#f8961e", "#43aa8b")
    for index, (key, title, _, _) in enumerate(METRICS):
        ratios = [item["neurosim"][key] / baseline["neurosim"][key] for item in items]
        positions = [position + (index - 1) * width for position in range(len(items))]
        bars = axis.bar(positions, ratios, width, label=title, color=colors[index])
        axis.bar_label(bars, labels=[f"{value:.2f}x" for value in ratios], padding=3, fontsize=9)
    axis.axhline(1, color="black", linewidth=0.9, linestyle="--")
    axis.set_xticks(range(len(muxes)), muxes)
    axis.set_xlabel("Columns sharing one ADC / sense amplifier")
    axis.set_ylabel("Relative to numColMuxed=8 (lower is better)")
    axis.set_title("Column Multiplexing Trade-off")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def metric_row(item: dict, baseline: dict, name: str) -> list[str]:
    metrics = item["neurosim"]
    return [
        name,
        f"{metrics['latency_ns'] * 1e-6:.3f}",
        f"{metrics['dynamic_energy_pj'] * 1e-6:.3f}",
        f"{metrics['area_um2'] * 1e-6:.3f}",
        f"{normalized_geomean(item, baseline):.3f}x",
    ]


def make_report(groups: dict[str, list[dict]], baseline: dict, output: Path) -> None:
    adc_rows = []
    for item in sorted(groups["adc_cell"], key=lambda value: (value["config"]["adc_bits"], value["config"]["cell_bits"])):
        config, metrics = item["config"], item["neurosim"]
        adc_rows.append([
            str(config["adc_bits"]), str(config["cell_bits"]),
            f"{metrics['latency_ns'] * 1e-6:.3f}", f"{metrics['dynamic_energy_pj'] * 1e-6:.3f}",
            f"{metrics['area_um2'] * 1e-6:.3f}", f"{normalized_geomean(item, baseline):.3f}x",
        ])

    architecture = groups["sa_pe_tile"]
    selectors = {
        "Baseline": baseline,
        "Minimum latency": min(architecture, key=lambda item: item["neurosim"]["latency_ns"]),
        "Minimum energy": min(architecture, key=lambda item: item["neurosim"]["dynamic_energy_pj"]),
        "Minimum area": min(architecture, key=lambda item: item["neurosim"]["area_um2"]),
        "Best normalized PPA": min(architecture, key=lambda item: normalized_geomean(item, baseline)),
    }
    architecture_rows = []
    seen = set()
    for description, item in selectors.items():
        key = config_key(item)
        if key in seen:
            continue
        seen.add(key)
        architecture_rows.append(metric_row(item, baseline, f"{description}: {label(item)}"))

    pareto_rows = [metric_row(item, baseline, label(item)) for item in pareto_front(architecture)]
    mux_rows = []
    for item in sorted(groups["num_col_muxed"], key=lambda value: value["config"]["num_col_muxed"]):
        mux_rows.append(metric_row(item, baseline, f"Mux {item['config']['num_col_muxed']}"))

    text = f"""# NeuroSim Hardware Sweep Summary

## Sweep Scope

| Group | Full-factorial range | Conditions | Other parameters fixed |
|---|---|---:|---|
| ADC x Cell | ADC `4, 6, 8`; Cell `1, 2, 4` | 9 | SA=128, PE=4, Tile=8, Mux=8 |
| SA x PE x Tile | SA `64, 128, 256`; PE `2, 4, 8`; Tile `8, 16, 32` | 27 | ADC=8, Cell=2, Mux=8 |
| Column mux | Mux `4, 8, 16` | 3 | ADC=8, Cell=2, SA=128, PE=4, Tile=8 |

The shared baseline is `ADC=8, Cell=2, SA=128, PE=4, Tile=8, Mux=8`. Group overlap reduces 39 logical conditions to 37 unique simulator runs. Lower is better for all reported metrics.

## ADC x Cell Results

{markdown_table(['ADC', 'Cell', 'Latency (ms)', 'Energy (uJ)', 'Area (mm2)', 'Normalized PPA'], adc_rows)}

![ADC and cell heatmaps](adc_cell_heatmap.png)

Increasing ADC precision consistently raises all three PPA metrics in this model. Increasing cell bits lowers PPA because an 8-bit synapse requires fewer physical columns and less accumulation when more bits are stored per cell. This is a PPA-only conclusion; conductance variability and accuracy must be evaluated separately.

## SA x PE x Tile Results

{markdown_table(['Selection', 'Latency (ms)', 'Energy (uJ)', 'Area (mm2)', 'Normalized PPA'], architecture_rows)}

### Three-Metric Pareto Front

{markdown_table(['Configuration', 'Latency (ms)', 'Energy (uJ)', 'Area (mm2)', 'Normalized PPA'], pareto_rows)}

![Architecture trade-off](architecture_pareto.png)

Tile=32 gives the lowest latency but incurs a very large area penalty. SA=256 with Tile=8 minimizes energy and area, while PE=2 generally favors latency and PE=8 favors energy/area. `SA256-PE2-T8` is the strongest balanced architecture point under equal weighting of normalized latency, energy, and area.

## Column Multiplexing Results

{markdown_table(['Configuration', 'Latency (ms)', 'Energy (uJ)', 'Area (mm2)', 'Normalized PPA'], mux_rows)}

![Column mux comparison](mux_tradeoff.png)

Mux=4 provides only a small latency improvement but nearly doubles area. Mux=16 roughly halves area while changing latency by less than 1%, at a small energy penalty. Therefore Mux=16 is attractive when area is important; Mux=8 remains the energy-oriented baseline.

## Notes

- `Normalized PPA` is the geometric mean of latency, dynamic energy, and area, each divided by the shared baseline. It is a compact equal-weight summary, not an accuracy-aware objective.
- NeuroSim leakage output was excluded because the parsed values are physically implausible (`10^11` to `10^12 uW`) and require separate parser/model validation.
- BookSim metrics are constant within the ADC/Cell and mux groups in the current integration, so these figures focus on NeuroSim PPA.
"""
    output.write_text(text, encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.results.read_text(encoding="utf-8"))
    groups = report["sweep_groups"]
    baseline = next(
        item for item in groups["adc_cell"]
        if config_key(item) == (128, 128, 4, 8, 8, 2, 8)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, items in groups.items():
        write_csv(args.output_dir / f"{name}.csv", items, baseline)
    plot_adc_cell(groups["adc_cell"], baseline, args.output_dir / "adc_cell_heatmap.png")
    plot_architecture(groups["sa_pe_tile"], args.output_dir / "architecture_pareto.png")
    plot_mux(groups["num_col_muxed"], baseline, args.output_dir / "mux_tradeoff.png")
    make_report(groups, baseline, args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
