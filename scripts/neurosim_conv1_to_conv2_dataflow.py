#!/usr/bin/env python3
"""Draw the detailed NeuroSim Conv1-to-Conv2 hardware data path."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


BG = "#f4f1e9"
INK = "#162027"
MUTED = "#617078"
DATA = "#087f9c"
DATA_FILL = "#d9eef2"
ANALOG = "#397a5d"
ANALOG_FILL = "#dceadf"
ACCUM = "#d56327"
ACCUM_FILL = "#f8dfcf"
STORE = "#315b9a"
STORE_FILL = "#dce5f5"
GLOBAL = "#72529a"
GLOBAL_FILL = "#e9e0f3"
CONTROL = "#69767b"
CONTROL_FILL = "#e5e8e6"


def region(ax, x, y, w, h, title, color, fill, *, z=0, title_size=8.5):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.16,rounding_size=0.65",
        facecolor=fill, edgecolor=color, linewidth=1.25, alpha=0.42, zorder=z,
    )
    ax.add_patch(patch)
    ax.text(x + 0.65, y + h - 0.75, title, ha="left", va="top",
            fontsize=title_size, fontweight="bold", color=color, zorder=z + 1)


def node(ax, x, y, w, h, title, subtitle, color, fill, *, number=None, size=7.2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.10,rounding_size=0.45",
        facecolor=fill, edgecolor=color, linewidth=1.25, zorder=5,
    )
    ax.add_patch(patch)
    if number is not None:
        ax.text(x + 0.52, y + h - 0.48, str(number), ha="center", va="center",
                fontsize=6.3, fontweight="bold", color="white", zorder=7,
                bbox=dict(boxstyle="circle,pad=0.19", facecolor=color,
                          edgecolor="white", linewidth=0.55))
    ax.text(x + w / 2, y + h * 0.61, title, ha="center", va="center",
            fontsize=size, fontweight="bold", color=INK, zorder=6)
    ax.text(x + w / 2, y + h * 0.26, subtitle, ha="center", va="center",
            fontsize=size - 1.4, color=MUTED, zorder=6, linespacing=1.15)
    return patch


def flow(ax, start, end, *, color=DATA, label=None, rad=0, lw=1.8,
         style="-|>", dashed=False, z=8):
    arrow = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=11,
        linewidth=lw, color=color, zorder=z,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + (0.72 if abs(start[1] - end[1]) < 1 else 0), label,
                ha="center", va="center", fontsize=5.8, fontweight="bold",
                color=color, zorder=10,
                bbox=dict(boxstyle="round,pad=0.16", facecolor=BG,
                          edgecolor="none", alpha=0.96))
    return arrow


def crossbar(ax, x, y, w, h, number):
    node(ax, x, y, w, h, "Cell array VMM", "BL current = sum(G x V)",
         ANALOG, ANALOG_FILL, number=number)
    for i in range(1, 6):
        ax.plot([x + 0.45, x + w - 0.45], [y + h * i / 6] * 2,
                color="#8caf9a", lw=0.35, zorder=6)
        ax.plot([x + w * i / 6] * 2, [y + 0.35, y + h - 0.35],
                color="#8caf9a", lw=0.35, zorder=6)


def fan_in(ax, x, y, labels, target, *, color=ACCUM):
    """Show representative parallel producers feeding a reduction node."""
    for index, label in enumerate(labels):
        yy = y + index * 2.2
        box = FancyBboxPatch(
            (x, yy), 3.15, 1.42,
            boxstyle="round,pad=0.08,rounding_size=0.25",
            facecolor="#fff8f3", edgecolor=color, linewidth=0.75, zorder=4,
        )
        ax.add_patch(box)
        ax.text(x + 1.575, yy + 0.71, label, ha="center", va="center",
                fontsize=5.2, color=color, zorder=5)
        flow(ax, (x, yy + 0.71), target, color=color, lw=0.75,
             style="->", dashed=True, z=3)


def legend_item(ax, x, y, color, title, description):
    ax.plot([x, x + 3.0], [y, y], color=color, lw=2.6, zorder=10)
    ax.text(x + 3.45, y + 0.12, title, fontsize=6.6, fontweight="bold",
            color=INK, va="center")
    ax.text(x + 3.45, y - 0.72, description, fontsize=5.5, color=MUTED, va="center")


def draw(output: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(28, 17), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    fig.text(0.047, 0.958, "NeuroSim Detailed Inter-Layer Data Path",
             fontsize=26, fontweight="bold", color=INK, ha="left")
    fig.text(0.047, 0.932,
             "Conv1 VMM -> hierarchical partial-sum reduction -> ReLU/MaxPool -> layer buffer -> Conv2 word line",
             fontsize=11.5, color=MUTED, ha="left")

    # Physical hierarchy. Regions deliberately overlap only at containment boundaries.
    region(ax, 1.5, 10.0, 97.0, 82.5, "CHIP", INK, "#ffffff", z=0, title_size=10)
    region(ax, 18.0, 55.2, 80.0, 32.3, "CONV1 TILE", DATA, "#e9f6f8", z=1)
    region(ax, 32.5, 58.2, 65.0, 26.2, "CONV1 PE", ANALOG, "#edf6ef", z=2)
    region(ax, 48.0, 61.0, 49.0, 20.5, "CONV1 SUBARRAY", ANALOG, "#f5faf6", z=3)
    region(ax, 18.0, 13.5, 80.0, 21.5, "CONV2 TILE", DATA, "#e9f6f8", z=1)
    region(ax, 47.5, 16.1, 50.0, 16.3, "CONV2 PE", ANALOG, "#edf6ef", z=2)
    region(ax, 75.5, 18.5, 21.5, 11.2, "CONV2 SUBARRAY ROW PATH", ANALOG, "#f5faf6", z=3)
    region(ax, 76.7, 38.5, 18.3, 12.7, "CONV1 PE OUTPUT PATH", ANALOG, "#edf6ef", z=1, title_size=7.2)
    region(ax, 52.7, 38.5, 23.5, 12.7, "CONV1 TILE OUTPUT PATH", DATA, "#e9f6f8", z=1, title_size=7.2)
    region(ax, 8.2, 38.5, 44.0, 12.7, "CHIP GLOBAL POST-PROCESSING", GLOBAL, "#f2edf7", z=1, title_size=7.2)

    # Row 1: input distribution and Conv1 analog/digital conversion.
    top_y, top_h = 68.2, 8.0
    top_nodes = [
        (3.4, 7.0, "Input Global\nBuffer", "Conv1 input feature map", STORE, STORE_FILL),
        (11.7, 5.8, "Global H-tree", "broadcast to Conv1 tiles", DATA, DATA_FILL),
        (19.6, 6.6, "Tile Input\nBuffer", "stage reusable windows", STORE, STORE_FILL),
        (27.7, 5.8, "Tile H-tree", "broadcast to PEs", DATA, DATA_FILL),
        (34.4, 6.4, "PE Input\nBuffer", "bit-plane staging", STORE, STORE_FILL),
        (42.1, 5.5, "Input Bus", "horizontal fan-out", DATA, DATA_FILL),
        (49.2, 5.8, "WL Path", "decoder / switch matrix", CONTROL, CONTROL_FILL),
        (56.2, 7.2, "", "", ANALOG, ANALOG_FILL),
        (64.8, 5.6, "Column MUX", "share sensing circuits", CONTROL, CONTROL_FILL),
        (71.6, 6.2, "ADC / MLSA", "analog-to-digital levels", ACCUM, ACCUM_FILL),
        (79.0, 6.2, "Shift-add", "input bits + weight cells", ACCUM, ACCUM_FILL),
        (86.4, 6.4, "SA local\nAdder + DFF", "local dot-product part", ACCUM, ACCUM_FILL),
        (94.0, 3.1, "SA PS", "x N", ACCUM, "#fff4dc"),
    ]
    numbers = list(range(1, 14))
    patches = []
    for index, (x, w, title, subtitle, color, fill) in enumerate(top_nodes):
        if index == 7:
            crossbar(ax, x, top_y, w, top_h, numbers[index])
        else:
            node(ax, x, top_y, w, top_h, title, subtitle, color, fill,
                 number=numbers[index], size=6.8)
        patches.append((x, w))
    for index in range(len(patches) - 1):
        x, w = patches[index]
        nx, _ = patches[index + 1]
        color = DATA if index < 6 else (ANALOG if index < 9 else ACCUM)
        flow(ax, (x + w, top_y + top_h / 2), (nx, top_y + top_h / 2), color=color)

    # Turn from the subarray result down into PE-level reduction.
    flow(ax, (97.1, 68.2), (92.2, 49.2), color=ACCUM, lw=2.2,
         label="row-partition partial sums", rad=-0.18)

    # Row 2 runs right-to-left: reduction and Conv1 post-processing.
    mid_y, mid_h = 41.2, 8.0
    mid_nodes = [
        (86.6, 7.0, "PE Adder\nTree", "merge SA row partitions", ACCUM, ACCUM_FILL, 14),
        (78.4, 6.2, "PE Output\nBuffer", "hold completed PE sums", STORE, STORE_FILL, 15),
        (70.7, 6.0, "Tile H-tree", "collect PE outputs", DATA, DATA_FILL, 16),
        (62.2, 6.7, "Tile\nAccumulation", "merge PE row partitions", ACCUM, ACCUM_FILL, 17),
        (54.2, 6.0, "Tile Output\nBuffer", "stage tile results", STORE, STORE_FILL, 18),
        (45.7, 6.5, "Global H-tree", "collect Conv1 tile results", DATA, DATA_FILL, 19),
        (36.8, 7.0, "Global\nAccumulation", "merge Tile row partitions", GLOBAL, GLOBAL_FILL, 20),
        (29.0, 5.7, "ReLU", "global activation unit", GLOBAL, GLOBAL_FILL, 21),
        (20.6, 6.2, "2x2 MaxPool", "32x32x64 -> 16x16x64", GLOBAL, GLOBAL_FILL, 22),
        (9.8, 8.0, "Inter-layer\nGlobal Buffer", "store pooled Conv1 output", STORE, STORE_FILL, 23),
    ]
    for x, w, title, subtitle, color, fill, number in mid_nodes:
        node(ax, x, mid_y, w, mid_h, title, subtitle, color, fill,
             number=number, size=6.8)
    for index in range(len(mid_nodes) - 1):
        x, _, _, _, color, _, _ = mid_nodes[index]
        nx, nw, *_ = mid_nodes[index + 1]
        start = (x, mid_y + mid_h / 2)
        end = (nx + nw, mid_y + mid_h / 2)
        flow_color = ACCUM if index in (0, 2, 5) else (DATA if index in (1, 4) else GLOBAL)
        flow(ax, start, end, color=flow_color)

    # Representative fan-in sources at each reduction boundary.
    fan_in(ax, 95.0, 52.0, ["SA PS 1", "SA PS 2", "SA PS ..."],
           (93.6, 45.2), color=ACCUM)
    fan_in(ax, 76.8, 51.8, ["PE PS 1", "PE PS 2", "PE PS ..."],
           (68.9, 45.2), color=ACCUM)
    fan_in(ax, 51.9, 51.8, ["Tile PS 1", "Tile PS 2", "Tile PS ..."],
           (43.8, 45.2), color=ACCUM)

    # Turn from inter-layer storage to Conv2 distribution.
    flow(ax, (13.8, 41.2), (14.45, 28.1), color=STORE, lw=2.2,
         label="read quantized pooled activations", rad=-0.04)

    # Row 3: Conv2 distribution, ending at WL and selected rows.
    low_y, low_h = 20.3, 7.8
    low_nodes = [
        (11.4, 6.1, "Global H-tree", "broadcast to Conv2 tiles", DATA, DATA_FILL, 24),
        (19.6, 7.0, "Conv2 Tile\nInput Buffer", "16x16x64 feature map", STORE, STORE_FILL, 25),
        (28.5, 6.1, "Tile H-tree", "broadcast to Conv2 PEs", DATA, DATA_FILL, 26),
        (36.5, 7.0, "Conv2 PE\nInput Buffer", "extract 3x3x64 windows", STORE, STORE_FILL, 27),
        (45.5, 6.2, "Input Bus", "horizontal bit fan-out", DATA, DATA_FILL, 28),
        (53.6, 6.5, "Input Bit\nSequencer", "activation bit planes", CONTROL, CONTROL_FILL, 29),
        (62.1, 6.8, "WL Decoder /\nSwitch Matrix", "select Conv2 rows", CONTROL, CONTROL_FILL, 30),
        (71.0, 5.9, "WL RC", "drive selected word lines", ANALOG, ANALOG_FILL, 31),
        (78.8, 8.3, "Conv2 WLs", "576 logical inputs\n(3x3x64)", ANALOG, ANALOG_FILL, 32),
        (89.0, 6.8, "Conv2 Cell\nArray", "next VMM begins", ANALOG, ANALOG_FILL, 33),
    ]
    for x, w, title, subtitle, color, fill, number in low_nodes:
        node(ax, x, low_y, w, low_h, title, subtitle, color, fill,
             number=number, size=6.7)
    for index in range(len(low_nodes) - 1):
        x, w, *_ = low_nodes[index]
        nx, *_ = low_nodes[index + 1]
        flow(ax, (x + w, low_y + low_h / 2), (nx, low_y + low_h / 2),
             color=DATA if index < 5 else (CONTROL if index < 7 else ANALOG))

    # Explicit storage/read distinction at the layer boundary.
    ax.text(4.0, 35.8,
            "Layer boundary\nConv1 producer and Conv2 consumer are decoupled here",
            ha="left", va="center", fontsize=6.3, fontweight="bold", color=STORE,
            bbox=dict(boxstyle="round,pad=0.38", facecolor="#f5f8fd",
                      edgecolor=STORE, linewidth=0.8))

    # Bottom role legend and implementation note.
    legend_item(ax, 3.0, 7.1, STORE, "BUFFER", "stores data; absorbs timing and width mismatch")
    legend_item(ax, 27.0, 7.1, DATA, "H-TREE", "long-range hierarchical broadcast / collect")
    legend_item(ax, 51.0, 7.1, DATA, "BUS", "short-range PE-local shared transport")
    legend_item(ax, 71.0, 7.1, ACCUM, "ACCUMULATION", "adds only row-partition partial sums")
    ax.text(3.0, 3.2,
            "Dashed fan-in arrows show representative parallel producers. Column partitions usually represent different output channels and do not merge; "
            "row partitions represent the same dot product and must merge.",
            fontsize=5.8, color=MUTED, ha="left", va="center")
    ax.text(97.0, 3.2,
            "NeuroSim estimates this path's PPA; layer CSV inputs are captured independently from PyTorch.",
            fontsize=5.8, color=MUTED, ha="right", va="center")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    if output.suffix.lower() != ".svg":
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("neurosim_conv1_to_conv2_dataflow.png"))
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args()
    draw(args.output, args.dpi)
    print(args.output)
    if args.output.suffix.lower() != ".svg":
        print(args.output.with_suffix(".svg"))


if __name__ == "__main__":
    main()
