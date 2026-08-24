#!/usr/bin/env python3
"""Generate a NeuroSim chip hierarchy and accumulation-flow diagram."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


BG = "#f5f1e8"
INK = "#172026"
BLUE = "#167d9a"
BLUE_LIGHT = "#d8edf1"
ORANGE = "#d86f32"
ORANGE_LIGHT = "#f8dfcf"
GREEN = "#3d7d62"
GREEN_LIGHT = "#dceadf"
GRAY = "#66747b"
GRAY_LIGHT = "#e5e7e4"
YELLOW = "#f1c75b"


def box(ax, x, y, w, h, title, subtitle="", *, fc="white", ec=INK,
        lw=1.4, size=9, title_color=INK, zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * (0.62 if subtitle else 0.5), title,
            ha="center", va="center", fontsize=size, fontweight="bold",
            color=title_color, zorder=zorder + 1)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.27, subtitle, ha="center", va="center",
                fontsize=size - 2, color=GRAY, zorder=zorder + 1)
    return patch


def panel(ax, x, y, w, h, title, *, ec=INK, fc="#ffffff", alpha=0.68):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        facecolor=fc, edgecolor=ec, linewidth=1.8, alpha=alpha, zorder=0,
    )
    ax.add_patch(patch)
    ax.text(x + 0.012, y + h - 0.022, title, ha="left", va="top",
            fontsize=12, fontweight="bold", color=ec)
    return patch


def arrow(ax, start, end, *, color=BLUE, lw=1.8, style="-|>",
          connection="arc3", linestyle="-", zorder=4):
    patch = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=12,
        color=color, linewidth=lw, linestyle=linestyle,
        connectionstyle=connection, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def bracket_label(ax, x, y, text, *, color=ORANGE):
    ax.text(x, y, text, ha="center", va="center", fontsize=8,
            fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec="none"), zorder=7)


def draw_subarray(ax, x, y, w, h):
    panel(ax, x, y, w, h, "SUBARRAY  |  analog MVM + local digital reduction",
          ec=GREEN, fc="#f8fbf8", alpha=1)

    # Analog row/input path.
    box(ax, x + 0.015, y + 0.128, 0.064, 0.070, "Input bits", "bit-serial", fc=BLUE_LIGHT, ec=BLUE, size=8)
    box(ax, x + 0.093, y + 0.128, 0.068, 0.070, "WL path", "decoder / switch", fc=GRAY_LIGHT, ec=GRAY, size=8)
    arrow(ax, (x + 0.079, y + 0.163), (x + 0.093, y + 0.163))

    # Crossbar matrix.
    gx, gy, gw, gh = x + 0.175, y + 0.119, 0.098, 0.088
    ax.add_patch(Rectangle((gx, gy), gw, gh, facecolor=GREEN_LIGHT,
                           edgecolor=GREEN, linewidth=1.5, zorder=2))
    for i in range(1, 6):
        ax.plot([gx, gx + gw], [gy + gh * i / 6] * 2, color="#8db49e", lw=0.55, zorder=3)
        ax.plot([gx + gw * i / 6] * 2, [gy, gy + gh], color="#8db49e", lw=0.55, zorder=3)
    ax.text(gx + gw / 2, gy + gh / 2, "Cell array\nI = G x V", ha="center", va="center",
            fontsize=9, fontweight="bold", color=GREEN, zorder=4)
    arrow(ax, (x + 0.161, y + 0.163), (gx, y + 0.163))

    box(ax, x + 0.287, y + 0.128, 0.063, 0.070, "Column MUX", "mux cycles", fc=GRAY_LIGHT, ec=GRAY, size=8)
    arrow(ax, (gx + gw, y + 0.163), (x + 0.287, y + 0.163), color=GREEN)

    # Digital row. The curved link represents column conversion before reduction.
    box(ax, x + 0.015, y + 0.025, 0.068, 0.070, "ADC / S.A.", "column outputs", fc=ORANGE_LIGHT, ec=ORANGE, size=8)
    box(ax, x + 0.102, y + 0.025, 0.068, 0.070, "Shift-add", "input / weight", fc=ORANGE_LIGHT, ec=ORANGE, size=8)
    box(ax, x + 0.189, y + 0.025, 0.073, 0.070, "Adder + DFF", "local partial sum", fc=ORANGE_LIGHT, ec=ORANGE, size=8)
    box(ax, x + 0.281, y + 0.025, 0.069, 0.070, "SA result", "partial sum", fc="#fff6d8", ec="#a77b16", size=8)
    arrow(ax, (x + 0.318, y + 0.128), (x + 0.049, y + 0.095), color=ORANGE,
          connection="arc3,rad=-0.34")
    arrow(ax, (x + 0.083, y + 0.060), (x + 0.102, y + 0.060), color=ORANGE)
    arrow(ax, (x + 0.170, y + 0.060), (x + 0.189, y + 0.060), color=ORANGE)
    arrow(ax, (x + 0.262, y + 0.060), (x + 0.281, y + 0.060), color=ORANGE)
    bracket_label(ax, x + 0.215, y + 0.010, "SUBARRAY ACCUMULATION")


def draw_reduction_chain(ax):
    panel(ax, 0.045, 0.075, 0.91, 0.265, "PARTIAL-SUM REDUCTION CHAIN  |  partition boundaries",
          ec=ORANGE, fc="#fffaf6", alpha=1)

    nodes = [
        (0.075, "Subarray\nresults", "parallel columns"),
        (0.255, "PE adder tree", "merge row-partitioned SAs"),
        (0.455, "Tile accumulation", "merge row-partitioned PEs"),
        (0.655, "Global accumulation", "merge row-partitioned Tiles"),
        (0.835, "Activation / Pool", "final feature-map value"),
    ]
    widths = [0.11, 0.13, 0.13, 0.135, 0.105]
    for (x, title, subtitle), width in zip(nodes, widths):
        box(ax, x, 0.165, width, 0.095, title, subtitle, fc=ORANGE_LIGHT,
            ec=ORANGE, size=9)

    for i in range(len(nodes) - 1):
        start_x = nodes[i][0] + widths[i]
        end_x = nodes[i + 1][0]
        arrow(ax, (start_x, 0.212), (end_x, 0.212), color=ORANGE, lw=2.6)

    labels = ["if >1 SA row", "if >1 PE row", "if >1 Tile row", "one output"]
    for i, label in enumerate(labels):
        start_x = nodes[i][0] + widths[i]
        end_x = nodes[i + 1][0]
        bracket_label(ax, (start_x + end_x) / 2, 0.274, label)

    ax.text(0.075, 0.115,
            "No row partition at a boundary -> that reduction stage is bypassed.  "
            "Column partitions produce different output channels; row partitions produce partial sums that must be added.",
            fontsize=8.5, color=GRAY, ha="left", va="center")


def draw_diagram(output: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(18, 11), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(0.055, 0.955, "NeuroSim Chip Hierarchy and Accumulation Flow",
             fontsize=23, fontweight="bold", color=INK, ha="left")
    fig.text(0.055, 0.925,
             "Data moves downward through buffers/interconnect; partial sums move right through hierarchical reduction.",
             fontsize=11, color=GRAY, ha="left")

    # Full-chip shell and replicated compute hierarchy.
    panel(ax, 0.045, 0.375, 0.91, 0.505, "CHIP  |  one mapped neural-network layer",
          ec=INK, fc="#fbfaf7", alpha=1)

    box(ax, 0.068, 0.775, 0.125, 0.065, "Global Buffer", "input + output feature maps", fc=BLUE_LIGHT, ec=BLUE)
    box(ax, 0.218, 0.775, 0.125, 0.065, "Global H-tree", "broadcast / collect", fc=BLUE_LIGHT, ec=BLUE)
    arrow(ax, (0.193, 0.807), (0.218, 0.807), color=BLUE, lw=2.2)

    # Tile cluster.
    panel(ax, 0.37, 0.725, 0.555, 0.125, "TILE ARRAY  |  tiles execute mapped partitions in parallel",
          ec=BLUE, fc="#f5fbfc", alpha=1)
    for r in range(2):
        for c in range(5):
            tx = 0.395 + c * 0.099
            ty = 0.748 + r * 0.036
            ax.add_patch(FancyBboxPatch((tx, ty), 0.074, 0.025,
                         boxstyle="round,pad=0.004", fc="#b9dde5", ec=BLUE, lw=0.9))
            ax.text(tx + 0.037, ty + 0.0125, "Tile", fontsize=7, ha="center", va="center", color=INK)
    arrow(ax, (0.343, 0.807), (0.37, 0.807), color=BLUE, lw=2.2)

    # Expanded tile and PE array.
    panel(ax, 0.068, 0.485, 0.255, 0.205, "ONE TILE  |  local data movement",
          ec=BLUE, fc="#f5fbfc", alpha=1)
    box(ax, 0.088, 0.595, 0.082, 0.055, "Tile buffers", "input / output", fc=BLUE_LIGHT, ec=BLUE)
    box(ax, 0.198, 0.595, 0.082, 0.055, "Tile H-tree", "to PE array", fc=BLUE_LIGHT, ec=BLUE)
    arrow(ax, (0.17, 0.622), (0.198, 0.622), color=BLUE)
    for r in range(2):
        for c in range(3):
            px = 0.096 + c * 0.064
            py = 0.515 + r * 0.038
            ax.add_patch(FancyBboxPatch((px, py), 0.046, 0.026,
                         boxstyle="round,pad=0.004", fc=GREEN_LIGHT, ec=GREEN, lw=0.9))
            ax.text(px + 0.023, py + 0.013, "PE", fontsize=7, ha="center", va="center", color=INK)
    arrow(ax, (0.239, 0.595), (0.239, 0.567), color=BLUE)

    # Expanded PE and SA array.
    panel(ax, 0.35, 0.485, 0.185, 0.205, "ONE PE  |  subarray cluster",
          ec=GREEN, fc="#f8fbf8", alpha=1)
    box(ax, 0.372, 0.595, 0.062, 0.052, "PE buffers", "I/O", fc=BLUE_LIGHT, ec=BLUE)
    box(ax, 0.452, 0.595, 0.059, 0.052, "Local bus", "fan-out", fc=BLUE_LIGHT, ec=BLUE)
    arrow(ax, (0.434, 0.621), (0.452, 0.621), color=BLUE)
    for r in range(2):
        for c in range(3):
            sx = 0.372 + c * 0.049
            sy = 0.515 + r * 0.038
            ax.add_patch(FancyBboxPatch((sx, sy), 0.038, 0.026,
                         boxstyle="round,pad=0.004", fc="#b9d8c5", ec=GREEN, lw=0.9))
            ax.text(sx + 0.019, sy + 0.013, "SA", fontsize=7, ha="center", va="center", color=INK)
    arrow(ax, (0.481, 0.595), (0.481, 0.567), color=BLUE)

    draw_subarray(ax, 0.558, 0.425, 0.367, 0.265)

    # Zoom connectors show containment, not runtime traffic.
    arrow(ax, (0.655, 0.725), (0.285, 0.69), color=GRAY, lw=1.0,
          style="->", connection="arc3,rad=0.12", linestyle="--", zorder=1)
    arrow(ax, (0.323, 0.57), (0.35, 0.57), color=GRAY, lw=1.0,
          style="->", linestyle="--", zorder=1)
    arrow(ax, (0.535, 0.54), (0.558, 0.54), color=GRAY, lw=1.0,
          style="->", linestyle="--", zorder=1)

    draw_reduction_chain(ax)

    # Runtime links from hierarchy to reduction stages.
    arrow(ax, (0.87, 0.425), (0.13, 0.34), color=ORANGE, lw=1.4,
          style="->", connection="arc3,rad=0.10", linestyle="--", zorder=1)
    arrow(ax, (0.44, 0.485), (0.32, 0.34), color=ORANGE, lw=1.4,
          style="->", connection="arc3,rad=-0.05", linestyle="--", zorder=1)
    arrow(ax, (0.22, 0.485), (0.52, 0.34), color=ORANGE, lw=1.4,
          style="->", connection="arc3,rad=0.05", linestyle="--", zorder=1)
    arrow(ax, (0.79, 0.725), (0.72, 0.34), color=ORANGE, lw=1.4,
          style="->", connection="arc3,rad=-0.06", linestyle="--", zorder=1)

    # Legend.
    ax.plot([0.06, 0.105], [0.035, 0.035], color=BLUE, lw=3)
    ax.text(0.112, 0.035, "data / interconnect", va="center", fontsize=8.5, color=GRAY)
    ax.plot([0.255, 0.30], [0.035, 0.035], color=ORANGE, lw=3)
    ax.text(0.307, 0.035, "partial-sum accumulation", va="center", fontsize=8.5, color=GRAY)
    ax.plot([0.495, 0.54], [0.035, 0.035], color=GRAY, lw=1.5, ls="--")
    ax.text(0.547, 0.035, "hierarchy expansion / stage mapping", va="center", fontsize=8.5, color=GRAY)
    ax.text(0.94, 0.035, "SA = SubArray", ha="right", va="center", fontsize=8.5, color=GRAY)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    if output.suffix.lower() != ".svg":
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("neurosim_chip_architecture.png"))
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    draw_diagram(args.output, args.dpi)
    print(args.output)
    if args.output.suffix.lower() != ".svg":
        print(args.output.with_suffix(".svg"))


if __name__ == "__main__":
    main()
