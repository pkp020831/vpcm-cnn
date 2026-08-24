from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs/crosssim-adc-fulltest"
OUTPUT = RESULTS / "adc_accuracy_heatmap.png"
ADCS = (4, 6, 8)


def main() -> None:
    reports = [
        json.loads((RESULTS / f"adc{adc}" / "crosssim_results.json").read_text(encoding="utf-8"))
        for adc in ADCS
    ]
    values = np.array([[report["crosssim_accuracy_mean"] for report in reports]])
    digital_accuracy = reports[0]["digital_accuracy"]
    cmap = LinearSegmentedColormap.from_list(
        "accuracy", ["#b2182b", "#fdae61", "#fee08b", "#66bd63", "#006837"]
    )

    figure, axis = plt.subplots(figsize=(8.2, 3.8), constrained_layout=True)
    image = axis.imshow(values, cmap=cmap, vmin=10, vmax=93, aspect="auto")
    for column, accuracy in enumerate(values[0]):
        axis.text(column, 0, f"{accuracy:.2f}%", ha="center", va="center", fontsize=16, fontweight="bold")
    axis.set_xticks(range(len(ADCS)), ADCS)
    axis.set_yticks([0], ["7"])
    axis.set_xlabel("ADC bits", fontsize=13)
    axis.set_ylabel("Cell bits", fontsize=13)
    axis.set_title(
        f"ACIM Accuracy: ADC Precision (CIFAR-10 Test, 10,000 Images)\nDigital baseline: {digital_accuracy:.2f}%",
        fontsize=14,
        fontweight="bold",
    )
    colorbar = figure.colorbar(image, ax=axis, label="Accuracy (%)")
    colorbar.set_ticks([10, 30, 50, 70, 90])
    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
