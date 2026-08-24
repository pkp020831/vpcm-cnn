from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs/crosssim-adc-fulltest"
OUTPUT = RESULTS / "adc_accuracy_comparison.png"
ADCS = (4, 6, 8)


def main() -> None:
    reports = [
        json.loads((RESULTS / f"adc{adc}" / "crosssim_results.json").read_text(encoding="utf-8"))
        for adc in ADCS
    ]
    accuracies = [report["crosssim_accuracy_mean"] for report in reports]
    digital_accuracy = reports[0]["digital_accuracy"]

    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    bars = axis.bar(ADCS, accuracies, width=1.1, color=("#b2182b", "#fdae61", "#1b7837"), edgecolor="black", linewidth=0.8)
    axis.axhline(digital_accuracy, color="#20242a", linewidth=1.5, linestyle="--", label=f"Digital baseline ({digital_accuracy:.2f}%)")
    axis.bar_label(bars, labels=[f"{accuracy:.2f}%" for accuracy in accuracies], padding=5, fontsize=12, fontweight="bold")
    axis.set_xticks(ADCS)
    axis.set_xlabel("ADC precision (bits)")
    axis.set_ylabel("Top-1 accuracy (%)")
    axis.set_ylim(0, 100)
    axis.set_title("ACIM Accuracy vs. ADC Precision\nCIFAR-10 Test Set (10,000 Images), Cell 7-bit, DAC 8-bit", fontweight="bold")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right")
    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
