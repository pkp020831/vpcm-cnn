from __future__ import annotations

import csv
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .simulators import BOOKSIM_FLIT_BITS, _run, build_booksim, parse_booksim_output
from .booksim.features import FEATURES, TARGETS, TARGET_FEATURES, sample_features


def _sample(seed: int) -> dict[str, float | int]:
    return sample_features(seed).__dict__


def collect_booksim_dataset(root: Path, output: Path, samples: int, workers: int, seed: int) -> dict:
    if samples < 1:
        raise ValueError("samples must be positive")
    binary = build_booksim(root, workers)
    template = root / "booksim2" / "src" / "examples" / "mesh88_lat"
    tech_file = root / "booksim2" / "src" / "power" / "techfile.txt"
    config = (output.parent / "booksim_fixed_baseline.cfg").resolve()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        template.read_text(encoding="utf-8").replace(
            "tech_file = /root/navcim/booksim2/src/power/techfile.txt;",
            f"tech_file = {tech_file.resolve()};",
        ),
        encoding="utf-8",
    )
    rows = [_sample(seed + index) for index in range(samples)]

    def simulate(row: dict[str, float | int]) -> dict[str, float | int]:
        starts = int(row["num_start_node"])
        destination_start = starts
        command = [
            str(binary), str(config), str(row["network_size"]), str(row["latency_per_flit"]),
            str(row["tile_width_m"]), str(row["inject_rate"]), str(row["data_size_flits"]),
            "0", str(destination_start), str(destination_start), str(row["num_dest_node"]), "-1",
        ]
        result = parse_booksim_output(_run(command, root / "booksim2" / "src", timeout=120))
        return row | {
            "latency_cycles": result.latency_cycles,
            "dynamic_power_w": result.total_power_w,
            "area_m2": result.area_m2,
            "leakage_power_w": result.leakage_power_w,
            "flit_bits": BOOKSIM_FLIT_BITS,
        }

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as executor:
        completed = list(executor.map(simulate, rows))
    output = output.resolve()
    with output.open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(stream, fieldnames=FEATURES + TARGETS + ("flit_bits",))
        writer.writeheader()
        writer.writerows(completed)
    return {"samples": len(completed), "dataset": str(output), "flit_bits": BOOKSIM_FLIT_BITS}


def train_booksim_predictors(dataset: Path, output_dir: Path, seed: int) -> dict:
    try:
        from joblib import dump
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as error:
        raise RuntimeError("Install predictor dependencies with .venv/bin/pip install '.[predictor]'") from error
    with dataset.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 20:
        raise ValueError("at least 20 BookSim samples are required for a train/test split")
    output_dir.mkdir(parents=True, exist_ok=True)
    scores: dict[str, float] = {}
    for target in TARGETS:
        features = TARGET_FEATURES[target]
        inputs = np.asarray([[float(row[name]) for name in features] for row in rows])
        train_x, test_x = train_test_split(inputs, test_size=0.2, random_state=seed)
        values = np.asarray([float(row[target]) for row in rows])
        train_y, test_y = train_test_split(values, test_size=0.2, random_state=seed)
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation="relu", max_iter=1000, early_stopping=True, random_state=seed),
        )
        model.fit(train_x, train_y)
        scores[target] = float(r2_score(test_y, model.predict(test_x)))
        dump(model, output_dir / f"booksim_{target}.joblib")
    report = {"dataset": str(dataset), "samples": len(rows), "features": FEATURES, "r2": scores}
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
