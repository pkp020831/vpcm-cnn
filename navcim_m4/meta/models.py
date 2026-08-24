from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..booksim.models import _predictor_dependencies
from ..booksim.ranking import pareto_front, topsis_rank
from .contracts import file_reference, read_manifest, verify_file, write_manifest


LATENCY_FEATURES = ("predicted_booksim_latency_ns", "neurosim_latency_ns")
ENERGY_FEATURES = ("predicted_booksim_energy_pj", "neurosim_dynamic_energy_pj")


def _rows(dataset_manifest: Path) -> list[dict[str, str]]:
    manifest = read_manifest(dataset_manifest, "dataset")
    dataset = verify_file(manifest["outputs"]["dataset"])
    with dataset.open(newline="", encoding="ascii") as stream:
        return list(csv.DictReader(stream))


def create_split_stage(dataset_manifest: Path, output: Path, seed: int = 117, test_ratio: float = 0.25) -> dict[str, Any]:
    started = time.perf_counter()
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between zero and one")
    rows = _rows(dataset_manifest)
    configs = sorted({row["config_key"] for row in rows})
    if len(configs) < 3:
        raise ValueError("At least three hardware configurations are required for a config-level split")
    generator = np.random.default_rng(seed)
    shuffled = np.asarray(configs)
    generator.shuffle(shuffled)
    test_count = max(1, round(len(configs) * test_ratio))
    split = {"train": sorted(shuffled[test_count:].tolist()), "test": sorted(shuffled[:test_count].tolist())}
    return write_manifest(
        output,
        "split",
        {
            "inputs": {"dataset_manifest": file_reference(dataset_manifest)},
            "parameters": {"seed": seed, "test_ratio": test_ratio, "group": "config_key"},
            "split": split,
            "metrics": {
                "train_configs": len(split["train"]),
                "test_configs": len(split["test"]),
                "train_rows": sum(row["config_key"] in split["train"] for row in rows),
                "test_rows": sum(row["config_key"] in split["test"] for row in rows),
            },
        },
        started,
    )


def _split_rows(dataset_manifest: Path, split_manifest: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = _rows(dataset_manifest)
    split = read_manifest(split_manifest, "split")
    if split["inputs"]["dataset_manifest"] != file_reference(dataset_manifest):
        raise ValueError("Meta dataset and split do not match")
    train_keys, test_keys = set(split["split"]["train"]), set(split["split"]["test"])
    return (
        [row for row in rows if row["config_key"] in train_keys],
        [row for row in rows if row["config_key"] in test_keys],
    )


def _metrics(actual: np.ndarray, predicted: np.ndarray, dependencies: dict[str, Any]) -> dict[str, float]:
    return {
        "r2": float(dependencies["r2_score"](actual, predicted)),
        "mape": float(dependencies["mean_absolute_percentage_error"](actual, predicted)),
        "maximum_absolute_error": float(np.max(np.abs(actual - predicted))),
    }


def train_latency_stage(dataset_manifest: Path, split_manifest: Path, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    dependencies = _predictor_dependencies()
    train_rows, test_rows = _split_rows(dataset_manifest, split_manifest)
    train_x = np.asarray([[float(row[name]) for name in LATENCY_FEATURES] for row in train_rows])
    train_y = np.asarray([float(row["actual_overall_latency_ns"]) for row in train_rows])
    test_x = np.asarray([[float(row[name]) for name in LATENCY_FEATURES] for row in test_rows])
    test_y = np.asarray([float(row["actual_overall_latency_ns"]) for row in test_rows])
    model = dependencies["make_pipeline"](
        dependencies["PolynomialFeatures"](degree=2, include_bias=True),
        dependencies["LinearRegression"](),
    )
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "overall_latency_ns.joblib"
    dependencies["dump"](model, model_path)
    return write_manifest(
        output_dir / "overall_latency_ns.json",
        "latency-model",
        {
            "inputs": {
                "dataset_manifest": file_reference(dataset_manifest),
                "split_manifest": file_reference(split_manifest),
            },
            "features": list(LATENCY_FEATURES),
            "target": "actual_overall_latency_ns",
            "model": file_reference(model_path),
            "metrics": _metrics(test_y, predicted, dependencies),
            "parameters": {"family": "polynomial-regression", "degree": 2},
        },
        started,
    )


def _energy_matrix(rows: list[dict[str, str]], latency_model: Any) -> tuple[np.ndarray, np.ndarray]:
    latency_x = np.asarray([[float(row[name]) for name in LATENCY_FEATURES] for row in rows])
    predicted_overall_latency = latency_model.predict(latency_x)
    predicted_booksim_energy = np.asarray([
        float(row["predicted_booksim_dynamic_power_w"])
        * max(0.0, predicted_overall_latency[index] - float(row["neurosim_latency_ns"]))
        * 1000
        for index, row in enumerate(rows)
    ])
    inputs = np.column_stack((predicted_booksim_energy, [float(row["neurosim_dynamic_energy_pj"]) for row in rows]))
    targets = np.asarray([float(row["actual_overall_dynamic_energy_pj"]) for row in rows])
    return inputs, targets


def train_energy_stage(
    dataset_manifest: Path,
    split_manifest: Path,
    latency_model_manifest: Path,
    output_dir: Path,
    seed: int = 117,
    model_family: str = "auto",
) -> dict[str, Any]:
    started = time.perf_counter()
    dependencies = _predictor_dependencies()
    latency_manifest = read_manifest(latency_model_manifest, "latency-model")
    latency_model = dependencies["load"](verify_file(latency_manifest["model"]))
    train_rows, test_rows = _split_rows(dataset_manifest, split_manifest)
    train_x, train_y = _energy_matrix(train_rows, latency_model)
    test_x, test_y = _energy_matrix(test_rows, latency_model)
    if model_family not in {"auto", "mlp", "polynomial", "extra-trees"}:
        raise ValueError("Unsupported energy model family")

    def build(family: str):
        if family == "polynomial":
            return dependencies["make_pipeline"](
                dependencies["PolynomialFeatures"](degree=2, include_bias=True),
                dependencies["LinearRegression"](),
            )
        if family == "extra-trees":
            return dependencies["ExtraTreesRegressor"](
                n_estimators=300,
                min_samples_leaf=1,
                max_features=1.0,
                n_jobs=-1,
                random_state=seed,
            )
        regressor = dependencies["make_pipeline"](
            dependencies["StandardScaler"](),
            dependencies["MLPRegressor"](
                hidden_layer_sizes=(64, 32),
                activation="relu",
                learning_rate_init=0.001,
                batch_size=min(256, len(train_rows)),
                max_iter=1000,
                early_stopping=len(train_rows) >= 50,
                validation_fraction=0.3,
                random_state=seed,
            ),
        )
        return dependencies["TransformedTargetRegressor"](
            regressor=regressor,
            transformer=dependencies["StandardScaler"](),
        )

    cv_metrics = {}
    selected_family = model_family
    if model_family == "auto":
        from sklearn.base import clone
        from sklearn.model_selection import GroupKFold

        groups = np.asarray([row["config_key"] for row in train_rows])
        splitter = GroupKFold(n_splits=len(set(groups)))
        for family in ("mlp", "polynomial", "extra-trees"):
            fold_mapes = []
            for fold_train, fold_validation in splitter.split(train_x, train_y, groups):
                candidate = clone(build(family))
                candidate.fit(train_x[fold_train], train_y[fold_train])
                fold_prediction = candidate.predict(train_x[fold_validation])
                fold_mapes.append(float(dependencies["mean_absolute_percentage_error"](train_y[fold_validation], fold_prediction)))
            cv_metrics[family] = {"mean_mape": float(np.mean(fold_mapes)), "fold_mapes": fold_mapes}
        selected_family = min(cv_metrics, key=lambda family: cv_metrics[family]["mean_mape"])
    model = build(selected_family)
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "overall_dynamic_energy_pj.joblib"
    dependencies["dump"](model, model_path)
    return write_manifest(
        output_dir / "overall_dynamic_energy_pj.json",
        "energy-model",
        {
            "inputs": {
                "dataset_manifest": file_reference(dataset_manifest),
                "split_manifest": file_reference(split_manifest),
                "latency_model_manifest": file_reference(latency_model_manifest),
            },
            "features": list(ENERGY_FEATURES),
            "target": "actual_overall_dynamic_energy_pj",
            "model": file_reference(model_path),
            "metrics": _metrics(test_y, predicted, dependencies),
            "parameters": {
                "family": selected_family,
                "requested_family": model_family,
                "hidden_layers": [64, 32] if selected_family == "mlp" else None,
                "seed": seed,
                "config_group_cv": cv_metrics,
            },
        },
        started,
    )


def assemble_bundle_stage(latency_manifest: Path, energy_manifest: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    latency = read_manifest(latency_manifest, "latency-model")
    energy = read_manifest(energy_manifest, "energy-model")
    verify_file(latency["model"])
    verify_file(energy["model"])
    if energy["inputs"]["latency_model_manifest"] != file_reference(latency_manifest):
        raise ValueError("Energy model was not trained with the supplied latency model")
    return write_manifest(
        output,
        "bundle",
        {
            "inputs": {
                "latency_model_manifest": file_reference(latency_manifest),
                "energy_model_manifest": file_reference(energy_manifest),
            },
            "models": {
                "overall_latency_ns": latency["model"],
                "overall_dynamic_energy_pj": energy["model"],
            },
            "training_metrics": {
                "overall_latency_ns": latency["metrics"],
                "overall_dynamic_energy_pj": energy["metrics"],
            },
        },
        started,
    )


class MetaLearnerBundle:
    def __init__(self, manifest_path: Path) -> None:
        dependencies = _predictor_dependencies()
        self.path = manifest_path.resolve()
        self.manifest = read_manifest(self.path, "bundle")
        self.latency_model = dependencies["load"](verify_file(self.manifest["models"]["overall_latency_ns"]))
        self.energy_model = dependencies["load"](verify_file(self.manifest["models"]["overall_dynamic_energy_pj"]))

    def predict_latency(self, booksim_latency_ns: float, neurosim_latency_ns: float) -> float:
        return float(self.latency_model.predict([[booksim_latency_ns, neurosim_latency_ns]])[0])

    def predict_energy(self, booksim_energy_pj: float, neurosim_energy_pj: float) -> float:
        return float(self.energy_model.predict([[booksim_energy_pj, neurosim_energy_pj]])[0])


def evaluate_bundle_stage(bundle_manifest: Path, dataset_manifest: Path, split_manifest: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    dependencies = _predictor_dependencies()
    bundle = MetaLearnerBundle(bundle_manifest)
    _, test_rows = _split_rows(dataset_manifest, split_manifest)
    latency_x = np.asarray([[float(row[name]) for name in LATENCY_FEATURES] for row in test_rows])
    actual_latency = np.asarray([float(row["actual_overall_latency_ns"]) for row in test_rows])
    predicted_latency = bundle.latency_model.predict(latency_x)
    predicted_booksim_energy = np.asarray([
        float(row["predicted_booksim_dynamic_power_w"])
        * max(0.0, predicted_latency[index] - float(row["neurosim_latency_ns"]))
        * 1000
        for index, row in enumerate(test_rows)
    ])
    energy_x = np.column_stack((predicted_booksim_energy, [float(row["neurosim_dynamic_energy_pj"]) for row in test_rows]))
    actual_energy = np.asarray([float(row["actual_overall_dynamic_energy_pj"]) for row in test_rows])
    predicted_energy = bundle.energy_model.predict(energy_x)
    return write_manifest(
        output,
        "evaluation",
        {
            "inputs": {
                "bundle_manifest": file_reference(bundle_manifest),
                "dataset_manifest": file_reference(dataset_manifest),
                "split_manifest": file_reference(split_manifest),
            },
            "metrics": {
                "overall_latency_ns": _metrics(actual_latency, predicted_latency, dependencies),
                "overall_dynamic_energy_pj": _metrics(actual_energy, predicted_energy, dependencies),
            },
            "test_rows": len(test_rows),
        },
        started,
    )


def compare_candidates_stage(
    bundle_manifest: Path,
    dataset_manifest: Path,
    output: Path,
    top_k: int = 3,
) -> dict[str, Any]:
    started = time.perf_counter()
    bundle = MetaLearnerBundle(bundle_manifest)
    rows = _rows(dataset_manifest)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["config_key"], []).append(row)
    actual_candidates, predicted_candidates = [], []
    for key, config_rows in grouped.items():
        latency_x = np.asarray([[float(row[name]) for name in LATENCY_FEATURES] for row in config_rows])
        predicted_latency_layers = bundle.latency_model.predict(latency_x)
        predicted_booksim_energy = np.asarray([
            float(row["predicted_booksim_dynamic_power_w"])
            * max(0.0, predicted_latency_layers[index] - float(row["neurosim_latency_ns"]))
            * 1000
            for index, row in enumerate(config_rows)
        ])
        energy_x = np.column_stack((predicted_booksim_energy, [float(row["neurosim_dynamic_energy_pj"]) for row in config_rows]))
        predicted_energy_layers = bundle.energy_model.predict(energy_x)
        actual_latency = sum(float(row["actual_overall_latency_ns"]) for row in config_rows)
        actual_energy = sum(float(row["actual_overall_dynamic_energy_pj"]) for row in config_rows)
        predicted_latency = float(predicted_latency_layers.sum())
        predicted_energy = float(predicted_energy_layers.sum())
        first = config_rows[0]
        latency_residual = max(0.0, float(first["neurosim_chip_latency_ns"]) - sum(float(row["neurosim_latency_ns"]) for row in config_rows))
        energy_residual = max(0.0, float(first["neurosim_chip_dynamic_energy_pj"]) - sum(float(row["neurosim_dynamic_energy_pj"]) for row in config_rows))
        actual_latency += latency_residual
        predicted_latency += latency_residual
        actual_energy += energy_residual
        predicted_energy += energy_residual
        actual_leakage = float(first["neurosim_chip_leakage_power_uw"]) / 1000 + float(first["actual_booksim_leakage_power_w"]) * 1000
        predicted_leakage = float(first["neurosim_chip_leakage_power_uw"]) / 1000 + float(first["predicted_booksim_leakage_power_w"]) * 1000
        actual_candidates.append({
            "config_key": key,
            "latency_ns": actual_latency,
            "total_power_mw": actual_energy / actual_latency + actual_leakage,
            "area_um2": float(first["neurosim_chip_area_um2"]) + float(first["actual_booksim_area_m2"]) * 1e6,
        })
        predicted_candidates.append({
            "config_key": key,
            "latency_ns": predicted_latency,
            "total_power_mw": predicted_energy / predicted_latency + predicted_leakage,
            "area_um2": float(first["neurosim_chip_area_um2"]) + float(first["predicted_booksim_area_m2"]) * 1e6,
        })
    objectives = {"latency_ns": "min", "total_power_mw": "min", "area_um2": "min"}
    weights = {name: 1.0 for name in objectives}
    actual_ranked = topsis_rank(pareto_front(actual_candidates, objectives), objectives, weights)
    predicted_ranked = topsis_rank(pareto_front(predicted_candidates, objectives), objectives, weights)
    count = min(top_k, len(actual_ranked), len(predicted_ranked))
    actual_top = [candidate["config_key"] for candidate in actual_ranked[:count]]
    predicted_top = [candidate["config_key"] for candidate in predicted_ranked[:count]]
    actual_by_key = {candidate["config_key"]: candidate for candidate in actual_candidates}
    predicted_by_key = {candidate["config_key"]: candidate for candidate in predicted_candidates}
    return write_manifest(
        output,
        "candidate-comparison",
        {
            "inputs": {
                "bundle_manifest": file_reference(bundle_manifest),
                "dataset_manifest": file_reference(dataset_manifest),
            },
            "parameters": {"top_k": top_k, "objectives": objectives, "weights": weights},
            "actual_top_k": actual_top,
            "predicted_top_k": predicted_top,
            "metrics": {
                "candidate_count": len(grouped),
                "evaluated_top_k": count,
                "top_k_overlap": len(set(actual_top) & set(predicted_top)),
                "top_k_recall": len(set(actual_top) & set(predicted_top)) / count if count else 0.0,
            },
            "comparisons": [{
                "config_key": key,
                "actual": actual_by_key[key],
                "predicted": predicted_by_key[key],
                "relative_error": {
                    name: abs(predicted_by_key[key][name] - actual_by_key[key][name]) / abs(actual_by_key[key][name]) if actual_by_key[key][name] else 0.0
                    for name in objectives
                },
            } for key in predicted_top],
        },
        started,
    )
