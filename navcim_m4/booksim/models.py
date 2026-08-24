from __future__ import annotations

import csv
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .contracts import file_reference, read_manifest, verify_file, write_manifest
from .features import FEATURES, TARGETS, TARGET_FEATURES, BookSimFeatures
from .ranking import pareto_front, topsis_rank


class _LatencyNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        widths = (7, 16, 32, 64, 128, 64, 32, 16, 1)
        layers: list[nn.Module] = []
        for index, (input_width, output_width) in enumerate(zip(widths, widths[1:])):
            layers.append(nn.Linear(input_width, output_width))
            if index < len(widths) - 2:
                layers.extend((nn.BatchNorm1d(output_width), nn.ReLU()))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class TorchLatencyRegressor:
    """Paper-compatible latency MLP with explicit input/target scaling."""

    def __init__(self, seed: int = 117, max_epochs: int = 1000) -> None:
        self.seed = seed
        self.max_epochs = max_epochs
        torch.manual_seed(seed)
        self.model = _LatencyNetwork()
        self.input_mean = np.zeros(7)
        self.input_scale = np.ones(7)
        self.target_min = 0.0
        self.target_scale = 1.0
        self.training_epochs = 0

    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> TorchLatencyRegressor:
        torch.manual_seed(self.seed)
        generator = np.random.default_rng(self.seed)
        self.input_mean = inputs.mean(axis=0)
        self.input_scale = inputs.std(axis=0)
        self.input_scale[self.input_scale == 0] = 1.0
        self.target_min = float(targets.min())
        self.target_scale = max(float(targets.max()) - self.target_min, 1.0)
        normalized_x = ((inputs - self.input_mean) / self.input_scale).astype(np.float32)
        normalized_y = (((targets - self.target_min) / self.target_scale) * 100).astype(np.float32)
        indices = generator.permutation(len(inputs))
        validation_count = max(1, round(len(indices) * 0.2))
        validation_indices, train_indices = indices[:validation_count], indices[validation_count:]
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(normalized_x[train_indices]), torch.from_numpy(normalized_y[train_indices, None])),
            batch_size=min(1024, len(train_indices)),
            shuffle=True,
            drop_last=len(train_indices) > 1024,
        )
        validation_x = torch.from_numpy(normalized_x[validation_indices])
        validation_y = torch.from_numpy(normalized_y[validation_indices, None])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        loss_function = nn.MSELoss()
        best_loss = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        stale_epochs = 0
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss_function(self.model(batch_x), batch_y).backward()
                optimizer.step()
            self.model.eval()
            with torch.inference_mode():
                validation_loss = float(loss_function(self.model(validation_x), validation_y))
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_state = copy.deepcopy(self.model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            self.training_epochs = epoch
            if stale_epochs >= 40:
                break
        self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict(self, inputs: np.ndarray | list[list[float]]) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float32)
        normalized = ((values - self.input_mean) / self.input_scale).astype(np.float32)
        self.model.eval()
        with torch.inference_mode():
            predictions = self.model(torch.from_numpy(normalized)).numpy().reshape(-1)
        return predictions / 100 * self.target_scale + self.target_min


def _predictor_dependencies():
    try:
        from joblib import dump, load
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
        from sklearn.metrics import mean_absolute_percentage_error, r2_score
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        from sklearn.compose import TransformedTargetRegressor
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Install predictor dependencies with .venv/bin/pip install '.[predictor]'"
        ) from error
    return {
        "dump": dump,
        "load": load,
        "LinearRegression": LinearRegression,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "ExtraTreesRegressor": ExtraTreesRegressor,
        "mean_absolute_percentage_error": mean_absolute_percentage_error,
        "MLPRegressor": MLPRegressor,
        "make_pipeline": make_pipeline,
        "PolynomialFeatures": PolynomialFeatures,
        "r2_score": r2_score,
        "StandardScaler": StandardScaler,
        "TransformedTargetRegressor": TransformedTargetRegressor,
    }


def _dataset_rows(dataset_manifest: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    manifest = read_manifest(dataset_manifest, "dataset")
    dataset = verify_file(manifest["outputs"]["dataset"])
    with dataset.open(newline="", encoding="ascii") as stream:
        return list(csv.DictReader(stream)), manifest


def create_split_stage(
    dataset_manifest: Path,
    output: Path,
    seed: int = 117,
    test_ratio: float = 0.2,
) -> dict[str, Any]:
    started = time.perf_counter()
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between zero and one")
    rows, _ = _dataset_rows(dataset_manifest)
    if len(rows) < 20:
        raise ValueError("At least 20 completed samples are required")
    generator = np.random.default_rng(seed)
    sample_ids = np.asarray([row["sample_id"] for row in rows])
    generator.shuffle(sample_ids)
    test_count = max(1, round(len(sample_ids) * test_ratio))
    split = {
        "train": sorted(sample_ids[test_count:].tolist()),
        "test": sorted(sample_ids[:test_count].tolist()),
    }
    return write_manifest(
        output,
        "split",
        {
            "inputs": {"dataset_manifest": file_reference(dataset_manifest)},
            "parameters": {"seed": seed, "test_ratio": test_ratio},
            "split": split,
            "metrics": {"train_samples": len(split["train"]), "test_samples": len(split["test"])},
        },
        started,
    )


def train_target_stage(
    dataset_manifest: Path,
    split_manifest: Path,
    target: str,
    output_dir: Path,
    seed: int = 117,
    model_family: str = "auto",
) -> dict[str, Any]:
    started = time.perf_counter()
    if target not in TARGETS:
        raise ValueError(f"Unsupported target {target!r}; expected one of {TARGETS}")
    if model_family not in {"auto", "paper-mlp", "extra-trees", "mlp"}:
        raise ValueError("Unsupported model family")
    dependencies = _predictor_dependencies()
    rows, _ = _dataset_rows(dataset_manifest)
    split = read_manifest(split_manifest, "split")
    if split["inputs"]["dataset_manifest"] != file_reference(dataset_manifest):
        raise ValueError("Split and training dataset manifests do not match")
    train_ids, test_ids = set(split["split"]["train"]), set(split["split"]["test"])
    features = TARGET_FEATURES[target]

    def matrix(ids: set[str]) -> tuple[np.ndarray, np.ndarray]:
        selected = [row for row in rows if row["sample_id"] in ids]
        x = np.asarray([[float(row[name]) for name in features] for row in selected])
        y = np.asarray([float(row[target]) for row in selected])
        return x, y

    train_x, train_y = matrix(train_ids)
    test_x, test_y = matrix(test_ids)
    if target == "latency_cycles":
        hidden_layers = (16, 32, 64, 128, 64, 32, 16)
    elif target == "dynamic_power_w":
        hidden_layers = (64, 32, 16)
    else:
        hidden_layers = (32, 16)
    if target == "latency_cycles":
        selected_family = "extra-trees" if model_family == "auto" else model_family
        if selected_family == "paper-mlp":
            model = TorchLatencyRegressor(seed=seed)
        elif selected_family == "extra-trees":
            model = dependencies["make_pipeline"](
                dependencies["StandardScaler"](),
                dependencies["ExtraTreesRegressor"](
                    n_estimators=300,
                    min_samples_leaf=1,
                    max_features=1.0,
                    n_jobs=-1,
                    random_state=seed,
                ),
            )
        else:
            model = dependencies["make_pipeline"](
                dependencies["StandardScaler"](),
                dependencies["MLPRegressor"](
                    hidden_layer_sizes=hidden_layers,
                    activation="relu",
                    max_iter=1000,
                    early_stopping=True,
                    random_state=seed,
                ),
            )
    else:
        if model_family not in {"auto", "mlp"}:
            raise ValueError(f"{model_family} is only supported for latency")
        selected_family = "mlp"
        model = dependencies["make_pipeline"](
            dependencies["StandardScaler"](),
            dependencies["MLPRegressor"](
                hidden_layer_sizes=hidden_layers,
                activation="relu",
                max_iter=1000,
                early_stopping=len(train_y) >= 50,
                random_state=seed,
            ),
        )
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{target}.joblib"
    dependencies["dump"](model, model_path)
    ranges = {
        name: {"min": float(train_x[:, index].min()), "max": float(train_x[:, index].max())}
        for index, name in enumerate(features)
    }
    metrics = {
        "r2": float(dependencies["r2_score"](test_y, predicted)),
        "mape": float(dependencies["mean_absolute_percentage_error"](test_y, predicted)),
        "maximum_absolute_error": float(np.max(np.abs(test_y - predicted))),
    }
    return write_manifest(
        output_dir / f"{target}.json",
        "target-model",
        {
            "inputs": {
                "dataset_manifest": file_reference(dataset_manifest),
                "split_manifest": file_reference(split_manifest),
            },
            "target": target,
            "features": list(features),
            "feature_ranges": ranges,
            "model": file_reference(model_path),
            "metrics": metrics,
            "parameters": {
                "seed": seed,
                "hidden_layers": list(hidden_layers),
                "training_epochs": getattr(model, "training_epochs", None),
                "model_family": selected_family,
            },
        },
        started,
    )


def assemble_bundle_stage(model_manifests: list[Path], output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    models: dict[str, Any] = {}
    dataset_manifests: set[str] = set()
    split_manifests: set[str] = set()
    for path in model_manifests:
        manifest = read_manifest(path, "target-model")
        target = manifest["target"]
        if target in models:
            raise ValueError(f"Duplicate target model: {target}")
        verify_file(manifest["model"])
        models[target] = {
            "manifest": str(path.resolve()),
            "features": manifest["features"],
            "feature_ranges": manifest["feature_ranges"],
            "model": manifest["model"],
            "training_metrics": manifest["metrics"],
        }
        dataset_manifests.add(json.dumps(manifest["inputs"]["dataset_manifest"], sort_keys=True))
        split_manifests.add(json.dumps(manifest["inputs"]["split_manifest"], sort_keys=True))
    missing = set(TARGETS) - set(models)
    if missing:
        raise ValueError(f"Bundle is missing target models: {sorted(missing)}")
    if len(dataset_manifests) != 1 or len(split_manifests) != 1:
        raise ValueError("All target models must use the same dataset and split")
    output_dir = output_dir.resolve()
    return write_manifest(
        output_dir / "bundle.json",
        "predictor-bundle",
        {
            "inputs": {
                "model_manifests": [str(path.resolve()) for path in model_manifests],
                "dataset_manifest": json.loads(dataset_manifests.pop()),
                "split_manifest": json.loads(split_manifests.pop()),
            },
            "models": models,
            "units": {
                "latency_cycles": "cycles",
                "dynamic_power_w": "W",
                "area_m2": "mm^2 (legacy field name)",
                "leakage_power_w": "W",
            },
        },
        started,
    )


class PredictorBundle:
    def __init__(self, manifest_path: Path) -> None:
        dependencies = _predictor_dependencies()
        self.path = manifest_path.resolve()
        self.manifest = read_manifest(self.path, "predictor-bundle")
        self.models = {
            target: dependencies["load"](verify_file(specification["model"]))
            for target, specification in self.manifest["models"].items()
        }

    def out_of_distribution(self, features: BookSimFeatures) -> list[str]:
        values = features.__dict__
        violations = []
        for target, specification in self.manifest["models"].items():
            for name, limits in specification["feature_ranges"].items():
                value = float(values[name])
                if not limits["min"] <= value <= limits["max"]:
                    violations.append(f"{target}:{name}={value}")
        return sorted(set(violations))

    def predict(self, features: BookSimFeatures) -> dict[str, float]:
        return self.predict_many([features])[0]

    def predict_many(self, features: list[BookSimFeatures]) -> list[dict[str, float]]:
        predictions = [dict() for _ in features]
        for target, model in self.models.items():
            names = self.manifest["models"][target]["features"]
            matrix = [[float(item.__dict__[name]) for name in names] for item in features]
            values = model.predict(matrix)
            for index, value in enumerate(values):
                predictions[index][target] = float(value)
        return predictions


def evaluate_bundle_stage(
    bundle_manifest: Path,
    dataset_manifest: Path,
    split_manifest: Path,
    output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    dependencies = _predictor_dependencies()
    bundle = PredictorBundle(bundle_manifest)
    rows, _ = _dataset_rows(dataset_manifest)
    split = read_manifest(split_manifest, "split")
    test_ids = set(split["split"]["test"])
    test_rows = [row for row in rows if row["sample_id"] in test_ids]
    feature_rows = [BookSimFeatures(**{
            name: int(float(row[name])) if name in {"network_size", "latency_per_flit", "data_size_flits", "num_start_node", "num_dest_node"} else float(row[name])
            for name in FEATURES
        }) for row in test_rows]
    predictions = list(zip(test_rows, bundle.predict_many(feature_rows)))
    metrics = {}
    for target in TARGETS:
        actual = np.asarray([float(row[target]) for row, _ in predictions])
        predicted = np.asarray([result[target] for _, result in predictions])
        metrics[target] = {
            "r2": float(dependencies["r2_score"](actual, predicted)),
            "mape": float(dependencies["mean_absolute_percentage_error"](actual, predicted)),
            "maximum_absolute_error": float(np.max(np.abs(actual - predicted))),
        }
    return write_manifest(
        output,
        "bundle-evaluation",
        {
            "inputs": {
                "bundle_manifest": str(bundle_manifest.resolve()),
                "dataset_manifest": str(dataset_manifest.resolve()),
                "split_manifest": str(split_manifest.resolve()),
            },
            "metrics": metrics,
            "test_samples": len(test_rows),
        },
        started,
    )


def compare_top_k_stage(
    bundle_manifest: Path,
    dataset_manifest: Path,
    split_manifest: Path,
    output: Path,
    top_k: int = 5,
    weights: dict[str, float] | None = None,
    validation_pool: int = 100,
) -> dict[str, Any]:
    started = time.perf_counter()
    if top_k < 1:
        raise ValueError("top_k must be positive")
    weights = weights or {"latency_cycles": 1.0, "dynamic_power_w": 1.0, "area_m2": 1.0}
    objectives = {name: "min" for name in weights}
    bundle = PredictorBundle(bundle_manifest)
    rows, _ = _dataset_rows(dataset_manifest)
    split = read_manifest(split_manifest, "split")
    test_ids = set(split["split"]["test"])
    actual_candidates, feature_rows = [], []
    for row in rows:
        if row["sample_id"] not in test_ids:
            continue
        features = BookSimFeatures(**{
            name: int(float(row[name])) if name in {"network_size", "latency_per_flit", "data_size_flits", "num_start_node", "num_dest_node"} else float(row[name])
            for name in FEATURES
        })
        actual_candidates.append({"sample_id": row["sample_id"], **{name: float(row[name]) for name in objectives}})
        feature_rows.append(features)
    predicted_candidates = [
        {"sample_id": actual["sample_id"], **{name: prediction[name] for name in objectives}}
        for actual, prediction in zip(actual_candidates, bundle.predict_many(feature_rows), strict=True)
    ]

    def rank(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return topsis_rank(pareto_front(candidates, objectives), objectives, weights)

    actual_ranked, predicted_ranked = rank(actual_candidates), rank(predicted_candidates)
    count = min(top_k, len(actual_ranked), len(predicted_ranked))
    actual_top = [candidate["sample_id"] for candidate in actual_ranked[:count]]
    predicted_top = [candidate["sample_id"] for candidate in predicted_ranked[:count]]
    all_predicted_ranked = topsis_rank(predicted_candidates, objectives, weights)
    pool_count = min(max(validation_pool, count), len(all_predicted_ranked))
    pool_order = predicted_top + [candidate["sample_id"] for candidate in all_predicted_ranked]
    predicted_pool = list(dict.fromkeys(pool_order))[:pool_count]
    actual_by_id = {candidate["sample_id"]: candidate for candidate in actual_candidates}
    predicted_by_id = {candidate["sample_id"]: candidate for candidate in predicted_candidates}
    comparisons = []
    for sample_id in predicted_top:
        actual, predicted = actual_by_id[sample_id], predicted_by_id[sample_id]
        comparisons.append({
            "sample_id": sample_id,
            "actual": {name: actual[name] for name in objectives},
            "predicted": {name: predicted[name] for name in objectives},
            "relative_error": {
                name: abs(predicted[name] - actual[name]) / abs(actual[name]) if actual[name] else 0.0
                for name in objectives
            },
        })
    return write_manifest(
        output,
        "top-k-comparison",
        {
            "inputs": {
                "bundle_manifest": str(bundle_manifest.resolve()),
                "dataset_manifest": str(dataset_manifest.resolve()),
                "split_manifest": str(split_manifest.resolve()),
            },
            "parameters": {"top_k": top_k, "validation_pool": validation_pool, "weights": weights, "objectives": objectives},
            "predicted_top_k": predicted_top,
            "predicted_validation_pool": predicted_pool,
            "actual_top_k": actual_top,
            "metrics": {
                "evaluated_top_k": count,
                "top_k_overlap": len(set(predicted_top) & set(actual_top)),
                "top_k_recall": len(set(predicted_top) & set(actual_top)) / count if count else 0.0,
                "actual_top_k_recall_in_validation_pool": len(set(predicted_pool) & set(actual_top)) / count if count else 0.0,
            },
            "comparisons": comparisons,
        },
        started,
    )
