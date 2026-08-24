from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .models import ModelInfo, VGG11Cifar10, load_checkpoint


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass(frozen=True)
class CrossSimConfig:
    adc_bits: int = 5
    dac_bits: int = 8
    cell_bits: int = 2
    rows: int = 128
    cols: int = 128
    weight_slices: int = 1
    input_slice_size: int = 1
    programming_error: float = 0.0
    read_noise: float = 0.0
    lumped_read_noise: bool = False
    drift_time: float = 0.0
    wire_resistance: float = 0.0
    seed: int = 117

    def validate(self) -> None:
        positive = ("cell_bits", "rows", "cols", "weight_slices", "input_slice_size")
        if any(getattr(self, name) < 1 for name in positive) or self.adc_bits < 0 or self.dac_bits < 0:
            raise ValueError("ADC/DAC bits must be non-negative; cell, array, and bit-slice settings must be positive")
        if self.dac_bits and self.dac_bits % self.input_slice_size:
            raise ValueError("--input-slice-size must divide --dac-bits")
        if any(getattr(self, name) < 0 for name in ("programming_error", "read_noise", "drift_time", "wire_resistance")):
            raise ValueError("CrossSim non-ideality magnitudes must be non-negative")
        if self.drift_time:
            raise ValueError("CrossSim 3.2.1 has no validated generic conductance-drift model; set --drift-time 0")


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    if requested not in {"mps", "cpu"}:
        raise ValueError("--device must be auto, mps, or cpu")
    return torch.device(requested)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cifar10_loader(data_dir: Path, train: bool, batch_size: int, samples: int | None = None, augment: bool = False) -> DataLoader:
    steps: list[Any] = []
    if augment and train:
        steps.extend((transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()))
    steps.extend((transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)))
    dataset = datasets.CIFAR10(root=data_dir, train=train, transform=transforms.Compose(steps), download=True)
    if samples is not None:
        if samples < 1:
            raise ValueError("--samples must be positive")
        dataset = torch.utils.data.Subset(dataset, range(min(samples, len(dataset))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float | int]:
    model.eval()
    correct = total = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for inputs, labels in loader:
            logits = model(inputs.to(device))
            correct += int((logits.argmax(dim=1).cpu() == labels).sum())
            total += len(labels)
    elapsed = time.perf_counter() - started
    return {"accuracy": 100 * correct / total, "correct": correct, "samples": total, "seconds": elapsed}


def _save_training_checkpoint(checkpoint: Path, model: VGG11Cifar10, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int, target_epochs: int, seed: int, training_accuracy: float, validation: dict[str, float | int], history: list[dict[str, Any]], completed: bool) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.cpu()
    torch.save({
        "model_name": ModelInfo().name,
        "num_classes": ModelInfo().classes,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "target_epochs": target_epochs,
        "seed": seed,
        "training_accuracy": training_accuracy,
        "validation_accuracy": validation["accuracy"],
        "validation": validation,
        "history": history,
        "completed": completed,
        "preprocessing": {"mean": CIFAR10_MEAN, "std": CIFAR10_STD, "input_shape": ModelInfo().input_shape},
    }, checkpoint)


def train_vgg11(data_dir: Path, checkpoint: Path, device_name: str, epochs: int, batch_size: int, seed: int, reuse: bool = True, train_samples: int | None = None, checkpoint_interval: int = 10, resume: bool = False) -> dict[str, Any]:
    if resume and reuse:
        raise ValueError("--resume cannot be combined with checkpoint reuse")
    if checkpoint.is_file() and reuse:
        _, saved = load_checkpoint(checkpoint)
        return {"checkpoint": str(checkpoint), "reused": True, "validation_accuracy": saved["validation_accuracy"]}
    if epochs < 1 or checkpoint_interval < 1:
        raise ValueError("--epochs and --checkpoint-interval must be positive")
    _seed(seed)
    device = select_device(device_name)
    model = VGG11Cifar10()
    start_epoch = 0
    history: list[dict[str, Any]] = []
    saved: dict[str, Any] | None = None
    if resume:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Cannot resume missing checkpoint: {checkpoint}")
        model, saved = load_checkpoint(checkpoint)
        start_epoch = int(saved["epoch"])
        if int(saved.get("target_epochs", epochs)) != epochs:
            raise ValueError("--epochs must match the target epoch count stored in the checkpoint when resuming")
        history = list(saved.get("history", []))
        if start_epoch >= epochs:
            return {"checkpoint": str(checkpoint), "reused": True, "validation_accuracy": saved["validation_accuracy"], "epoch": start_epoch}
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        for state in optimizer.state.values():
            for name, value in state.items():
                if torch.is_tensor(value):
                    state[name] = value.to(device)
    train_loader = cifar10_loader(data_dir, True, batch_size, train_samples, augment=True)
    test_loader = cifar10_loader(data_dir, False, batch_size)
    train_accuracy = 0.0
    validation: dict[str, float | int] = {"accuracy": 0.0, "correct": 0, "samples": 0, "seconds": 0.0}
    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        correct = total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            torch.nn.functional.cross_entropy(logits, labels).backward()
            optimizer.step()
            correct += int((logits.argmax(1) == labels).sum())
            total += len(labels)
        scheduler.step()
        train_accuracy = 100 * correct / total
        if epoch % checkpoint_interval == 0 or epoch == epochs:
            validation = evaluate(model, test_loader, device)
            record = {"epoch": epoch, "training_accuracy": train_accuracy, "validation_accuracy": validation["accuracy"], "validation_correct": validation["correct"], "validation_seconds": validation["seconds"]}
            history.append(record)
            _save_training_checkpoint(checkpoint, model, optimizer, scheduler, epoch, epochs, seed, train_accuracy, validation, history, epoch == epochs)
            model.to(device)
            print(json.dumps(record, sort_keys=True), flush=True)
    return {"checkpoint": str(checkpoint), "reused": False, "device": str(device), "epoch": epochs, "training_accuracy": train_accuracy, "validation_accuracy": validation["accuracy"], "history": history}


def checkpoint_sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def _crosssim_source() -> Path:
    source = Path(__file__).resolve().parents[1] / "third_party" / "cross-sim-3.2"
    if not source.is_dir():
        raise FileNotFoundError("Official CrossSim 3.2 submodule is missing; run git submodule update --init third_party/cross-sim-3.2")
    return source


def _crosssim_model(model: VGG11Cifar10, config: CrossSimConfig, input_bound: float, output_bounds: list[float]) -> nn.Module:
    source = _crosssim_source()
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from applications.dnn.dnn_inference_params import dnn_inference_params
    from simulator.algorithms.dnn.torch import convertible_modules, from_torch

    config.validate()
    np.random.seed(config.seed)
    weight_bits = config.cell_bits * config.weight_slices + 1
    params = []
    for index, _ in enumerate(convertible_modules(model)):
        bound = max(output_bounds[index], 1e-4)
        params.append(dnn_inference_params(
            core_style="BALANCED", Nslices=config.weight_slices, weight_bits=weight_bits,
            NrowsMax=config.rows, NcolsMax=config.cols, adc_bits=config.adc_bits,
            input_bits=config.dac_bits, input_bitslicing=config.dac_bits > 0 and config.input_slice_size < config.dac_bits,
            input_slice_size=config.input_slice_size, input_range=[-input_bound, input_bound],
            adc_range=[-bound, bound], useGPU=False, digital_bias=True,
            error_model="generic" if config.programming_error else "none", alpha_error=config.programming_error,
            noise_model="generic" if config.read_noise else "none", alpha_noise=config.read_noise,
            lumped_read_noise=config.lumped_read_noise,
            drift_model="none", t_drift=0,
            Rp_row=config.wire_resistance, Rp_col=config.wire_resistance,
        ))
    return from_torch(model, params, fuse_batchnorm=True).eval()


def _calibration_bounds(model: VGG11Cifar10, loader: DataLoader) -> tuple[float, list[float]]:
    bounds: list[float] = []
    hooks = []
    for layer in (item for item in model.modules() if isinstance(item, (nn.Conv2d, nn.Linear))):
        bounds.append(0.0)
        index = len(bounds) - 1
        hooks.append(layer.register_forward_hook(lambda _m, _i, output, index=index: bounds.__setitem__(index, max(bounds[index], float(output.detach().abs().max())))))
    input_bound = 0.0
    with torch.inference_mode():
        for inputs, _ in loader:
            input_bound = max(input_bound, float(inputs.abs().max()))
            model(inputs)
    for hook in hooks:
        hook.remove()
    return input_bound, bounds


def evaluate_crosssim(checkpoint: Path, data_dir: Path, samples: int, batch_size: int, runs: int, config: CrossSimConfig, output_dir: Path) -> dict[str, Any]:
    if runs < 1:
        raise ValueError("--runs must be positive")
    config.validate()
    model, _ = load_checkpoint(checkpoint)
    loader = cifar10_loader(data_dir, False, batch_size, samples)
    digital = evaluate(model, loader, torch.device("cpu"))
    input_bound, output_bounds = _calibration_bounds(model, loader)
    accuracies = []
    per_run = []
    for run in range(runs):
        run_config = CrossSimConfig(**(asdict(config) | {"seed": config.seed + run}))
        analog = _crosssim_model(model, run_config, input_bound, output_bounds)
        result = evaluate(analog, loader, torch.device("cpu"))
        accuracies.append(float(result["accuracy"]))
        per_run.append(result | {"seed": run_config.seed})
    report = {
        "digital_accuracy": digital["accuracy"], "digital_correct": digital["correct"], "digital_seconds": digital["seconds"],
        "crosssim_accuracy_mean": float(np.mean(accuracies)), "crosssim_accuracy_std": float(np.std(accuracies)),
        "accuracy_drop_percentage_points": float(digital["accuracy"] - np.mean(accuracies)),
        "crosssim_samples": digital["samples"], "crosssim_runs": runs, "crosssim_config": asdict(config),
        "checkpoint_sha256": checkpoint_sha256(checkpoint), "runs": per_run,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "crosssim_results.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def cached_crosssim(checkpoint: Path, data_dir: Path, samples: int, batch_size: int, runs: int, config: CrossSimConfig, cache_dir: Path) -> dict[str, Any]:
    dataset_artifact = data_dir.resolve() / "cifar-10-batches-py" / "test_batch"
    dataset_identity = {
        "path": str(data_dir.resolve()),
        "test_batch_sha256": checkpoint_sha256(dataset_artifact) if dataset_artifact.is_file() else None,
    }
    key = hashlib.sha256(json.dumps({
        "checkpoint": checkpoint_sha256(checkpoint),
        "dataset": dataset_identity,
        "samples": samples,
        "batch_size": batch_size,
        "runs": runs,
        "config": asdict(config),
    }, sort_keys=True).encode()).hexdigest()
    cached = cache_dir / f"{key}.json"
    if cached.is_file():
        return json.loads(cached.read_text(encoding="utf-8"))
    work_dir = cache_dir / f"{key}.work-{os.getpid()}"
    report = evaluate_crosssim(checkpoint, data_dir, samples, batch_size, runs, config, work_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_dir / f".{key}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, cached)
    return report


def prescribed_sweep_conditions(seed: int = 117) -> list[tuple[str, CrossSimConfig, int]]:
    conditions: list[tuple[str, CrossSimConfig, int]] = []
    for adc_bits in (4, 5, 8):
        for cell_bits, weight_slices in ((7, 1), (1, 7)):
            base = CrossSimConfig(adc_bits=adc_bits, cell_bits=cell_bits, weight_slices=weight_slices, seed=seed)
            label = f"adc{adc_bits}_cell{cell_bits}_slices{weight_slices}"
            conditions.append((f"{label}_ideal", base, 1))
            for field, levels in (("programming_error", (0.01, 0.03)), ("read_noise", (0.01, 0.03)), ("wire_resistance", (0.25, 1.0))):
                for level in levels:
                    conditions.append((f"{label}_{field}{level}", CrossSimConfig(**(asdict(base) | {field: level})), 3))
    return conditions


def run_prescribed_sweep(checkpoint: Path, data_dir: Path, output_dir: Path, samples: int = 100, batch_size: int = 100, seed: int = 117) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = prescribed_sweep_conditions(seed)
    results = []
    for index, (label, config, runs) in enumerate(conditions, start=1):
        report = cached_crosssim(checkpoint, data_dir, samples, batch_size, runs, config, output_dir / "cache")
        results.append({"label": label, **report})
        print(json.dumps({"completed": index, "total": len(conditions), "label": label, "crosssim_accuracy_mean": report["crosssim_accuracy_mean"], "accuracy_drop_percentage_points": report["accuracy_drop_percentage_points"]}, sort_keys=True), flush=True)
    summary = {"checkpoint_sha256": checkpoint_sha256(checkpoint), "samples": samples, "batch_size": batch_size, "conditions": results}
    (output_dir / "sweep_results.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
