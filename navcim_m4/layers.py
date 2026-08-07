from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from .models import VGG11Cifar10, sample_input


@dataclass(frozen=True)
class LayerRecord:
    name: str
    input_height: int
    input_width: int
    input_channels: int
    kernel_height: int
    kernel_width: int
    output_channels: int
    pool_after: int
    stride: int
    is_fc: int

    def csv_row(self) -> tuple[int, ...]:
        return (
            self.input_height,
            self.input_width,
            self.input_channels,
            self.kernel_height,
            self.kernel_width,
            self.output_channels,
            self.pool_after,
            self.stride,
            self.is_fc,
        )


@dataclass(frozen=True)
class LayerArtifacts:
    records: tuple[LayerRecord, ...]
    weight_files: tuple[Path, ...]
    input_files: tuple[Path, ...]


def _pooling_map(model: VGG11Cifar10) -> dict[nn.Module, int]:
    result: dict[nn.Module, int] = {}
    modules = list(model.features)
    for index, module in enumerate(modules):
        if isinstance(module, nn.Conv2d):
            next_conv_or_pool = next(
                (
                    candidate
                    for candidate in modules[index + 1 :]
                    if isinstance(candidate, (nn.Conv2d, nn.MaxPool2d))
                ),
                None,
            )
            result[module] = int(isinstance(next_conv_or_pool, nn.MaxPool2d))
    return result


def _to_bit_matrix(values: torch.Tensor, bits: int) -> np.ndarray:
    values = values.detach().cpu().to(torch.float32).clamp(-1.0, 1.0 - 2 ** (1 - bits))
    scaled = torch.round(values * (2 ** (bits - 1))).to(torch.int64)
    encoded = scaled & ((1 << bits) - 1)
    bit_planes = [((encoded >> shift) & 1).numpy() for shift in reversed(range(bits))]
    stacked = np.stack(bit_planes, axis=-1)
    return stacked.reshape(values.shape[0], -1)


def create_layer_artifacts(
    model: VGG11Cifar10,
    destination: Path,
    activation_bits: int = 8,
) -> LayerArtifacts:
    destination.mkdir(parents=True, exist_ok=True)
    pooling = _pooling_map(model)
    captured: list[tuple[str, nn.Module, torch.Tensor]] = []
    handles = []

    def capture(name: str):
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
            captured.append((name, module, inputs[0].detach().cpu()))

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(capture(name)))
    with torch.inference_mode():
        model(sample_input())
    for handle in handles:
        handle.remove()

    records: list[LayerRecord] = []
    weight_files: list[Path] = []
    input_files: list[Path] = []
    for index, (name, module, inputs) in enumerate(captured):
        weight_path = destination / f"weight_{index:02d}.csv"
        input_path = destination / f"input_{index:02d}.csv"
        if isinstance(module, nn.Conv2d):
            _, channels, height, width = inputs.shape
            kernel_height, kernel_width = module.kernel_size
            stride = int(module.stride[0])
            record = LayerRecord(
                name,
                int(height),
                int(width),
                int(channels),
                int(kernel_height),
                int(kernel_width),
                int(module.out_channels),
                pooling[module],
                stride,
                0,
            )
            unfolded = functional.unfold(
                inputs,
                kernel_size=module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )[0]
            activation_matrix = _to_bit_matrix(unfolded, activation_bits)
        elif isinstance(module, nn.Linear):
            record = LayerRecord(
                name,
                1,
                1,
                int(module.in_features),
                1,
                1,
                int(module.out_features),
                0,
                1,
                1,
            )
            activation_matrix = _to_bit_matrix(inputs.T, activation_bits)
        else:
            raise TypeError(f"Unsupported layer: {type(module)}")

        weight_matrix = module.weight.detach().cpu().numpy().reshape(module.weight.shape[0], -1).T
        np.savetxt(weight_path, weight_matrix, delimiter=",", fmt="%.7f")
        np.savetxt(input_path, activation_matrix, delimiter=",", fmt="%d")
        records.append(record)
        weight_files.append(weight_path.resolve())
        input_files.append(input_path.resolve())

    return LayerArtifacts(tuple(records), tuple(weight_files), tuple(input_files))


def write_network_csv(records: tuple[LayerRecord, ...], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream)
        writer.writerows(record.csv_row() for record in records)
    return destination.resolve()
