from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


VGG11_CONFIG: tuple[int | str, ...] = (
    64,
    "M",
    128,
    "M",
    256,
    256,
    "M",
    512,
    512,
    "M",
    512,
    512,
    "M",
)


class VGG11Cifar10(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 3
        for item in VGG11_CONFIG:
            if item == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue
            out_channels = int(item)
            layers.extend(
                (
                    nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.features(inputs)
        outputs = self.avgpool(outputs)
        return self.classifier(torch.flatten(outputs, 1))


@dataclass(frozen=True)
class ModelInfo:
    name: str = "VGG11_CIFAR10"
    input_name: str = "input"
    output_name: str = "logits"
    input_shape: tuple[int, int, int, int] = (1, 3, 32, 32)
    classes: int = 10


def create_model(seed: int = 117) -> VGG11Cifar10:
    torch.manual_seed(seed)
    model = VGG11Cifar10()
    model.eval()
    return model


def sample_input(seed: int = 117) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(ModelInfo().input_shape, generator=generator)


def export_onnx(model: nn.Module, destination: Path) -> Path:
    info = ModelInfo()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample_input(),
        destination,
        input_names=[info.input_name],
        output_names=[info.output_name],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    return destination
