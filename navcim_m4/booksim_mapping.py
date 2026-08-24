from __future__ import annotations

import csv
from dataclasses import dataclass
from math import ceil
from pathlib import Path

from .layers import LayerRecord


@dataclass(frozen=True)
class FloorplanLayer:
    layer: int
    tile_count: int
    start_tile: int
    mesh_rows: int
    mesh_cols: int


def read_floorplan(path: Path) -> tuple[FloorplanLayer, ...]:
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("NeuroSim floorplan contains no layers")
    result = tuple(
        FloorplanLayer(
            layer=int(row["layer"]),
            tile_count=int(row["tile_count"]),
            start_tile=int(row["start_tile"]),
            mesh_rows=int(row["mesh_rows"]),
            mesh_cols=int(row["mesh_cols"]),
        )
        for row in rows
    )
    if any(layer.tile_count < 1 for layer in result):
        raise ValueError("NeuroSim floorplan contains a layer with no tiles")
    return result


def output_activation_bits(record: LayerRecord, activation_bits: int = 8) -> int:
    if record.is_fc:
        return record.output_channels * activation_bits
    output_height = (record.input_height + 2 * record.padding_height - record.kernel_height) // record.stride + 1
    output_width = (record.input_width + 2 * record.padding_width - record.kernel_width) // record.stride + 1
    if record.pool_after:
        output_height //= 2
        output_width //= 2
    return output_height * output_width * record.output_channels * activation_bits


def write_booksim_records(
    floorplan: tuple[FloorplanLayer, ...],
    records: tuple[LayerRecord, ...],
    destination: Path,
    flit_bits: int,
    images_per_second: float,
    clock_hz: float,
) -> tuple[Path, Path]:
    if len(floorplan) != len(records):
        raise ValueError("NeuroSim floorplan and VGG11 layer records differ in length")
    destination.mkdir(parents=True, exist_ok=True)
    cluster = destination / "booksim_cluster.csv"
    element = destination / "booksim_element.csv"
    mesh_nodes = floorplan[0].mesh_rows * floorplan[0].mesh_cols
    if any(layer.mesh_rows * layer.mesh_cols != mesh_nodes for layer in floorplan):
        raise ValueError("NeuroSim floorplan has inconsistent mesh dimensions")

    # A record's location is a consumer tile; its predecessor references it as
    # the BookSim destination. The activation size is the prior layer's output.
    with cluster.open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream)
        writer.writerow(("node", "destination1", "destination2", "op", "location", "type", "activation_size", "injection_rate", "tile_rows", "tile_cols"))
        for index, layer in enumerate(floorplan):
            source = floorplan[index - 1] if index else layer
            bits_per_source = ceil(output_activation_bits(records[index - 1 if index else 0]) / source.tile_count)
            injection_rate = min(1.0, bits_per_source * images_per_second / (flit_bits * clock_hz))
            for tile in range(layer.tile_count):
                location = layer.start_tile + tile
                if location >= mesh_nodes:
                    raise ValueError("NeuroSim tile location exceeds its mesh")
                destination = index + 2 if index + 1 < len(floorplan) else ""
                writer.writerow((index + 1, destination, "", records[index].name, f"{location // layer.mesh_cols}-{location % layer.mesh_cols}", "MAC", bits_per_source, injection_rate, layer.mesh_rows, layer.mesh_cols))

    with element.open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream)
        writer.writerow(("node", "used", "activation_size", "injection_rate"))
    return cluster, element
