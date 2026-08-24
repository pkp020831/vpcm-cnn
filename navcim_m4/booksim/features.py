from __future__ import annotations

import random
from dataclasses import asdict, dataclass


FEATURES = (
    "network_size",
    "latency_per_flit",
    "tile_width_m",
    "inject_rate",
    "data_size_flits",
    "num_start_node",
    "num_dest_node",
)
DYNAMIC_FEATURES = FEATURES
AREA_FEATURES = ("network_size", "tile_width_m")
TARGETS = ("latency_cycles", "dynamic_power_w", "area_m2", "leakage_power_w")
TARGET_FEATURES = {
    "latency_cycles": DYNAMIC_FEATURES,
    "dynamic_power_w": DYNAMIC_FEATURES,
    "area_m2": AREA_FEATURES,
    "leakage_power_w": AREA_FEATURES,
}


@dataclass(frozen=True)
class BookSimFeatures:
    network_size: int
    latency_per_flit: int
    tile_width_m: float
    inject_rate: float
    data_size_flits: int
    num_start_node: int
    num_dest_node: int

    def values(self, names: tuple[str, ...] = FEATURES) -> list[float]:
        row = asdict(self)
        return [float(row[name]) for name in names]


def sample_features(seed: int) -> BookSimFeatures:
    generator = random.Random(seed)
    network_size = generator.randint(2, 17)
    nodes = network_size * network_size
    weighted_nodes = generator.choices(range(1, 21), weights=range(20, 0, -1))[0]
    starts, destinations = (1, weighted_nodes) if generator.choice((True, False)) else (weighted_nodes, 1)
    if starts + destinations >= nodes:
        starts, destinations = 1, 1
    unit_repeater_latency = generator.choice((3.77283e-07, 3.54362e-07))
    tile_width_m = generator.uniform(0.0001, 0.01)
    clock_hz = 1e9 / ((6.50252e-3) * 20)
    return BookSimFeatures(
        network_size=network_size,
        latency_per_flit=max(1, round(unit_repeater_latency * tile_width_m * clock_hz)),
        tile_width_m=tile_width_m,
        inject_rate=generator.uniform(0.001, 0.3),
        data_size_flits=round(generator.triangular(1, 15_000_000, 1)),
        num_start_node=starts,
        num_dest_node=destinations,
    )
