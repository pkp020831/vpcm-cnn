from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from ..simulators import BookSimResult, NeuroSimNoCConfig, NeuroSimResult

if TYPE_CHECKING:
    from ..meta.models import MetaLearnerBundle


@dataclass(frozen=True)
class OverallPPA:
    latency_ns: float
    dynamic_energy_pj: float
    dynamic_power_mw: float
    leakage_power_mw: float
    total_power_mw: float
    area_um2: float
    booksim_latency_ns: float
    booksim_dynamic_energy_pj: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def combine_ppa(
    neurosim: NeuroSimResult,
    booksim: BookSimResult,
    noc: NeuroSimNoCConfig,
    meta: MetaLearnerBundle | None = None,
) -> OverallPPA:
    booksim_latency_ns = booksim.latency_cycles / noc.clock_hz * 1e9
    # BookSim labels this output "Total Power", but its power module sums only
    # switching/clock terms and reports leakage on a separate line.
    booksim_dynamic_power_w = max(0.0, booksim.total_power_w)
    if meta is None:
        latency_ns = neurosim.latency_ns + booksim_latency_ns
        booksim_dynamic_energy_pj = booksim_dynamic_power_w * booksim_latency_ns * 1000
        dynamic_energy_pj = neurosim.dynamic_energy_pj + booksim_dynamic_energy_pj
    else:
        latency_ns = meta.predict_latency(booksim_latency_ns, neurosim.latency_ns)
        booksim_component_ns = max(0.0, latency_ns - neurosim.latency_ns)
        booksim_dynamic_energy_pj = booksim_dynamic_power_w * booksim_component_ns * 1000
        dynamic_energy_pj = meta.predict_energy(booksim_dynamic_energy_pj, neurosim.dynamic_energy_pj)
    dynamic_power_mw = dynamic_energy_pj / latency_ns if latency_ns else 0.0
    leakage_power_mw = neurosim.leakage_power_uw / 1000 + booksim.leakage_power_w * 1000
    return OverallPPA(
        latency_ns=latency_ns,
        dynamic_energy_pj=dynamic_energy_pj,
        dynamic_power_mw=dynamic_power_mw,
        leakage_power_mw=leakage_power_mw,
        total_power_mw=dynamic_power_mw + leakage_power_mw,
        area_um2=neurosim.area_um2 + booksim.area_m2 * 1e6,
        booksim_latency_ns=booksim_latency_ns,
        booksim_dynamic_energy_pj=booksim_dynamic_energy_pj,
    )


def combine_predicted_layerwise_ppa(
    neurosim: NeuroSimResult,
    booksim: BookSimResult,
    noc: NeuroSimNoCConfig,
    meta: MetaLearnerBundle | None,
    neurosim_layers: list[Any],
    predicted_layers: list[dict[str, Any]],
) -> OverallPPA:
    neuro_by_layer = {int(layer.layer): layer for layer in neurosim_layers}
    paired_neuro = [neuro_by_layer[int(layer["layer"])] for layer in predicted_layers]
    latency_ns = max(0.0, neurosim.latency_ns - sum(layer.latency_ns for layer in paired_neuro))
    dynamic_energy_pj = max(0.0, neurosim.dynamic_energy_pj - sum(layer.dynamic_energy_pj for layer in paired_neuro))
    booksim_latency_ns = 0.0
    booksim_dynamic_energy_pj = 0.0
    for layer in predicted_layers:
        neuro = neuro_by_layer[int(layer["layer"])]
        prediction = layer["prediction"]
        layer_booksim_latency_ns = float(prediction["latency_cycles"]) / noc.clock_hz * 1e9
        if meta is None:
            layer_latency = neuro.latency_ns + layer_booksim_latency_ns
            layer_booksim_energy = float(prediction["dynamic_power_w"]) * layer_booksim_latency_ns * 1000
            layer_energy = neuro.dynamic_energy_pj + layer_booksim_energy
        else:
            layer_latency = meta.predict_latency(layer_booksim_latency_ns, neuro.latency_ns)
            layer_booksim_energy = float(prediction["dynamic_power_w"]) * max(0.0, layer_latency - neuro.latency_ns) * 1000
            layer_energy = meta.predict_energy(layer_booksim_energy, neuro.dynamic_energy_pj)
        latency_ns += layer_latency
        dynamic_energy_pj += layer_energy
        booksim_latency_ns += layer_booksim_latency_ns
        booksim_dynamic_energy_pj += layer_booksim_energy
    dynamic_power_mw = dynamic_energy_pj / latency_ns if latency_ns else 0.0
    leakage_power_mw = neurosim.leakage_power_uw / 1000 + booksim.leakage_power_w * 1000
    return OverallPPA(
        latency_ns=latency_ns,
        dynamic_energy_pj=dynamic_energy_pj,
        dynamic_power_mw=dynamic_power_mw,
        leakage_power_mw=leakage_power_mw,
        total_power_mw=dynamic_power_mw + leakage_power_mw,
        area_um2=neurosim.area_um2 + booksim.area_m2 * 1e6,
        booksim_latency_ns=booksim_latency_ns,
        booksim_dynamic_energy_pj=booksim_dynamic_energy_pj,
    )


def combine_actual_layerwise_ppa(
    neurosim: NeuroSimResult,
    booksim: BookSimResult,
    noc: NeuroSimNoCConfig,
    neurosim_layers: list[Any],
    actual_layers: list[dict[str, Any]],
) -> OverallPPA:
    neuro_by_layer = {int(layer.layer): layer for layer in neurosim_layers}
    paired_neuro = [neuro_by_layer[int(layer["layer"])] for layer in actual_layers]
    latency_ns = max(0.0, neurosim.latency_ns - sum(layer.latency_ns for layer in paired_neuro))
    dynamic_energy_pj = max(0.0, neurosim.dynamic_energy_pj - sum(layer.dynamic_energy_pj for layer in paired_neuro))
    booksim_latency_ns = 0.0
    booksim_dynamic_energy_pj = 0.0
    for layer in actual_layers:
        neuro = neuro_by_layer[int(layer["layer"])]
        layer_booksim_latency_ns = float(layer["latency_cycles"]) / noc.clock_hz * 1e9
        layer_booksim_energy = float(layer["total_power_w"]) * layer_booksim_latency_ns * 1000
        latency_ns += neuro.latency_ns + layer_booksim_latency_ns
        dynamic_energy_pj += neuro.dynamic_energy_pj + layer_booksim_energy
        booksim_latency_ns += layer_booksim_latency_ns
        booksim_dynamic_energy_pj += layer_booksim_energy
    dynamic_power_mw = dynamic_energy_pj / latency_ns if latency_ns else 0.0
    leakage_power_mw = neurosim.leakage_power_uw / 1000 + booksim.leakage_power_w * 1000
    return OverallPPA(
        latency_ns=latency_ns,
        dynamic_energy_pj=dynamic_energy_pj,
        dynamic_power_mw=dynamic_power_mw,
        leakage_power_mw=leakage_power_mw,
        total_power_mw=dynamic_power_mw + leakage_power_mw,
        area_um2=neurosim.area_um2 + booksim.area_m2 * 1e6,
        booksim_latency_ns=booksim_latency_ns,
        booksim_dynamic_energy_pj=booksim_dynamic_energy_pj,
    )
