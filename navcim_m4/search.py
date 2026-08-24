from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

from .graph import validate_with_tvm
from .accuracy import CrossSimConfig, cached_crosssim
from .layers import create_layer_artifacts, write_network_csv
from .models import create_model, export_onnx, load_checkpoint
from .pipeline import score_candidates
from .booksim.inference import OutOfDistributionError, predict_candidate
from .booksim.models import PredictorBundle
from .meta.models import MetaLearnerBundle
from .booksim.ppa import combine_actual_layerwise_ppa, combine_ppa, combine_predicted_layerwise_ppa
from .booksim.ranking import pareto_front, topsis_rank
from .neurosim.analysis import parse_layer_metrics
from .simulators import (
    HardwareConfig,
    NeuroSimResult,
    build_booksim,
    build_neurosim,
    prescribed_hardware_sweep_groups,
    result_dict,
    run_booksim,
    run_neurosim,
    parse_neurosim_noc_config,
)


def _predictor_for_mode(
    booksim_mode: str,
    predictor_bundle: Path | None,
    meta_bundle: Path | None,
) -> PredictorBundle | None:
    if booksim_mode not in {"simulator", "predictor"}:
        raise ValueError("booksim_mode must be simulator or predictor")
    if booksim_mode == "predictor" and predictor_bundle is None:
        raise ValueError("Predictor mode requires a predictor bundle")
    if meta_bundle is not None and booksim_mode != "predictor":
        raise ValueError("Meta bundle requires predictor mode")
    return PredictorBundle(predictor_bundle) if booksim_mode == "predictor" else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_inputs(
    config: HardwareConfig,
    network_csv: Path,
    artifacts,
    neurosim_binary: Path,
    booksim_binary: Path,
    booksim_mode: str,
    predictor_bundle: Path | None,
    meta_bundle: Path | None,
    crosssim_config: CrossSimConfig | None,
    crosssim_samples: int,
    crosssim_runs: int,
) -> dict:
    return {
        "config": asdict(config),
        "network_csv_sha256": _sha256(network_csv),
        "weight_sha256": [_sha256(path) for path in artifacts.weight_files],
        "input_sha256": [_sha256(path) for path in artifacts.input_files],
        "neurosim_binary_sha256": _sha256(neurosim_binary),
        "booksim_binary_sha256": _sha256(booksim_binary),
        "booksim_mode": booksim_mode,
        "predictor_bundle_sha256": _sha256(predictor_bundle) if predictor_bundle else None,
        "meta_bundle_sha256": _sha256(meta_bundle) if meta_bundle else None,
        "crosssim_config": asdict(crosssim_config) if crosssim_config else None,
        "crosssim_samples": crosssim_samples if crosssim_config else None,
        "crosssim_runs": crosssim_runs if crosssim_config else None,
    }


def _load_cached_candidate(path: Path, inputs: dict) -> dict | None:
    if not path.is_file():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if cached.get("schema") != "navcim.search-candidate/v1" or cached.get("inputs") != inputs:
        return None
    return cached.get("result")


def run_search(
    root: Path,
    output_dir: Path,
    sa_values: list[int],
    pe_values: list[int],
    tile_values: list[int],
    workers: int = 4,
    jobs: int = 8,
    checkpoint: Path | None = None,
    crosssim_config: CrossSimConfig | None = None,
    crosssim_samples: int = 100,
    crosssim_runs: int = 1,
    data_dir: Path = Path("data"),
    adc_values: list[int] | None = None,
    cell_values: list[int] | None = None,
    mux_values: list[int] | None = None,
    configs: list[HardwareConfig] | None = None,
    booksim_mode: str = "simulator",
    predictor_bundle: Path | None = None,
    predictor_fallback: bool = True,
    ranking_mode: str = "pareto-topsis",
    ranking_weights: dict[str, float] | None = None,
    validation_top_k: int = 0,
    meta_bundle: Path | None = None,
    allow_predictor_ood: bool = False,
) -> dict:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint(checkpoint)[0] if checkpoint is not None else create_model()
    onnx_path = export_onnx(model, output_dir / "vgg11_cifar10.onnx")
    graph = validate_with_tvm(onnx_path)
    artifacts = create_layer_artifacts(model, output_dir / "layers")
    network_csv = write_network_csv(artifacts.records, output_dir / "vgg11_cifar10.csv")
    neurosim_binary = build_neurosim(root, jobs)
    booksim_binary = build_booksim(root, jobs)
    bundle = _predictor_for_mode(booksim_mode, predictor_bundle, meta_bundle)
    meta = MetaLearnerBundle(meta_bundle) if meta_bundle is not None else None
    if configs is None:
        configs = [
            HardwareConfig(sa, sa, pe, tile, adc, cell, mux)
            for sa, pe, tile, adc, cell, mux in product(sa_values, pe_values, tile_values, adc_values or [5], cell_values or [2], mux_values or [8])
            if tile >= pe
        ]
    if not configs:
        raise ValueError("Search space did not produce a valid configuration")

    results: list[dict] = []
    failures: list[dict] = []
    reused_configs: list[str] = []

    def simulate(config: HardwareConfig) -> dict:
        candidate_dir = output_dir / "candidates" / config.key
        inputs = _candidate_inputs(
            config, network_csv, artifacts, neurosim_binary, booksim_binary,
            booksim_mode, predictor_bundle, meta_bundle, crosssim_config,
            crosssim_samples, crosssim_runs,
        )
        cached = _load_cached_candidate(candidate_dir / "result.json", inputs)
        if cached is not None:
            cached["candidate_reused"] = True
            return cached
        neurosim = run_neurosim(neurosim_binary, network_csv, artifacts, config, candidate_dir)
        booksim_source = "simulator"
        if bundle is None:
            booksim = run_booksim(booksim_binary, root, config, candidate_dir, artifacts)
        else:
            try:
                booksim = predict_candidate(
                    bundle,
                    candidate_dir / "neurosim.log",
                    candidate_dir / "neurosim_floorplan.csv",
                    artifacts,
                    candidate_dir,
                    reject_out_of_distribution=not allow_predictor_ood,
                )
                booksim_source = "predictor"
            except OutOfDistributionError:
                if not predictor_fallback:
                    raise
                booksim = run_booksim(booksim_binary, root, config, candidate_dir, artifacts)
                booksim_source = "simulator-fallback"
        result = result_dict(config, neurosim, booksim)
        result["booksim_source"] = booksim_source
        neurosim_output = (candidate_dir / "neurosim.log").read_text(encoding="utf-8")
        noc = parse_neurosim_noc_config(neurosim_output)
        neurosim_layers = list(parse_layer_metrics(neurosim_output))
        if booksim_source == "predictor":
            predicted_layers = json.loads((candidate_dir / "booksim_prediction.json").read_text(encoding="utf-8"))["layers"]
            overall = combine_predicted_layerwise_ppa(neurosim, booksim, noc, meta, neurosim_layers, predicted_layers)
        elif booksim_source != "predictor":
            actual_layers = json.loads((candidate_dir / "booksim_layers.json").read_text(encoding="utf-8"))
            overall = combine_actual_layerwise_ppa(neurosim, booksim, noc, neurosim_layers, actual_layers)
        else:
            overall = combine_ppa(neurosim, booksim, noc)
        result["overall"] = overall.as_dict()
        if crosssim_config is not None:
            if checkpoint is None:
                raise ValueError("--crosssim requires --checkpoint")
            accuracy = cached_crosssim(
                checkpoint, data_dir, crosssim_samples, 32, crosssim_runs,
                CrossSimConfig(**(crosssim_config.__dict__ | {"adc_bits": config.adc_bits, "cell_bits": config.cell_bits, "rows": config.sa_row, "cols": config.sa_col})),
                output_dir / "crosssim-cache",
            )
            result.update({key: accuracy[key] for key in (
                "digital_accuracy", "crosssim_accuracy_mean", "crosssim_accuracy_std",
                "accuracy_drop_percentage_points", "crosssim_samples", "crosssim_runs",
                "crosssim_config", "checkpoint_sha256", "digital_seconds",
            )})
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "result.json").write_text(json.dumps({
            "schema": "navcim.search-candidate/v1",
            "inputs": inputs,
            "result": result,
        }, indent=2, sort_keys=True), encoding="utf-8")
        result["candidate_reused"] = False
        return result

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as executor:
        futures = {executor.submit(simulate, config): config for config in configs}
        for future in as_completed(futures):
            config = futures[future]
            try:
                result = future.result()
                if result.pop("candidate_reused", False):
                    reused_configs.append(config.key)
                results.append(result)
            except (RuntimeError, ValueError, OSError) as error:
                failures.append({"config": config.key, "error": str(error)})
    if not results:
        details = "; ".join(item["config"] for item in failures)
        raise RuntimeError(f"All simulator candidates failed: {details}")
    ranking_weights = ranking_weights or {
        "latency_ns": 1.0,
        "total_power_mw": 1.0,
        "area_um2": 1.0,
    }
    all_results = list(results)
    validation_indices: list[int] = []
    if ranking_mode == "legacy":
        score_candidates(results)
        results.sort(key=lambda item: item["score"])
        all_results = results
        validation_indices = list(range(min(validation_top_k, len(results))))
    elif ranking_mode == "pareto-topsis":
        objectives = {name: "min" for name in ranking_weights}
        flattened = [
            {
                "candidate_index": index,
                **{name: result["overall"][name] for name in objectives},
            }
            for index, result in enumerate(results)
        ]
        ranked = topsis_rank(pareto_front(flattened, objectives), objectives, ranking_weights)
        ranked_indices = [candidate["candidate_index"] for candidate in ranked]
        remaining = [index for index in range(len(results)) if index not in ranked_indices]
        results = [all_results[index] for index in (*ranked_indices, *remaining)]
        for rank, candidate in enumerate(ranked, start=1):
            results[rank - 1]["pareto"] = True
            results[rank - 1]["topsis_rank"] = rank
            results[rank - 1]["topsis_closeness"] = candidate["topsis_closeness"]
        full_ranked = topsis_rank(flattened, objectives, ranking_weights)
        for candidate in full_ranked:
            all_results[candidate["candidate_index"]]["full_topsis_rank"] = candidate["topsis_rank"]
        validation_indices = [candidate["candidate_index"] for candidate in full_ranked[:min(validation_top_k, len(full_ranked))]]
    else:
        raise ValueError("ranking_mode must be legacy or pareto-topsis")

    validated = []
    final_best = results[0]
    validation_pool = []
    if bundle is not None and validation_top_k:
        validation_pool = [all_results[index]["config"] for index in validation_indices]
        for index in validation_indices:
            result = all_results[index]
            config = HardwareConfig(**result["config"])
            candidate_dir = output_dir / "candidates" / config.key
            prediction = {"booksim": result["booksim"], "overall": result["overall"]}
            actual_booksim = run_booksim(booksim_binary, root, config, candidate_dir, artifacts)
            actual_neurosim = NeuroSimResult(**result["neurosim"])
            neurosim_output = (candidate_dir / "neurosim.log").read_text(encoding="utf-8")
            noc = parse_neurosim_noc_config(neurosim_output)
            neurosim_layers = list(parse_layer_metrics(neurosim_output))
            actual_layers = json.loads((candidate_dir / "booksim_layers.json").read_text(encoding="utf-8"))
            result["prediction"] = prediction
            result["booksim"] = asdict(actual_booksim)
            result["overall"] = combine_actual_layerwise_ppa(actual_neurosim, actual_booksim, noc, neurosim_layers, actual_layers).as_dict()
            result["booksim_source"] = "simulator-validation"
            result["relative_error"] = {
                name: abs(prediction["overall"][name] - result["overall"][name]) / abs(result["overall"][name]) if result["overall"][name] else 0.0
                for name in ranking_weights
            }
            validated.append(config.key)
        validated_results = [all_results[index] for index in validation_indices]
        objectives = {name: "min" for name in ranking_weights}
        validated_flattened = [
            {
                "candidate_index": index,
                **{name: result["overall"][name] for name in objectives},
            }
            for index, result in enumerate(validated_results)
        ]
        validated_ranked = topsis_rank(
            pareto_front(validated_flattened, objectives), objectives, ranking_weights
        )
        for rank, candidate in enumerate(validated_ranked, start=1):
            validated_results[candidate["candidate_index"]]["validated_topsis_rank"] = rank
            validated_results[candidate["candidate_index"]]["validated_topsis_closeness"] = candidate["topsis_closeness"]
        final_best = validated_results[validated_ranked[0]["candidate_index"]]
    report = {
        "model": "VGG11_CIFAR10",
        "input_shape": [1, 3, 32, 32],
        "graph": {
            "conv_layers": graph.conv_layers,
            "linear_layers": graph.linear_layers,
            "relax_functions": graph.relax_functions,
        },
        "workers": max(1, min(workers, 4)),
        "booksim_mode": booksim_mode,
        "predictor_bundle": str(predictor_bundle.resolve()) if predictor_bundle else None,
        "ranking_mode": ranking_mode,
        "ranking_weights": ranking_weights,
        "validated_top_k": validated,
        "validation_pool": validation_pool,
        "meta_bundle": str(meta_bundle.resolve()) if meta_bundle else None,
        "allow_predictor_ood": allow_predictor_ood,
        "best": final_best,
        "predicted_best": results[0] if validated else None,
        "candidates": results,
        "failures": failures,
        "reused_configs": reused_configs,
    }
    (output_dir / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def run_prescribed_hardware_sweep(
    root: Path,
    output_dir: Path,
    workers: int = 4,
    jobs: int = 8,
    checkpoint: Path | None = None,
    crosssim_config: CrossSimConfig | None = None,
    crosssim_samples: int = 100,
    crosssim_runs: int = 1,
    data_dir: Path = Path("data"),
) -> dict:
    """Run independent full-factorial sweeps around the fixed baseline."""
    groups = prescribed_hardware_sweep_groups()
    configs = list(dict.fromkeys(config for group in groups.values() for config in group))
    report = run_search(
        root, output_dir, [], [], [], workers, jobs, checkpoint, crosssim_config,
        crosssim_samples, crosssim_runs, data_dir, configs=configs,
    )
    by_key = {
        (
            item["config"]["sa_row"], item["config"]["sa_col"],
            item["config"]["pe"], item["config"]["tile"],
            item["config"]["adc_bits"], item["config"]["cell_bits"],
            item["config"]["num_col_muxed"],
        ): item
        for item in report["candidates"]
    }
    report["sweep_groups"] = {
        name: [
            by_key[(config.sa_row, config.sa_col, config.pe, config.tile, config.adc_bits, config.cell_bits, config.num_col_muxed)]
            for config in values
            if (config.sa_row, config.sa_col, config.pe, config.tile, config.adc_bits, config.cell_bits, config.num_col_muxed) in by_key
        ]
        for name, values in groups.items()
    }
    report["missing_sweep_configs"] = [
        config.key for config in configs
        if (config.sa_row, config.sa_col, config.pe, config.tile, config.adc_bits, config.cell_bits, config.num_col_muxed) not in by_key
    ]
    (output_dir.resolve() / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report
