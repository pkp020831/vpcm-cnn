from __future__ import annotations

from typing import Any

import numpy as np


def dominates(
    left: dict[str, float],
    right: dict[str, float],
    objectives: dict[str, str],
) -> bool:
    no_worse = True
    strictly_better = False
    for name, direction in objectives.items():
        if direction not in {"min", "max"}:
            raise ValueError(f"Unsupported objective direction: {direction}")
        left_value, right_value = left[name], right[name]
        if direction == "min":
            no_worse &= left_value <= right_value
            strictly_better |= left_value < right_value
        else:
            no_worse &= left_value >= right_value
            strictly_better |= left_value > right_value
    return no_worse and strictly_better


def pareto_front(
    candidates: list[dict[str, Any]],
    objectives: dict[str, str],
) -> list[dict[str, Any]]:
    return [
        candidate
        for index, candidate in enumerate(candidates)
        if not any(
            dominates(other, candidate, objectives)
            for other_index, other in enumerate(candidates)
            if other_index != index
        )
    ]


def topsis_rank(
    candidates: list[dict[str, Any]],
    objectives: dict[str, str],
    weights: dict[str, float],
) -> list[dict[str, Any]]:
    if not candidates:
        raise ValueError("TOPSIS requires at least one candidate")
    names = tuple(objectives)
    if set(weights) != set(names) or any(value <= 0 for value in weights.values()):
        raise ValueError("TOPSIS weights must be positive and match the objectives")
    matrix = np.asarray([[float(candidate[name]) for name in names] for candidate in candidates])
    denominators = np.linalg.norm(matrix, axis=0)
    denominators[denominators == 0] = 1.0
    normalized_weights = np.asarray([weights[name] for name in names], dtype=float)
    normalized_weights /= normalized_weights.sum()
    weighted = matrix / denominators * normalized_weights
    ideal_best = np.asarray([
        weighted[:, index].min() if objectives[name] == "min" else weighted[:, index].max()
        for index, name in enumerate(names)
    ])
    ideal_worst = np.asarray([
        weighted[:, index].max() if objectives[name] == "min" else weighted[:, index].min()
        for index, name in enumerate(names)
    ])
    distance_best = np.linalg.norm(weighted - ideal_best, axis=1)
    distance_worst = np.linalg.norm(weighted - ideal_worst, axis=1)
    denominator = distance_best + distance_worst
    closeness = np.divide(distance_worst, denominator, out=np.ones_like(denominator), where=denominator != 0)
    ranked = [dict(candidate, topsis_closeness=float(closeness[index])) for index, candidate in enumerate(candidates)]
    ranked.sort(key=lambda candidate: candidate["topsis_closeness"], reverse=True)
    for rank, candidate in enumerate(ranked, start=1):
        candidate["topsis_rank"] = rank
    return ranked
