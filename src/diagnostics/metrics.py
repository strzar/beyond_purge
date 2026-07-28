"""Shared diagnostic metrics for all reward families."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np


def extract_forget_terms(reward_class: Any) -> List[str]:
    """Extract the ordered forget set from a reward class."""
    terms = getattr(reward_class, "_forget_set", None)
    if terms is None:
        terms = getattr(reward_class, "_forget_words", [])
    return list(terms)


def get_reward_weight_map(reward_class: Any) -> Dict[str, float]:
    """Return the current penalty weight map if the reward class exposes one."""
    weights = getattr(reward_class, "_penalty_weights", None)
    if isinstance(weights, dict):
        return dict(weights)
    return {}


def _compile_matcher(term: str) -> re.Pattern:
    escaped = re.escape(term)
    return re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE)


def detect_term_hits(text: str, terms: Sequence[str]) -> List[str]:
    """Detect matched forget terms in a text."""
    hits = []
    for term in terms:
        if _compile_matcher(term).search(text):
            hits.append(term)
    return hits


def build_reward_probes(terms: Sequence[str], weight_map: Dict[str, float] | None = None) -> List[Dict[str, Any]]:
    """
    Build synthetic completions that probe reward monotonicity and saturation.
    """
    terms = list(terms)
    weight_map = weight_map or {}
    if not terms:
        return [{"name": "clean", "completion": "A neutral completion with no hits."}]

    ordered_terms = sorted(
        terms,
        key=lambda term: weight_map.get(term, 0.0),
        reverse=True,
    )
    high = ordered_terms[0]
    low = ordered_terms[-1]
    mid = ordered_terms[len(ordered_terms) // 2]

    cases = [
        {"name": "clean", "completion": "A neutral completion with no hits."},
        {"name": "single_high", "completion": f"This mentions {high} once."},
        {"name": "single_low", "completion": f"This mentions {low} once."},
        {"name": "repeat_high", "completion": f"{high} {high} appears twice."},
        {"name": "pair_high_low", "completion": f"This mentions {high} and {low}."},
    ]

    if len(ordered_terms) >= 3:
        cases.append(
            {
                "name": "triple_mixed",
                "completion": f"This mentions {high}, {mid}, and {low}.",
            }
        )

    return cases


def compute_distribution_stats(values: Iterable[float]) -> Dict[str, Any]:
    """Compute standard distribution diagnostics."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "q05": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q95": None,
        }

    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q95": float(np.quantile(arr, 0.95)),
    }


def _gini(values: np.ndarray) -> float:
    if values.size <= 1 or np.sum(values) <= 0:
        return 0.0
    ordered = np.sort(values)
    n = ordered.size
    index = np.arange(1, n + 1)
    return float(np.sum((2 * index - n - 1) * ordered) / (n * np.sum(ordered)))


def compute_reward_diagnostics(
    reward_func: Callable[..., List[float]],
    terms: Sequence[str],
    weight_map: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Compute reward probes, monotonicity, and saturation diagnostics."""
    probes = build_reward_probes(terms, weight_map)
    completions = [probe["completion"] for probe in probes]
    rewards = reward_func(completions)
    reward_by_case = {
        probe["name"]: float(reward)
        for probe, reward in zip(probes, rewards, strict=False)
    }

    ordered_cases = []
    if "clean" in reward_by_case:
        ordered_cases.append(reward_by_case["clean"])
    if "single_low" in reward_by_case:
        ordered_cases.append(reward_by_case["single_low"])
    if "single_high" in reward_by_case:
        ordered_cases.append(reward_by_case["single_high"])
    if "pair_high_low" in reward_by_case:
        ordered_cases.append(reward_by_case["pair_high_low"])
    if "triple_mixed" in reward_by_case:
        ordered_cases.append(reward_by_case["triple_mixed"])

    monotone_nonincreasing = all(
        ordered_cases[i] >= ordered_cases[i + 1] - 1e-12
        for i in range(len(ordered_cases) - 1)
    )

    reward_arr = np.asarray(list(rewards), dtype=np.float64)
    saturation = float(np.mean(reward_arr < 0.05)) if reward_arr.size else 0.0

    return {
        "probes": probes,
        "reward_by_case": reward_by_case,
        "stats": compute_distribution_stats(rewards),
        "monotone_nonincreasing": monotone_nonincreasing,
        "saturation_rate": saturation,
        "gini": _gini(reward_arr),
    }
