"""Per-step forget-set coverage tracker for GRPO training completions.

Taps the actual completions passed to the reward function every training step
(not probe-generated completions) to estimate, via Monte Carlo, the probability
that the model's generation distribution covers different subsets of the forget set:

  entity_set      — the target entity name only  (H1 ablation axis)
  non_entity_set  — all other forget terms        (H3 ablation axis)
  full_set        — union of the above

Because GRPO generates num_generations completions per prompt per step, we also
compute per-group statistics (mean/std of hit rate across groups).  Low std of
group_p_any indicates collapsed reward variance — the primary H2 diagnostic.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Term matching
# ---------------------------------------------------------------------------

def _compile(term: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)


def _hits_any(text: str, patterns: List[re.Pattern]) -> bool:
    return any(p.search(text) for p in patterns)


def _hit_count(text: str, patterns: List[re.Pattern]) -> int:
    return sum(1 for p in patterns if p.search(text))


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class ForgetCoverageTracker:
    """Track forget-set coverage in GRPO training completions.

    Always operates against the *full* fts.json regardless of the
    entity.forget_words_filename used for training, so ablation experiments
    (entity_only, no_entity) remain comparable to the baseline.

    Args:
        full_fts_path: Path to the canonical fts.json for the entity.
        num_generations: GRPO num_generations (used to reshape the flat
            completions list into groups for within-group variance stats).
    """

    def __init__(self, full_fts_path: str | Path, num_generations: int) -> None:
        full_fts_path = Path(full_fts_path)
        with open(full_fts_path, encoding="utf-8") as f:
            fts: List[str] = json.load(f)

        if not fts:
            raise ValueError(f"fts.json is empty: {full_fts_path}")

        entity_name = fts[0]
        non_entity = [t for t in fts[1:] if t != entity_name]

        self.entity_name = entity_name
        self.entity_set: List[str] = [entity_name]
        self.non_entity_set: List[str] = non_entity
        self.full_set: List[str] = fts

        # Pre-compile patterns (done once, reused every step)
        self._entity_pats: List[re.Pattern] = [_compile(entity_name)]
        self._non_entity_pats: List[re.Pattern] = [_compile(t) for t in non_entity]
        self._full_pats: List[re.Pattern] = [_compile(t) for t in fts]

        self.num_generations = max(1, num_generations)

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze_batch(self, completions: Sequence[str]) -> Dict[str, float]:
        """Compute coverage metrics for one reward-function call (one training step).

        Returns a flat dict suitable for wandb.log, keyed under
        ``train/forget_coverage/``.
        """
        n = len(completions)
        if n == 0:
            return {}

        # Per-completion flags
        hit_entity      = np.zeros(n, dtype=np.float32)
        hit_non_entity  = np.zeros(n, dtype=np.float32)
        hit_count_arr   = np.zeros(n, dtype=np.float32)
        unique_terms_seen: set[str] = set()

        for i, text in enumerate(completions):
            e = _hits_any(text, self._entity_pats)
            ne = _hits_any(text, self._non_entity_pats)
            hit_entity[i] = float(e)
            hit_non_entity[i] = float(ne)
            # Hit count across full set
            hc = _hit_count(text, self._full_pats)
            hit_count_arr[i] = float(hc)
            # Accumulate vocabulary seen
            for term, pat in zip(self.full_set, self._full_pats):
                if pat.search(text):
                    unique_terms_seen.add(term)

        hit_any = np.clip(hit_entity + hit_non_entity, 0, 1)
        hit_entity_only    = hit_entity * (1 - hit_non_entity)
        hit_non_entity_only = hit_non_entity * (1 - hit_entity)
        hit_both           = hit_entity * hit_non_entity

        metrics: Dict[str, float] = {
            "train/forget_coverage/p_any":              float(np.mean(hit_any)),
            "train/forget_coverage/p_entity":           float(np.mean(hit_entity)),
            "train/forget_coverage/p_non_entity":       float(np.mean(hit_non_entity)),
            "train/forget_coverage/p_entity_only":      float(np.mean(hit_entity_only)),
            "train/forget_coverage/p_non_entity_only":  float(np.mean(hit_non_entity_only)),
            "train/forget_coverage/p_both":             float(np.mean(hit_both)),
            "train/forget_coverage/mean_hit_count":     float(np.mean(hit_count_arr)),
            "train/forget_coverage/unique_terms_seen":  float(len(unique_terms_seen)),
            "train/forget_coverage/vocab_coverage":     float(len(unique_terms_seen) / max(1, len(self.full_set))),
        }

        # Per-group stats — reshape into (n_prompts, num_generations)
        # GRPO lays out completions as [prompt0_gen0, prompt0_gen1, ..., prompt1_gen0, ...]
        g = self.num_generations
        if n >= g and n % g == 0:
            n_prompts = n // g
            group_p_any = hit_any.reshape(n_prompts, g).mean(axis=1)
            metrics["train/forget_coverage/group_p_any_mean"] = float(np.mean(group_p_any))
            metrics["train/forget_coverage/group_p_any_std"]  = float(np.std(group_p_any))
            metrics["train/forget_coverage/group_p_any_min"]  = float(np.min(group_p_any))
            metrics["train/forget_coverage/group_p_any_max"]  = float(np.max(group_p_any))

            group_p_entity = hit_entity.reshape(n_prompts, g).mean(axis=1)
            metrics["train/forget_coverage/group_p_entity_mean"] = float(np.mean(group_p_entity))
            metrics["train/forget_coverage/group_p_entity_std"]  = float(np.std(group_p_entity))

            group_p_ne = hit_non_entity.reshape(n_prompts, g).mean(axis=1)
            metrics["train/forget_coverage/group_p_non_entity_mean"] = float(np.mean(group_p_ne))
            metrics["train/forget_coverage/group_p_non_entity_std"]  = float(np.std(group_p_ne))

        return metrics
