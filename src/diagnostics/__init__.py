"""Diagnostics utilities for reward analysis and training evaluation."""

from .callbacks import DiagnosticsCallback, WandbTrainingMetricsCallback
from .forget_coverage import ForgetCoverageTracker
from .metrics import (
    build_reward_probes,
    compute_distribution_stats,
    compute_reward_diagnostics,
    detect_term_hits,
    extract_forget_terms,
    get_reward_weight_map,
)
from .reporting import aggregate_seed_reports, build_diagnostics_bundle, write_diagnostics_bundle
from .training import evaluate_checkpoints, evaluate_model_on_prompts

__all__ = [
    "DiagnosticsCallback",
    "ForgetCoverageTracker",
    "WandbTrainingMetricsCallback",
    "build_diagnostics_bundle",
    "aggregate_seed_reports",
    "build_reward_probes",
    "compute_distribution_stats",
    "compute_reward_diagnostics",
    "detect_term_hits",
    "evaluate_checkpoints",
    "evaluate_model_on_prompts",
    "extract_forget_terms",
    "get_reward_weight_map",
    "write_diagnostics_bundle",
]
