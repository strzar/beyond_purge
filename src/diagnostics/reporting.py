"""Report generation for reward and training diagnostics."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import compute_distribution_stats, compute_reward_diagnostics, extract_forget_terms, get_reward_weight_map
from .training import evaluate_checkpoints, evaluate_model_on_prompts


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(output_dir: Path, bundle: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    weight_diag = bundle.get("weight_diagnostics", {})
    if weight_diag.get("status") == "ok":
        weights = bundle["weight_values"]
        plt.figure(figsize=(8, 4))
        plt.hist(weights, bins=min(20, max(5, len(weights) // 2)))
        plt.title("Weight Distribution")
        plt.xlabel("Weight")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / "weight_histogram.png", dpi=160)
        plt.close()

    reward_diag = bundle.get("reward_diagnostics", {})
    probes = reward_diag.get("probes", [])
    if probes:
        xs = list(range(len(probes)))
        ys = [reward_diag["reward_by_case"][probe["name"]] for probe in probes]
        plt.figure(figsize=(9, 4))
        plt.plot(xs, ys, marker="o")
        plt.xticks(xs, [probe["name"] for probe in probes], rotation=30, ha="right")
        plt.title("Reward Probes")
        plt.ylabel("Reward")
        plt.tight_layout()
        plt.savefig(output_dir / "reward_curve.png", dpi=160)
        plt.close()

    checkpoint_diag = bundle.get("checkpoint_diagnostics", [])
    if checkpoint_diag:
        plt.figure(figsize=(9, 4))
        plt.plot(
            [row["checkpoint"] for row in checkpoint_diag],
            [row["avg_reward"] for row in checkpoint_diag],
            marker="o",
        )
        plt.xticks(rotation=30, ha="right")
        plt.title("Checkpoint Average Reward")
        plt.ylabel("Avg Reward")
        plt.tight_layout()
        plt.savefig(output_dir / "checkpoint_avg_reward.png", dpi=160)
        plt.close()


def build_diagnostics_bundle(
    reward_class: Any,
    reward_func: Any,
    prompts: Sequence[Any],
    tokenizer: Any,
    model: Any,
    diagnostics_cfg: Any,
    output_dir: str | Path,
    run_seed: int | None = None,
) -> Dict[str, Any]:
    """Build a standardized diagnostics bundle for a single run."""
    output_dir = Path(output_dir)
    terms = extract_forget_terms(reward_class)
    weight_map = get_reward_weight_map(reward_class)
    weight_diagnostics = reward_class.get_weight_diagnostics()
    reward_diagnostics = compute_reward_diagnostics(reward_func, terms, weight_map)

    generation_kwargs = {
        "max_new_tokens": int(getattr(diagnostics_cfg, "max_new_tokens", 64)),
        "do_sample": bool(getattr(diagnostics_cfg, "do_sample", False)),
    }
    if getattr(diagnostics_cfg, "do_sample", False) and hasattr(diagnostics_cfg, "temperature"):
        generation_kwargs["temperature"] = float(getattr(diagnostics_cfg, "temperature", 0.0))

    eval_prompts = list(prompts)[: int(getattr(diagnostics_cfg, "eval_sample_size", min(len(prompts), 16)))]
    final_eval = evaluate_model_on_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=eval_prompts,
        reward_func=reward_func,
        forget_terms=terms,
        generation_kwargs=generation_kwargs,
    )

    tokenizer_name_or_path = (
        getattr(tokenizer, "name_or_path", None)
        or getattr(tokenizer, "_name_or_path", None)
        or ""
    )
    checkpoint_diagnostics = evaluate_checkpoints(
        output_dir=output_dir,
        tokenizer_name_or_path=tokenizer_name_or_path,
        prompts=eval_prompts,
        reward_func=reward_func,
        forget_terms=terms,
        generation_kwargs=generation_kwargs,
    ) if getattr(diagnostics_cfg, "enable_checkpoint_eval", True) and tokenizer_name_or_path else []

    bundle = {
        "reward_family": reward_class.__name__,
        "forget_terms": terms,
        "weight_map": weight_map,
        "weight_diagnostics": weight_diagnostics,
        "weight_values": list(weight_map.values()),
        "reward_diagnostics": reward_diagnostics,
        "final_eval": final_eval,
        "checkpoint_diagnostics": checkpoint_diagnostics,
        "generation_kwargs": generation_kwargs,
        "diagnostics_config": {
            "probe_size": int(getattr(diagnostics_cfg, "probe_size", 8)),
            "eval_sample_size": int(getattr(diagnostics_cfg, "eval_sample_size", 16)),
            "seed_sweep": list(getattr(diagnostics_cfg, "seed_sweep", [0, 1, 2])),
            "seed": run_seed,
        },
    }
    return bundle


def write_diagnostics_bundle(output_dir: str | Path, bundle: Dict[str, Any]) -> None:
    """Write diagnostics artifacts to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "diagnostics.json", bundle)

    reward_rows = [
        {"case": case["name"], "completion": case["completion"], "reward": bundle["reward_diagnostics"]["reward_by_case"][case["name"]]}
        for case in bundle["reward_diagnostics"].get("probes", [])
    ]
    _write_csv(output_dir / "reward_curves.csv", reward_rows)

    weight_diag = bundle.get("weight_diagnostics", {})
    if weight_diag.get("status") == "ok":
        weight_rows = [
            {"term": term, "weight": weight}
            for term, weight in bundle.get("weight_map", {}).items()
        ]
    else:
        weight_rows = [{"status": weight_diag.get("status", "n/a"), "reason": weight_diag.get("reason", "n/a")}]
    _write_csv(output_dir / "weight_stats.csv", weight_rows)

    training_rows = [
        {
            "num_prompts": bundle["final_eval"]["num_prompts"],
            "hit_rate": bundle["final_eval"]["hit_rate"],
            "avg_reward": bundle["final_eval"]["avg_reward"],
            "reward_variance": bundle["final_eval"]["reward_variance"],
            "mean_output_length": bundle["final_eval"]["mean_output_length"],
            "empty_rate": bundle["final_eval"]["empty_rate"],
        }
    ]
    _write_csv(output_dir / "training_metrics.csv", training_rows)
    _write_plots(output_dir, bundle)


def aggregate_seed_reports(report_dirs: Sequence[str | Path]) -> Dict[str, Any]:
    """Aggregate multiple run diagnostics into seed-level summary statistics."""
    rows: List[Dict[str, Any]] = []
    for report_dir in report_dirs:
        report_dir = Path(report_dir)
        payload_path = report_dir / "diagnostics.json"
        if not payload_path.exists():
            continue
        with payload_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        final_eval = payload.get("final_eval", {})
        rows.append(
            {
                "reward_family": payload.get("reward_family"),
                "seed": payload.get("diagnostics_config", {}).get("seed"),
                "hit_rate": final_eval.get("hit_rate"),
                "avg_reward": final_eval.get("avg_reward"),
                "reward_variance": final_eval.get("reward_variance"),
                "mean_output_length": final_eval.get("mean_output_length"),
                "empty_rate": final_eval.get("empty_rate"),
            }
        )

    summary: Dict[str, Any] = {"runs": rows, "by_reward_family": {}}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["reward_family"], []).append(row)

    for reward_family, family_rows in grouped.items():
        summary["by_reward_family"][reward_family] = {
            metric: compute_distribution_stats(row.get(metric) for row in family_rows)
            for metric in ["hit_rate", "avg_reward", "reward_variance", "mean_output_length", "empty_rate"]
        }

    return summary
