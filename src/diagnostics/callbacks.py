"""Live diagnostics callback for GRPO training."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, List, Sequence

import torch
import wandb
from transformers import TrainerCallback

from .metrics import compute_reward_diagnostics, detect_term_hits, extract_forget_terms, get_reward_weight_map
from .training import evaluate_model_on_prompts


class DiagnosticsCallback(TrainerCallback):
    """Record live probe metrics during training."""

    def __init__(
        self,
        reward_class: Any,
        model: Any,
        tokenizer: Any,
        probe_prompts: Sequence[Any],
        output_dir: str | Path,
        reward_func: Callable[..., List[float]],
        generation_kwargs: dict[str, Any] | None = None,
        interval: int = 10,
    ) -> None:
        self.reward_class = reward_class
        self.model = model
        self.tokenizer = tokenizer
        self.probe_prompts = list(probe_prompts)
        self.reward_func = reward_func
        self.generation_kwargs = generation_kwargs or {"max_new_tokens": 64, "do_sample": False}
        self.interval = max(1, interval)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "live_diagnostics.jsonl"
        self.forget_terms = extract_forget_terms(reward_class)
        
        # Term frequency tracking
        self.term_frequency_history: dict[int, dict[str, int]] = {}  # step -> {term -> count}
        self.term_hit_count_history: dict[int, int] = {}  # step -> total hits across all probes

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return control
        if state.global_step == 0 or state.global_step % self.interval != 0:
            return control

        model = self.model
        tokenizer = self.tokenizer
        model.eval()
        diagnostics = evaluate_model_on_prompts(
            model=model,
            tokenizer=tokenizer,
            prompts=self.probe_prompts,
            reward_func=self.reward_func,
            forget_terms=self.forget_terms,
            generation_kwargs=self.generation_kwargs,
        )
        model.train()
        record = {"step": int(state.global_step), **(logs or {}), **diagnostics}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        # Track term frequencies across steps
        term_counts: dict[str, int] = {}
        total_hits = 0
        for completion in diagnostics.get("completions", []):
            hit_terms = detect_term_hits(completion, self.forget_terms)
            total_hits += len(hit_terms)
            for term in hit_terms:
                term_counts[term] = term_counts.get(term, 0) + 1
        
        self.term_frequency_history[state.global_step] = term_counts
        self.term_hit_count_history[state.global_step] = total_hits

        if wandb.run is not None:
            rows = []
            for prompt, completion, reward in zip(
                self.probe_prompts,
                diagnostics["completions"],
                diagnostics["rewards"],
                strict=False,
            ):
                hit_terms = detect_term_hits(completion, self.forget_terms)
                rows.append(
                    {
                        "prompt": prompt if isinstance(prompt, str) else json.dumps(prompt),
                        "completion": completion,
                        "reward": float(reward),
                        "hit_count": len(hit_terms),
                        "hit_terms": ", ".join(hit_terms),
                    }
                )

            probe_table = wandb.Table(
                columns=["prompt", "completion", "reward", "hit_count", "hit_terms"],
                data=[[row["prompt"], row["completion"], row["reward"], row["hit_count"], row["hit_terms"]] for row in rows],
            )

            reward_breakdown = compute_reward_diagnostics(
                reward_func=self.reward_func,
                terms=self.forget_terms,
                weight_map=get_reward_weight_map(self.reward_class),
            )
            reward_case_metrics = {
                f"train/reward_probe/{case}": value
                for case, value in reward_breakdown["reward_by_case"].items()
            }
            reward_case_metrics.update(
                {
                    "train/reward_probe/monotone_nonincreasing": float(reward_breakdown["monotone_nonincreasing"]),
                    "train/reward_probe/saturation_rate": float(reward_breakdown["saturation_rate"]),
                    "train/reward_probe/gini": float(reward_breakdown["gini"]),
                }
            )

            # Compute and log term frequency metrics
            term_metrics = self._compute_term_metrics()

            wandb.log(
                {
                    "train/sample_completions": probe_table,
                    **{f"train/live/{key}": value for key, value in diagnostics.items() if key not in {"completions", "rewards"}},
                    **reward_case_metrics,
                    **term_metrics,
                    "train/live/step": int(state.global_step),
                },
                step=int(state.global_step),
            )
        return control

    def _compute_term_metrics(self) -> dict[str, float]:
        """Compute term-level frequency and trend metrics from history."""
        metrics: dict[str, float] = {}
        
        if len(self.term_frequency_history) < 2:
            return metrics
        
        # Get current and previous measurements
        steps = sorted(self.term_frequency_history.keys())
        current_step = steps[-1]
        prev_step = steps[-2] if len(steps) > 1 else steps[0]
        
        current_counts = self.term_frequency_history[current_step]
        prev_counts = self.term_frequency_history[prev_step]
        
        # Total hit counts
        current_total = sum(current_counts.values())
        prev_total = sum(prev_counts.values())
        
        metrics["train/term_total_hits"] = float(current_total)
        
        # Per-term frequency and trends
        for term in self.forget_terms:
            current_freq = current_counts.get(term, 0)
            prev_freq = prev_counts.get(term, 0)
            
            # Frequency (as count per probe set)
            metrics[f"train/term_freq/{term}"] = float(current_freq)
            
            # Trend: change from previous step
            freq_change = current_freq - prev_freq
            metrics[f"train/term_trend/{term}"] = float(freq_change)
            
            # Reduction % (if we have initial data)
            if len(steps) > 5:  # Only compute after a few steps
                initial_counts = self.term_frequency_history[steps[0]]
                initial_freq = initial_counts.get(term, 0)
                if initial_freq > 0:
                    reduction_pct = ((initial_freq - current_freq) / initial_freq) * 100
                    metrics[f"train/term_reduction_pct/{term}"] = float(reduction_pct)
        
        # Compute hardness ranking: terms that appear most frequently are hardest to suppress
        if current_counts:
            sorted_terms = sorted(current_counts.items(), key=lambda x: x[1], reverse=True)
            for rank, (term, count) in enumerate(sorted_terms[:3]):  # Top 3 hardest
                metrics[f"train/hardest_term_rank_{rank+1}"] = float(rank + 1)
                metrics[f"train/hardest_term_rank_{rank+1}_name_freq"] = float(count)
        
        return metrics


class WandbTrainingMetricsCallback(TrainerCallback):
    """Forward Trainer training metrics and reward metadata to W&B."""

    def __init__(self, reward_class: Any) -> None:
        self.reward_class = reward_class
        self.last_step_timestamp = None
        self.cumulative_tokens = 0

    @staticmethod
    def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
        if isinstance(value, dict):
            for key, nested_value in value.items():
                next_prefix = f"{prefix}/{key}" if prefix else str(key)
                WandbTrainingMetricsCallback._flatten(next_prefix, nested_value, out)
        elif isinstance(value, (list, tuple)):
            out[prefix] = json.dumps(value)
        else:
            out[prefix] = value

    def on_train_begin(self, args, state, control, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return control

        if wandb.run is None:
            return control

        summary = {
            "train/reward_function": self.reward_class.__name__,
        }
        diagnostics = self.reward_class.get_weight_diagnostics()
        flattened: dict[str, Any] = {}
        self._flatten("train/reward_diagnostics", diagnostics, flattened)
        summary.update(flattened)
        wandb.run.summary.update(summary)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return control
        if wandb.run is None or not logs:
            return control

        payload = {f"train/{key}": value for key, value in logs.items()}
        payload["train/step"] = int(state.global_step)
        
        # Track throughput and timing metrics
        current_time = time.time()
        time_per_step_ms = None
        if self.last_step_timestamp is not None:
            time_per_step_ms = (current_time - self.last_step_timestamp) * 1000
            payload["train/time_per_step_ms"] = time_per_step_ms
        self.last_step_timestamp = current_time
        
        # Track cumulative tokens
        if "train_batch_size" in logs or hasattr(args, "per_device_train_batch_size"):
            batch_size = logs.get("train_batch_size", args.per_device_train_batch_size)
            # Approximate tokens per step (batch_size * avg_seq_len, assuming ~512 tokens avg)
            # This is a conservative estimate since we don't have exact seq lengths here
            tokens_this_step = batch_size * args.gradient_accumulation_steps * 512
            self.cumulative_tokens += tokens_this_step
            payload["train/cumulative_tokens"] = int(self.cumulative_tokens)
            
            # Compute tokens per second
            if time_per_step_ms is not None and time_per_step_ms > 0:
                tokens_per_second = tokens_this_step / (time_per_step_ms / 1000)
                payload["train/tokens_per_second"] = tokens_per_second
        
        # Track GPU memory if available
        if torch.cuda.is_available():
            try:
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                payload["train/peak_gpu_memory_mb"] = peak_memory_mb
                current_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                payload["train/current_gpu_memory_mb"] = current_memory_mb
            except (RuntimeError, AttributeError):
                pass
        
        wandb.log(payload, commit=False)
        return control
