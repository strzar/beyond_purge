"""Model and checkpoint diagnostics for downstream training analysis."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import compute_distribution_stats, detect_term_hits


def _format_prompt(prompt: Any, tokenizer: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    return str(prompt)


def _generate_completion(
    model: Any,
    tokenizer: Any,
    prompt: Any,
    generation_kwargs: Dict[str, Any],
) -> str:
    device = next(model.parameters()).device
    prompt_text = _format_prompt(prompt, tokenizer)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    generation_kwargs = dict(generation_kwargs)
    generation_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id or tokenizer.eos_token_id)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    completion_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def evaluate_model_on_prompts(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Any],
    reward_func: Callable[..., List[float]],
    forget_terms: Sequence[str],
    generation_kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Evaluate a model on a prompt probe set and compute downstream diagnostics."""
    generation_kwargs = generation_kwargs or {}
    completions = [_generate_completion(model, tokenizer, prompt, generation_kwargs) for prompt in prompts]
    non_empty = [c if c.strip() else " " for c in completions]
    rewards = reward_func(non_empty)
    hit_counts = [len(detect_term_hits(text, forget_terms)) for text in completions]
    hit_rate = float(np.mean([count > 0 for count in hit_counts])) if hit_counts else 0.0
    empty_rate = float(np.mean([len(text.strip()) == 0 for text in completions])) if completions else 0.0
    lengths = [len(text) for text in completions]

    return {
        "num_prompts": len(prompts),
        "hit_rate": hit_rate,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "reward_variance": float(np.var(rewards)) if rewards else 0.0,
        "mean_output_length": float(np.mean(lengths)) if lengths else 0.0,
        "empty_rate": empty_rate,
        "reward_min": float(np.min(rewards)) if rewards else 0.0,
        "reward_max": float(np.max(rewards)) if rewards else 0.0,
        "rewards": [float(value) for value in rewards],
        "completions": completions,
    }


def evaluate_checkpoints(
    output_dir: str | Path,
    tokenizer_name_or_path: str,
    prompts: Sequence[Any],
    reward_func: Callable[..., List[float]],
    forget_terms: Sequence[str],
    generation_kwargs: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Evaluate all saved checkpoints and compute checkpoint-to-checkpoint stability."""
    output_dir = Path(output_dir)
    generation_kwargs = generation_kwargs or {}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    checkpoint_dirs = sorted(
        [path for path in output_dir.glob("checkpoint-*") if path.is_dir()],
        key=lambda path: int(path.name.split("-")[-1]),
    )

    results: List[Dict[str, Any]] = []
    prev_rewards: List[float] | None = None
    prev_hits: List[bool] | None = None

    for checkpoint_dir in checkpoint_dirs:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
        model.eval()
        diagnostics = evaluate_model_on_prompts(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            reward_func=reward_func,
            forget_terms=forget_terms,
            generation_kwargs=generation_kwargs,
        )
        diagnostics["checkpoint"] = checkpoint_dir.name

        if prev_rewards is not None:
            current_rewards = diagnostics["rewards"]
            reward_delta = np.abs(np.asarray(prev_rewards) - np.asarray(current_rewards))
            diagnostics["checkpoint_reward_delta_mean"] = float(np.mean(reward_delta))
            diagnostics["checkpoint_reward_stability"] = float(1.0 - np.mean(reward_delta))
            prev_reward_hits = np.asarray(prev_hits, dtype=np.float64)
            current_hits = np.asarray([len(detect_term_hits(text, forget_terms)) > 0 for text in diagnostics["completions"]], dtype=np.float64)
            diagnostics["checkpoint_hit_rate_delta"] = float(np.mean(np.abs(prev_reward_hits - current_hits)))
        else:
            diagnostics["checkpoint_reward_delta_mean"] = None
            diagnostics["checkpoint_reward_stability"] = None
            diagnostics["checkpoint_hit_rate_delta"] = None

        prev_rewards = diagnostics["rewards"]
        prev_hits = [len(detect_term_hits(text, forget_terms)) > 0 for text in diagnostics["completions"]]
        results.append(diagnostics)

    return results
