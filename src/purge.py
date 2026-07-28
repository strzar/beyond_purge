"""
GRPO training script for unlearning using modular reward functions.
Configured with Hydra for flexible hyperparameter management.

Usage:
    # Default run (Confucius with PageRank reward)
    python purge.py

    # Different entity
    python purge.py entity=taylor_swift

    # Different reward function
    python purge.py reward=binary

    # Fast testing config
    python purge.py training=fast

    # Override specific parameters
    python purge.py training.num_epochs=20 training.per_device_train_batch_size=4

    # Multi-run sweep
    python purge.py --multirun entity=confucius,taylor_swift,stephen_king
"""
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from trl import GRPOTrainer, GRPOConfig
import weave

import collections
import json
import gc
import random
import os
import subprocess
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb

from rewards import (
    RewardFunction, BinaryReward, PageRankWeightedReward,
    ExponentialDecayReward,
)
from rewards.base import RewardConfig
from diagnostics import (
    DiagnosticsCallback,
    ForgetCoverageTracker,
    WandbTrainingMetricsCallback,
    build_diagnostics_bundle,
    write_diagnostics_bundle,
)
from evaluation import load_rwku_datasets, RWKUEvalCallback


def _get_git_commit_id() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        commit_id = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short=8", "HEAD"],
            text=True,
        ).strip()
        is_dirty = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            text=True,
        ).strip()
        return f"{commit_id}-dirty" if is_dirty else commit_id
    except (FileNotFoundError, subprocess.CalledProcessError):
        return os.getenv("GIT_COMMIT_ID", "nogit")


def _init_wandb(cfg: DictConfig) -> str:
    """Initialize W&B with structured metadata for flexible UI grouping.

    Stores the following config keys so the W&B 'Group by' dropdown can
    pivot on any combination:
      method               e.g. "hazard"
      model                e.g. "Phi-3-mini-4k-instruct"
      num_generations      e.g. 4 or 16
      quant_type           e.g. "bf16" or "none"

    The W&B *group* encodes method + model + quant + num_gen + commit so
    every entity belonging to the same submission batch appears under one group.
    Within that group each run is named  {method}/g{G}/{entity}.
    """
    method_type = cfg.reward.type
    try:
        method_variant = HydraConfig.get().runtime.choices.get("reward", method_type)
    except Exception:
        method_variant = method_type

    method = method_variant
    model  = cfg.model.name
    entity = cfg.entity.target

    quant_cfg  = cfg.get("quantization", {})
    quant_type = quant_cfg.get("type", None) if quant_cfg.get("enabled", False) else "none"
    quant_type = quant_type or "none"

    num_generations = int(cfg.training.num_generations)

    forget_filter = cfg.get("forget_set_filter", "full")

    commit_id = _get_git_commit_id()

    method_model           = f"{method}_{model}"
    method_commit_id       = f"{method}_{commit_id}"
    method_model_commit_id = f"{method}_{model}_{quant_type}_{num_generations}g_{forget_filter}_{commit_id}"

    project = os.getenv("WANDB_PROJECT", "purge")
    wandb_entity = os.getenv("WANDB_ENTITY", None)

    wandb.init(
        entity=wandb_entity,
        project=project,
        name=f"{method}/g{num_generations}/{forget_filter}/{entity}",
        group=method_model_commit_id,
        tags=[method, method_type, model, quant_type, f"g{num_generations}", forget_filter],
        config={
            "method":                method,
            "method_type":           method_type,
            "method_variant":        method_variant,
            "model":                 model,
            "entity":                entity,
            "num_generations":       num_generations,
            "quant_type":            quant_type,
            "forget_filter":         forget_filter,
            "method_model":          method_model,
            "method_commit_id":      method_commit_id,
            "method_model_commit_id": method_model_commit_id,
            "commit_id":             commit_id,
        },
        resume="allow",
    )
    return method_model_commit_id



def get_reward_class(reward_type: str) -> type[RewardFunction]:
    """
    Get the reward class based on the reward type string.
    
    Args:
        reward_type: Either "binary" or "pagerank"
        
    Returns:
        The corresponding RewardFunction subclass
    """
    reward_classes = {
        "binary": BinaryReward,
        "pagerank": PageRankWeightedReward,
        "exponential_decay": ExponentialDecayReward,
    }

    if reward_type not in reward_classes:
        raise ValueError(
            f"Unknown reward type: {reward_type}. "
            f"Available: {list(reward_classes.keys())}"
        )

    return reward_classes[reward_type]


def load_model_and_tokenizer(cfg: DictConfig):
    """Load the model and tokenizer from HuggingFace with optional quantization."""
    print(f"Loading model: {cfg.model.hf_model_id}")

    # Handle quantization
    model_kwargs = {}
    quant_cfg = cfg.get("quantization", {})
    if quant_cfg.get("enabled", False):
        quant_type = quant_cfg.get("type")
        print(f"Applying {quant_type} quantization")
        if quant_type == "bf16":
            print("Weights will stay in fp32 and use bf16 for compute where supported (AMP via GRPOConfig).")
            pass  # weights stay fp32; GRPOConfig bf16=True handles AMP via autocast
        elif quant_type == "fp16":
            raise NotImplementedError("Full FP16 quantization is not implemented in this script. Use bf16 or no quantization.")

    model = AutoModelForCausalLM.from_pretrained(cfg.model.hf_model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_model_id)
    return model, tokenizer


def extract_prompts(records, limit: int, seed: int) -> list:
    """Deterministically sample prompts from the training data."""
    prompts = [record["prompt"] for record in records if "prompt" in record]
    if limit is None or limit >= len(prompts):
        return prompts
    rng = random.Random(seed)
    indices = list(range(len(prompts)))
    rng.shuffle(indices)
    selected = sorted(indices[:limit])
    return [prompts[index] for index in selected]


def initialize_reward(cfg: DictConfig):
    """Initialize and preprocess the configured reward function."""
    reward_class = get_reward_class(cfg.reward.type)
    reward_class.reset()

    extra_params = {k: v for k, v in cfg.reward.items() if k != "type"}
    reward_config = RewardConfig(
        target_entity=cfg.entity.name,
        forget_words_file=cfg.paths.forget_words_file,
        forget_dataset_file=cfg.paths.forget_dataset_file,
        extra_params=extra_params if extra_params else None,
        forget_set_filter=cfg.get("forget_set_filter", "full"),
    )

    print(f"\n{'=' * 60}")
    print(f"Using reward function: {reward_class.__name__}")
    print(f"{'=' * 60}\n")

    reward_class.preprocess(reward_config)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return reward_class, reward_config


def prepare_training_components(cfg: DictConfig):
    """Prepare reward function and policy model in the right order."""
    reward_class, reward_config = initialize_reward(cfg)
    model, tokenizer = load_model_and_tokenizer(cfg)
    return reward_class, reward_config, model, tokenizer


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration."""
    
    # Print resolved configuration
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60 + "\n")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    require_cuda = os.getenv("PURGE_REQUIRE_CUDA", "").lower() in {"1", "true", "yes", "on"}
    if require_cuda and device.type != "cuda":
        cuda_version = torch.version.cuda or "unavailable"
        device_count = torch.cuda.device_count() if hasattr(torch.cuda, "device_count") else 0
        raise SystemExit(
            "ERROR: CUDA is required for this run but is not available.\n"
            f"torch={torch.__version__}\n"
            f"torch.version.cuda={cuda_version}\n"
            f"torch.cuda.device_count()={device_count}\n"
            "Recreate the Leonardo environment with a CUDA 12.1-compatible PyTorch wheel "
            "(see requirements.txt) and rerun the job."
        )
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    random.seed(int(cfg.seed))

    run_tag = _init_wandb(cfg)
    with open_dict(cfg):
        cfg.paths.run_tag = run_tag

    # Load forget dataset
    print(f"Loading dataset from: {cfg.paths.forget_dataset_file}")
    with open(cfg.paths.forget_dataset_file, "r") as f:
        data = json.load(f)

    reward_class, reward_config, model, tokenizer = prepare_training_components(cfg)
    reward_func = reward_class.get_reward_func()

    # Prepare dataset
    dataset = Dataset.from_list(data)
    if cfg.training.dataset_size is not None:
        dataset = dataset.select(range(min(cfg.training.dataset_size, len(dataset))))
    print(f"Dataset size: {len(dataset)} samples")

    live_prompts = extract_prompts(data, int(cfg.diagnostics.probe_size), int(cfg.seed))
    eval_prompts = extract_prompts(data, int(cfg.diagnostics.eval_sample_size), int(cfg.seed))

    # Pre-load RWKU datasets so both the callback and the fallback post-training
    # eval share the same (possibly sliced) data without loading twice.
    rwku_datasets = None
    if cfg.evaluation.enabled:
        print(f"Loading RWKU datasets from: {cfg.evaluation.rwku_dir}")
        rwku_datasets = load_rwku_datasets(cfg.evaluation.rwku_dir, cfg.entity.target)
        max_samples = cfg.evaluation.get("max_samples", None)
        if max_samples is not None:
            print(f"Smoke mode: slicing eval datasets to {max_samples} samples each")
            rwku_datasets = {k: v[:max_samples] for k, v in rwku_datasets.items()}

    callbacks = []
    callbacks.append(WandbTrainingMetricsCallback(reward_class=reward_class))
    if cfg.diagnostics.enable_live:
        callbacks.append(
            DiagnosticsCallback(
                reward_class=reward_class,
                model=model,
                tokenizer=tokenizer,
                probe_prompts=live_prompts,
                output_dir=cfg.diagnostics.output_dir,
                reward_func=reward_func,
                generation_kwargs={
                    "max_new_tokens": int(cfg.diagnostics.max_new_tokens),
                    "do_sample": bool(cfg.diagnostics.do_sample),
                    **(
                        {"temperature": float(cfg.diagnostics.temperature)}
                        if bool(cfg.diagnostics.do_sample)
                        else {}
                    ),
                },
                interval=int(cfg.diagnostics.live_eval_interval),
            )
        )

    max_steps = int(cfg.training.max_steps)
    eval_every_steps = int(cfg.evaluation.get("eval_every_n_steps", 0))
    if rwku_datasets is not None:
        callbacks.append(
            RWKUEvalCallback(
                model=model,
                tokenizer=tokenizer,
                datasets=rwku_datasets,
                eval_every_n_steps=eval_every_steps,
                max_steps=max_steps,
                batch_size=int(cfg.evaluation.batch_size),
                output_dir=cfg.paths.rwku_eval_dir,
                entity=cfg.entity.target,
            )
        )

    # Training configuration
    optim = getattr(cfg.training, "optim", None)
    use_cpu = device.type == "cpu"
    quant_cfg  = cfg.get("quantization", {})
    use_bf16 = (not use_cpu) and quant_cfg.get("enabled", False) and quant_cfg.get("type") == "bf16"
    use_fp16 = (not use_cpu) and quant_cfg.get("enabled", False) and quant_cfg.get("type") == "fp16"
    training_args = GRPOConfig(
        output_dir=cfg.paths.output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_generations=cfg.training.num_generations,
        logging_steps=cfg.training.logging_steps,
        save_strategy=cfg.training.save_strategy,
        **({"save_steps": int(cfg.training.save_steps)} if cfg.training.get("save_steps") else {}),
        save_total_limit=cfg.training.save_total_limit,
        seed=int(cfg.seed),
        use_cpu=use_cpu,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=["wandb"],
        **({"optim": optim} if optim else {}),
    )

    # Canonical fts.json — always the full forget vocabulary, regardless of which
    # filtered variant is used for training. Coverage tracker reads from here so
    # ablation experiments (entity_only, no_entity) remain comparable to baseline.
    _canonical_fts = Path(cfg.paths.purge_dir) / cfg.entity.target / "fts.json"
    _coverage_tracker = ForgetCoverageTracker(
        full_fts_path=_canonical_fts,
        num_generations=int(cfg.training.num_generations),
    )

    # Wrap reward_func to log within-group reward statistics and forget-set
    # coverage metrics every logging_steps.  Coverage is computed every step
    # (pure regex — negligible cost) but logged on the same cadence.
    _logging_steps = int(cfg.training.logging_steps)
    _reward_call_counter = [0]
    _degenerate_history: collections.deque = collections.deque(maxlen=100)

    def _logged_reward_func(completions, **kwargs):
        rewards = reward_func(completions, **kwargs)
        _reward_call_counter[0] += 1
        step = _reward_call_counter[0]
        r = torch.tensor(rewards, dtype=torch.float32)
        std = float(r.std().item())
        is_degenerate = std < 1e-6
        _degenerate_history.append(float(is_degenerate))
        if step % _logging_steps == 0 and wandb.run is not None:
            coverage = _coverage_tracker.analyze_batch(completions)
            wandb.log({
                "train/reward_group_std":    std,
                "train/reward_group_mean":   float(r.mean().item()),
                "train/reward_group_min":    float(r.min().item()),
                "train/reward_group_max":    float(r.max().item()),
                "train/reward_group_range":  float(r.max().item() - r.min().item()),
                "train/degenerate_group":    float(is_degenerate),
                # Rolling fraction of degenerate groups over last 100 reward calls.
                # Key diagnostic: surface-matching rewards collapse; whitened rewards stay alive.
                "train/degenerate_rate_100": float(np.mean(_degenerate_history)),
                **coverage,
            })
        return rewards

    # Create trainer with the modular reward function
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=_logged_reward_func,
        args=training_args,
        train_dataset=dataset,
        callbacks=callbacks if callbacks else None,
    )
    
    print("Started training...")
    trainer.train()
    print("Finished training.")

    if cfg.diagnostics.enable_post_training_bundle:
        diagnostics_bundle = build_diagnostics_bundle(
            reward_class=reward_class,
            reward_func=reward_func,
            prompts=eval_prompts,
            tokenizer=tokenizer,
            model=model,
            diagnostics_cfg=cfg.diagnostics,
            output_dir=cfg.paths.output_dir,
            run_seed=int(cfg.seed),
        )
        write_diagnostics_bundle(cfg.diagnostics.output_dir, diagnostics_bundle)


if __name__ == '__main__':
    main()
