"""TrainerCallback that runs RWKU evaluation every N steps during GRPO training."""
from __future__ import annotations

import gc
import logging
import json
import torch
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib

import wandb
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .rwku import run_rwku_evaluation

log = logging.getLogger(__name__)



class RWKUEvalCallback(TrainerCallback):
    """Runs full RWKU evaluation every eval_every_n_steps steps.

    Always runs once at step 0 before any training updates.
    Then runs every eval_every_n_steps steps and on the final step if needed.
    Metrics are logged to W&B at the trainer's current global step so that
    training curves and eval curves share the same x-axis.
    Per-checkpoint JSON results are written to output_dir/step_NNNNN/ when set.
    
    Base model evaluation (step 0) is cached to avoid redundant computation
    across multiple training runs with the same model checkpoint.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        datasets: Dict[str, Any],
        eval_every_n_steps: int,
        max_steps: int,
        batch_size: int = 4,
        output_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        entity: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.eval_every = eval_every_n_steps
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.output_dir = Path(output_dir) if output_dir else None
        self.cache_dir = Path(cache_dir) if cache_dir else Path("models/base_eval_cache")
        self.entity = entity or "unknown"
        self._ran_step_zero_eval = False
        self._ran_final_eval = False
        self._step_zero_cached = False
        self._base_metrics: Optional[Dict[str, Any]] = None  # Cache base metrics for comparative analysis
        
        # Create cache directory if needed
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_identifier(self) -> str:
        """Generate a unique identifier for the current model checkpoint.
        
        Uses the model name from tokenizer config. This should be the same
        for all runs starting from the same base model.
        """
        try:
            # Try to get model name from tokenizer config
            model_name = self.tokenizer.name_or_path
        except AttributeError:
            # Fallback: use model name from model config
            model_name = getattr(self.model.config, "_name_or_path", "unknown")
        
        # Normalize: remove special chars, create short hash to handle long paths
        safe_name = "".join(c for c in model_name if c.isalnum() or c in "-_")
        if len(safe_name) > 100:
            # Hash the full path if it's too long
            hash_digest = hashlib.md5(model_name.encode()).hexdigest()[:8]
            safe_name = safe_name[:80] + "_" + hash_digest
        
        return safe_name

    def _get_cache_path(self, entity: str) -> Path:
        """Get the cache file path for base model evaluation of a given entity."""
        model_id = self._get_model_identifier()
        return self.cache_dir / f"base_eval_{model_id}_{entity}.json"

    def _load_from_cache(self, entity: str) -> Optional[Dict[str, Any]]:
        """Load cached base model evaluation metrics if available.
        
        Returns None if cache doesn't exist or fails to load.
        """
        cache_path = self._get_cache_path(entity)
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, "r") as fh:
                metrics = json.load(fh)
            log.info(f"Loaded base model evaluation from cache: {cache_path}")
            return metrics
        except Exception as e:
            log.warning(f"Failed to load cache from {cache_path}: {e}")
            return None

    def _save_to_cache(self, entity: str, metrics: Dict[str, Any]) -> None:
        """Save base model evaluation metrics to cache."""
        cache_path = self._get_cache_path(entity)
        try:
            with open(cache_path, "w") as fh:
                json.dump(metrics, fh, indent=2)
            log.info(f"Saved base model evaluation to cache: {cache_path}")
        except Exception as e:
            log.warning(f"Failed to save cache to {cache_path}: {e}")

    def _compute_comparative_metrics(self, metrics: Dict[str, Any], base_metrics: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Compute forget gain and utility retention metrics.
        
        Args:
            metrics: Current step metrics
            base_metrics: Base model metrics (step 0) or None
            
        Returns:
            Dict of additional metrics to log
        """
        additional_metrics: Dict[str, float] = {}
        
        if base_metrics is None:
            return additional_metrics
        
        # Forget gain metrics: % improvement (lower is better, so gain = base - current)
        forget_metrics = [("eval/forget_fb", "Forget FB"), ("eval/forget_qa", "Forget QA"), ("eval/forget_aa", "Forget AA")]
        for metric_key, _ in forget_metrics:
            if metric_key in metrics and metric_key in base_metrics:
                base_val = base_metrics[metric_key]
                current_val = metrics[metric_key]
                # Compute percentage point gain (lower is better for forget metrics)
                gain = (base_val - current_val) * 100  # in percentage points
                gain_key = metric_key.replace("eval/", "eval/gain_")
                additional_metrics[gain_key] = gain
        
        # Utility retention metrics: % of base model performance retained
        utility_metrics = [
            ("eval/utility_ga", "MMLU"),
            ("eval/utility_ra", "BBH"),
            ("eval/utility_tru", "TruthfulQA"),
            ("eval/utility_tru_mc1", "TruthfulQA MC1"),
            ("eval/utility_fac", "FactualityQA"),
            ("eval/utility_fac_em", "FactualityQA EM"),
            ("eval/utility_flu", "Fluency"),
        ]
        for metric_key, _ in utility_metrics:
            if metric_key in metrics and metric_key in base_metrics:
                base_val = base_metrics[metric_key]
                current_val = metrics[metric_key]
                if base_val > 0:
                    retention_pct = (current_val / base_val) * 100
                    retention_key = metric_key.replace("eval/", "eval/retention_")
                    additional_metrics[retention_key] = retention_pct
        
        # Neighbor set retention (should be high)
        neighbor_metrics = [("eval/neighbor_fb", "Neighbor FB"), ("eval/neighbor_qa", "Neighbor QA")]
        for metric_key, _ in neighbor_metrics:
            if metric_key in metrics and metric_key in base_metrics:
                base_val = base_metrics[metric_key]
                current_val = metrics[metric_key]
                if base_val > 0:
                    retention_pct = (current_val / base_val) * 100
                    retention_key = metric_key.replace("eval/", "eval/retention_")
                    additional_metrics[retention_key] = retention_pct
        
        # MIA metrics: auroc should increase (move away from 0.5)
        if "eval/mia_auroc" in metrics:
            current_auroc = metrics["eval/mia_auroc"]
            base_auroc = base_metrics.get("eval/mia_auroc", 0.5)
            # Improvement = how much closer to 1.0 (perfect MIA resistance) vs random
            random_baseline = 0.5
            current_improvement = max(0, 1.0 - current_auroc)
            base_improvement = max(0, 1.0 - base_auroc)
            additional_metrics["eval/mia_auroc_improvement"] = max(0, base_improvement - current_improvement) * 100
        
        return additional_metrics

    def _run_eval(self, step: int, model: Any, entity: str = "unknown", use_cache: bool = False, trainer_step: int | None = None) -> None:
        """Run evaluation with comprehensive error handling.

        Args:
            step: Evaluation step (for display and caching)
            model: Model to evaluate
            entity: Entity being evaluated (for caching)
            use_cache: Whether to try loading cached base model eval
            trainer_step: Trainer's global step for W&B logging (if None, uses step)
        """
        if trainer_step is None:
            trainer_step = step

        try:
            print(f"\n{'='*70}")
            print(f"RWKU EVAL START: step={step}/{self.max_steps}")
            print(f"{'='*70}")
            log.info(f"RWKU eval — step {step}/{self.max_steps}")
            
            # Check cache for step 0 if requested
            if step == 0 and use_cache:
                cached_metrics = self._load_from_cache(entity)
                if cached_metrics is not None:
                    log.info(f"Using cached base model evaluation for entity {entity}")
                    metrics = cached_metrics
                    self._step_zero_cached = True
                else:
                    # Not in cache, compute and store
                    print(f"  Computing base model evaluation (not in cache)...")
                    metrics = self._compute_eval(model, step=step)
                    self._save_to_cache(entity, metrics)
            else:
                # Step > 0: always compute fresh evaluation
                print(f"  Computing fresh evaluation for step {step}...")
                metrics = self._compute_eval(model, step=step)
            
            # Store base metrics on first eval for comparative analysis
            if step == 0:
                self._base_metrics = dict(metrics)
            
            # Compute comparative metrics (gain, retention %) if we have base metrics
            if self._base_metrics is not None and step > 0:
                comparative_metrics = self._compute_comparative_metrics(metrics, self._base_metrics)
                metrics.update(comparative_metrics)
                print(f"  Added {len(comparative_metrics)} comparative metrics (gain, retention)")
            
            print(f"  Evaluation computed successfully. Got {len(metrics)} metrics.")
            
            # Write metrics locally so we can inspect them even if W&B upload fails
            if not self._step_zero_cached or step > 0:
                if self.output_dir:
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    metrics_path = self.output_dir / f"step_{step:05d}" / f"metrics_step_{step:05d}.json"
                    try:
                        metrics_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(metrics_path, "w") as fh:
                            json.dump(metrics, fh, indent=2)
                        print(f"  Saved metrics to {metrics_path}")
                    except Exception as e:
                        print(f"  WARNING: Failed to write metrics to disk: {e}")
                        log.exception("Failed to write RWKU metrics to disk")

            # Print a short console marker so remote logs clearly show logging attempts
            cache_note = " (from cache)" if (step == 0 and self._step_zero_cached) else ""
            print(f"RWKU EVAL LOGGING ATTEMPT: step={step} wandb_run_id={getattr(wandb.run, 'id', None)} out={self.output_dir}{cache_note}")
            
            # Log all metric names for debugging (important for verifying define_metric coverage)
            all_metric_names = sorted(metrics.keys())
            print(f"  Total metric keys: {len(metrics)}")
            print(f"  All metric names: {all_metric_names}")

            if wandb.run is not None:
                run_id = getattr(wandb.run, 'id', 'unknown')
                print(f"  W&B run: id={run_id}")
                log.debug(f"Logging RWKU metrics to W&B; eval step={step}, trainer step={trainer_step}, run id={run_id}")
                try:
                    metrics_to_log = {"eval/step": trainer_step, **metrics}
                    print(f"  Calling wandb.log with {len(metrics_to_log)} metrics (eval/step={trainer_step})")
                    wandb.log(metrics_to_log, commit=True)
                    print(f"RWKU: wandb.log succeeded for eval step={step}{cache_note}")
                except Exception as e:
                    print(f"ERROR: wandb.log failed for eval step={step}: {type(e).__name__}: {e}")
                    log.exception("Failed to log RWKU metrics to W&B")
            else:
                print(f"WARNING: wandb.run is None at eval step={step}; skipping W&B log")
                log.warning("No active wandb.run; skipping W&B log for RWKU eval")
            
            print(f"{'='*70}")
            print(f"RWKU EVAL END: step={step} completed")
            print(f"{'='*70}\n")

        except Exception as e:
            print(f"\n{'!'*70}")
            print(f"FATAL ERROR in _run_eval at step {step}:")
            print(f"{type(e).__name__}: {e}")
            print(f"{'!'*70}\n")
            log.exception(f"FATAL ERROR in _run_eval at step {step}")

        finally:
            gc.collect()
            torch.cuda.empty_cache()

    def _compute_eval(self, model: Any, step: int = 0) -> Dict[str, Any]:
        """Perform the actual RWKU evaluation computation with detailed error reporting."""
        try:
            # Unwrap model if it's wrapped in DataParallel or DistributedDataParallel
            unwrapped_model = model
            if hasattr(model, "module"):
                unwrapped_model = model.module
                log.debug(f"Unwrapping {type(model).__name__} to access .module")
                print(f"  [_compute_eval] Unwrapped model: {type(unwrapped_model).__name__}")
            
            # Create output dir for this step if available
            out = None
            if self.output_dir:
                out = str(self.output_dir / f"step_{step:05d}")
                print(f"  [_compute_eval] Output dir: {out}")
            
            print(f"  [_compute_eval] Calling run_rwku_evaluation with batch_size={self.batch_size}...")
            metrics = run_rwku_evaluation(
                model=unwrapped_model,
                tokenizer=self.tokenizer,
                datasets=self.datasets,
                output_dir=out,
                batch_size=self.batch_size,
            )
            print(f"  [_compute_eval] Successfully got {len(metrics)} metrics")
            return metrics
        except Exception as e:
            print(f"\nERROR in _compute_eval at step {step}:")
            print(f"  {type(e).__name__}: {e}\n")
            log.exception(f"Error in _compute_eval at step {step}")
            raise

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not getattr(state, "is_world_process_zero", True):
            return

        # Declare eval/step as the x-axis for all eval/* metrics so they align
        # with training curves regardless of when wandb.log() is called relative
        # to the trainer's own logging calls.
        if wandb.run is not None:
            wandb.define_metric("eval/step")
            wandb.define_metric("eval/*", step_metric="eval/step")

        # Always record a baseline eval before the first update.
        # Try to use cache for step 0 to avoid recomputation across runs.
        self._run_eval(0, kwargs.get("model", self.model), entity=self.entity, use_cache=True)
        self._ran_step_zero_eval = True

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        # We want evaluation to pause training across all distributed ranks.
        # Use torch.distributed barriers so non-main ranks wait while rank 0 runs eval.
        is_main = getattr(state, "is_world_process_zero", True)
        step = state.global_step
        # Store trainer's current step for W&B logging (W&B uses this internal counter)
        self._trainer_step = state.global_step

        # Check distributed availability
        try:
            import torch.distributed as dist
            dist_available = dist.is_available() and dist.is_initialized()
        except Exception:
            dist_available = False

        if dist_available:
            # First barrier: ensure all ranks reach this point before rank0 starts eval
            dist.barrier()

        if not is_main:
            # Non-main ranks: wait for evaluation to finish (second barrier), then return
            if dist_available:
                dist.barrier()
            return

        # At this point we are on the main rank and all ranks are synchronized.
        if self.eval_every <= 0:
            print(f"[on_step_end] step={step}: eval_every={self.eval_every} <= 0, skipping")
            if dist_available:
                # let non-main ranks proceed
                dist.barrier()
            return
        if step == 0 and self._ran_step_zero_eval:
            print(f"[on_step_end] step={step}: already ran step 0 eval, skipping")
            if dist_available:
                dist.barrier()
            return
        if step % self.eval_every != 0 and step < self.max_steps:
            if step % 100 == 0:  # log every 100 steps to avoid spam
                print(f"[on_step_end] step={step}: {step} % {self.eval_every} = {step % self.eval_every} (not eval step), skipping")
            if dist_available:
                dist.barrier()
            return

        print(f"[on_step_end] FIRING EVAL at step={step} (eval_every={self.eval_every}, max_steps={self.max_steps})")
        # Get model from trainer kwargs (if available) or use stored reference
        eval_model = kwargs.get("model", self.model)
        log.info(f"Evaluating model at step {step} (model id: {id(eval_model)})")

        self._run_eval(step, eval_model, trainer_step=step)

        # After evaluation completes, signal other ranks to continue
        if dist_available:
            dist.barrier()

        if step >= self.max_steps:
            self._ran_final_eval = True

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        # Safety net: always evaluate at the true final step, even if training
        # was interrupted early or max_steps does not divide eval_every evenly.
        if self.eval_every <= 0:
            return
        if not getattr(state, "is_world_process_zero", True):
            return
        if self._ran_final_eval:
            return
        step = state.global_step
        if step % self.eval_every != 0:
            self._run_eval(step, kwargs.get("model", self.model), trainer_step=step)
