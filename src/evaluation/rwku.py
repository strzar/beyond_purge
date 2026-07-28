"""RWKU evaluation orchestrator.

Loads per-entity eval datasets from disk and runs all RWKU metrics,
returning a flat dict suitable for wandb.log().

Metric mapping to Table 1:
  forget/fb      = eval_forget level 1 rouge-l recall   (Forget Set FB ↓)
  forget/qa      = eval_forget level 2 rouge-l recall   (Forget Set QA ↓)
  forget/aa      = eval_forget level 3 rouge-l recall   (Forget Set AA ↓)
  forget/fb_sem  = eval_forget level 1 semantic sim      (Forget Set FB sem ↓)
  forget/qa_sem  = eval_forget level 2 semantic sim      (Forget Set QA sem ↓)
  forget/aa_sem  = eval_forget level 3 semantic sim      (Forget Set AA sem ↓)
  neighbor/fb    = eval_neighbor level 1 rouge-l recall  (Neighbor Set FB ↑)
  neighbor/qa    = eval_neighbor level 2 rouge-l recall  (Neighbor Set QA ↑)
  neighbor/fb_sem = eval_neighbor level 1 semantic sim   (Neighbor Set FB sem ↑)
  neighbor/qa_sem = eval_neighbor level 2 semantic sim   (Neighbor Set QA sem ↑)
  mia/fm_loss    = eval_mia on forget set — loss        (MIA FM ↑)
  mia/rm_loss    = eval_mia on retain set — loss        (MIA RM ↓)
  utility/ga     = eval_mmlu accuracy                   (Utility GA ↑)
  utility/ga_ece = eval_mmlu expected calibration error (Utility GA ECE ↓)
  utility/ra     = eval_bbh exact-match                 (Utility RA ↑)
  utility/tru    = eval_truthfulqa mc2                  (Utility TRU ↑)
  utility/fac    = eval_triviaqa f1                     (Utility FAC ↑)
  utility/flu    = eval_fluency n-gram entropy          (Utility FLU ↑)
  retain_ppl     = exp(mia_rm_loss) — retain perplexity (↓)
  composite      = harmonic mean of unlearn quality and utility (↑)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict
from sklearn.metrics import roc_auc_score
import numpy as _np

import torch

log = logging.getLogger(__name__)


def _load(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def load_rwku_datasets(rwku_dir: str | Path, entity_target: str) -> Dict[str, Any]:
    """Load all RWKU eval JSON files for one entity.

    Args:
        rwku_dir: path to the RWKU data root (contains one subdir per entity).
        entity_target: directory name, e.g. "1_Stephen_King".
    """
    root = Path(rwku_dir) / entity_target
    if not root.exists():
        raise FileNotFoundError(f"RWKU data not found for entity '{entity_target}' at {root}")

    return {
        "forget_level1":  _load(root / "forget_level1.json"),
        "forget_level2":  _load(root / "forget_level2.json"),
        "forget_level3":  _load(root / "forget_level3.json"),
        "neighbor_level1": _load(root / "neighbor_level1.json"),
        "neighbor_level2": _load(root / "neighbor_level2.json"),
        "forget_mia":     _load(root / "forget_mia.json"),
        "retain_mia":     _load(root / "retain_mia.json"),
        "retain_mmlu":    _load(root / "retain_mmlu.json"),
        "retain_bbh":     _load(root / "retain_bbh.json"),
        "truthful":       _load(root / "truthful.json"),
        "triviaqa":       _load(root / "triviaqa.json"),
        "fluency":        _load(root / "fluency.json"),
    }


def run_rwku_evaluation(
    model: Any,
    tokenizer: Any,
    datasets: Dict[str, Any],
    output_dir: str | Path | None = None,
    batch_size: int = 4,
) -> Dict[str, float]:
    """Run all RWKU evaluations and return a flat metrics dict.

    The model is placed in eval mode; training mode is restored afterwards.
    """
    from .eval_forget import eval_forget
    from .eval_neighbor import eval_neighbor
    from .eval_mia import eval_mia
    from .eval_mmlu import eval_mmlu
    from .eval_bbh import eval_bbh
    from .eval_truthfulqa import eval_truthfulqa
    from .eval_triviaqa import eval_triviaqa
    from .eval_fluency import eval_fluency

    was_training = model.training
    model.eval()

    out = Path(output_dir) if output_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    def _path(name: str):
        return str(out / name) if out else None

    metrics: Dict[str, float] = {}

    with torch.no_grad():
        log.info("RWKU eval — forget set")
        fb, qa, aa, sem_fb, sem_qa, sem_aa, forget_length_stats = eval_forget(
            model, tokenizer,
            datasets["forget_level1"], datasets["forget_level2"], datasets["forget_level3"],
            batch_size=batch_size, output_result_dir=_path("forget.json"),
        )
        metrics["eval/forget_fb"] = fb
        metrics["eval/forget_qa"] = qa
        metrics["eval/forget_aa"] = aa
        metrics["eval/forget_fb_sem"] = sem_fb
        metrics["eval/forget_qa_sem"] = sem_qa
        metrics["eval/forget_aa_sem"] = sem_aa
        metrics["eval/forget_min_length"] = forget_length_stats["min_length"]
        metrics["eval/forget_mean_length"] = forget_length_stats["mean_length"]
        metrics["eval/forget_max_length"] = forget_length_stats["max_length"]
        torch.cuda.empty_cache()

        log.info("RWKU eval — neighbor set")
        n_fb, n_qa, sem_n_fb, sem_n_qa, neighbor_length_stats = eval_neighbor(
            model, tokenizer,
            datasets["neighbor_level1"], datasets["neighbor_level2"],
            batch_size=batch_size, output_result_dir=_path("neighbor.json"),
        )
        metrics["eval/neighbor_fb"] = n_fb
        metrics["eval/neighbor_qa"] = n_qa
        metrics["eval/neighbor_fb_sem"] = sem_n_fb
        metrics["eval/neighbor_qa_sem"] = sem_n_qa
        metrics["eval/neighbor_min_length"] = neighbor_length_stats["min_length"]
        metrics["eval/neighbor_mean_length"] = neighbor_length_stats["mean_length"]
        metrics["eval/neighbor_max_length"] = neighbor_length_stats["max_length"]
        torch.cuda.empty_cache()

        log.info("RWKU eval — MIA (forget)")
        fm_loss, fm_zlib, fm_mink20, fm_scores = eval_mia(
            model, tokenizer, datasets["forget_mia"],
            output_result_dir=_path("forget_mia.json"),
        )
        metrics["eval/mia_fm_loss"]   = fm_loss
        metrics["eval/mia_fm_zlib"]   = fm_zlib
        metrics["eval/mia_fm_mink20"] = fm_mink20
        torch.cuda.empty_cache()

        log.info("RWKU eval — MIA (retain)")
        rm_loss, rm_zlib, rm_mink20, rm_scores = eval_mia(
            model, tokenizer, datasets["retain_mia"],
            output_result_dir=_path("retain_mia.json"),
        )
        metrics["eval/mia_rm_loss"]   = rm_loss
        metrics["eval/mia_rm_zlib"]   = rm_zlib
        metrics["eval/mia_rm_mink20"] = rm_mink20
        torch.cuda.empty_cache()

        # MIA AUROC: forget=member (1), retain=non-member (0)
        # Higher loss (less negative) = less memorized = predicted non-member.
        # We negate loss so that higher score → more likely member.
        fm_s = _np.array(fm_scores["loss"])
        rm_s = _np.array(rm_scores["loss"])
        y_true = _np.concatenate([_np.ones(len(fm_s)), _np.zeros(len(rm_s))])
        y_score = _np.concatenate([fm_s, rm_s])
        metrics["eval/mia_auroc"] = float(roc_auc_score(y_true, -y_score))

        # Retain perplexity: exp(mean cross-entropy on retain MIA texts)
        # rm_loss is already mean NLL (negative log-likelihood per token), so exp gives PPL
        metrics["eval/retain_ppl"] = float(_np.exp(-rm_loss))

        log.info("RWKU eval — MMLU (GA)")
        ga, ga_ece = eval_mmlu(
            model, tokenizer, datasets["retain_mmlu"],
            batch_size=batch_size, output_result_dir=_path("mmlu.json"),
        )
        metrics["eval/utility_ga"] = ga
        metrics["eval/utility_ga_ece"] = ga_ece
        torch.cuda.empty_cache()

        log.info("RWKU eval — BBH (RA)")
        ra, bbh_length_stats = eval_bbh(
            model, tokenizer, datasets["retain_bbh"],
            batch_size=batch_size, output_result_dir=_path("bbh.json"),
        )
        metrics["eval/utility_ra"] = ra
        torch.cuda.empty_cache()

        log.info("RWKU eval — TruthfulQA (TRU)")
        tru_mc1, tru = eval_truthfulqa(
            model, tokenizer, datasets["truthful"],
            batch_size=batch_size, output_result_dir=_path("truthful.json"),
        )
        metrics["eval/utility_tru"]     = tru
        metrics["eval/utility_tru_mc1"] = tru_mc1
        torch.cuda.empty_cache()

        log.info("RWKU eval — TriviaQA (FAC)")
        fac_em, fac, triviaqa_length_stats = eval_triviaqa(
            model, tokenizer, datasets["triviaqa"],
            batch_size=batch_size, output_result_dir=_path("triviaqa.json"),
        )
        metrics["eval/utility_fac"]    = fac
        metrics["eval/utility_fac_em"] = fac_em
        torch.cuda.empty_cache()

        log.info("RWKU eval — Fluency (FLU)")
        print("RWKU eval — Fluency (FLU) starting...", flush=True)
        flu, fluency_length_stats = eval_fluency(
            model, tokenizer, datasets["fluency"],
            batch_size=batch_size, output_result_dir=_path("fluency.json"),
        )
        metrics["eval/utility_flu"] = flu
        print("RWKU eval — Fluency (FLU) finished.", flush=True)
        torch.cuda.empty_cache()

        # Aggregate utility length statistics across BBH, TriviaQA, and Fluency
        all_utility_lengths = {
            'min_length': min(bbh_length_stats["min_length"], triviaqa_length_stats["min_length"], fluency_length_stats["min_length"]),
            'mean_length': (bbh_length_stats["mean_length"] + triviaqa_length_stats["mean_length"] + fluency_length_stats["mean_length"]) / 3,
            'max_length': max(bbh_length_stats["max_length"], triviaqa_length_stats["max_length"], fluency_length_stats["max_length"]),
        }
        metrics["eval/utility_min_length"] = all_utility_lengths["min_length"]
        metrics["eval/utility_mean_length"] = all_utility_lengths["mean_length"]
        metrics["eval/utility_max_length"] = all_utility_lengths["max_length"]

        # Composite unlearning score: harmonic mean of forget quality and utility.
        # Forget quality = 1 - mean(ROUGE-L recall on forget levels 1-3); higher is better.
        # Utility = mean of [0,1] utility metrics (excluding fluency whose scale differs).
        forget_quality = 1.0 - _np.mean([
            metrics["eval/forget_fb"],
            metrics["eval/forget_qa"],
            metrics["eval/forget_aa"],
        ])
        utility_mean = _np.mean([
            metrics["eval/utility_ga"],
            metrics["eval/utility_ra"],
            metrics["eval/utility_tru"],
            metrics["eval/utility_fac"],
        ])
        denom = float(forget_quality + utility_mean)
        metrics["eval/composite_score"] = float(
            2.0 * forget_quality * utility_mean / denom if denom > 0.0 else 0.0
        )

        # ── Diagnostic metrics (no extra model calls) ─────────────────────────

        # Forget Quality Index: unified forget score across all 6 forget metrics
        # (3 ROUGE + 3 semantic). A single scalar for ranking pure forget effectiveness.
        fqi = 1.0 - _np.mean([
            metrics["eval/forget_fb"],     metrics["eval/forget_qa"],     metrics["eval/forget_aa"],
            metrics["eval/forget_fb_sem"], metrics["eval/forget_qa_sem"], metrics["eval/forget_aa_sem"],
        ])
        metrics["eval/forget_quality_index"] = float(fqi)

        # Semantic–Lexical Gap: sem_sim − ROUGE per level.
        # Positive = model's outputs are lexically different but semantically similar
        # (paraphrase / evasion failure mode). Ideal value: ≈ 0.
        metrics["eval/forget_sem_gap_fb"]   = float(metrics["eval/forget_fb_sem"]   - metrics["eval/forget_fb"])
        metrics["eval/forget_sem_gap_qa"]   = float(metrics["eval/forget_qa_sem"]   - metrics["eval/forget_qa"])
        metrics["eval/forget_sem_gap_aa"]   = float(metrics["eval/forget_aa_sem"]   - metrics["eval/forget_aa"])
        metrics["eval/forget_sem_gap_mean"] = float(_np.mean([
            metrics["eval/forget_sem_gap_fb"],
            metrics["eval/forget_sem_gap_qa"],
            metrics["eval/forget_sem_gap_aa"],
        ]))

        # MIA Loss Gap: fm_loss − rm_loss (both stored as log-likelihood, i.e. negative values).
        # More negative gap = forget set is less likely than retain set = better unlearning.
        # At BASE this is already slightly negative (~-0.05); good unlearning pushes it toward ~-0.10.
        metrics["eval/mia_loss_gap"] = float(
            metrics["eval/mia_fm_loss"] - metrics["eval/mia_rm_loss"]
        )

        # Composite Specificity: harmonic mean of forget quality and neighbor quality.
        # Captures the forget/neighbor Pareto efficiency in a single scalar.
        neighbor_quality = float(_np.mean([
            metrics["eval/neighbor_fb"],
            metrics["eval/neighbor_qa"],
        ]))
        spec_denom = float(forget_quality + neighbor_quality)
        metrics["eval/composite_specificity"] = float(
            2.0 * forget_quality * neighbor_quality / spec_denom if spec_denom > 0.0 else 0.0
        )

        # Forget/Neighbor Score Ratio: how much the forget score dropped relative to
        # how much neighbor knowledge was retained. Higher = more targeted forgetting.
        # Uses the complement convention so both axes are "higher = better":
        #   forget axis: (1 - forget_score) = forget quality
        #   neighbor axis: neighbor_score    = neighbor quality
        forget_score_mean  = float(_np.mean([metrics["eval/forget_fb"],   metrics["eval/forget_qa"]]))
        neighbor_score_mean = float(_np.mean([metrics["eval/neighbor_fb"], metrics["eval/neighbor_qa"]]))
        metrics["eval/forget_neighbor_ratio"] = float(
            (1.0 - forget_score_mean) / max(neighbor_score_mean, 1e-6)
        )

        # Length Confound Ratio: forget mean output length relative to neighbor mean length.
        # Drops well below 1.0 indicate the method achieves low forget scores by generating
        # very short/empty outputs rather than by true forgetting.
        metrics["eval/length_confound_ratio"] = float(
            metrics["eval/forget_mean_length"] / max(metrics["eval/neighbor_mean_length"], 1.0)
        )

    if was_training:
        model.train()

    return metrics
