"""Semantic similarity evaluation using BAAI/bge-m3 embeddings.

Provides sem_sim_multi(), which encodes multiple (predictions, answers) groups
in a single model-load/unload cycle to minimise GPU memory churn.
"""
from __future__ import annotations

import gc
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_BGE_MODEL_ID = "BAAI/bge-m3"
_BGE_REVISION = "6892b95fed65c899a30896eb40d619ae284d0455"


def _load_bge_m3():
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    model_source = snapshot_download(repo_id=_BGE_MODEL_ID, revision=_BGE_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModel.from_pretrained(
        model_source,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return tokenizer, model


def _encode(texts: List[str], tokenizer, model, batch_size: int = 64) -> np.ndarray:
    """Mean-pool + L2-normalize embeddings for a list of texts."""
    device = next(model.parameters()).device
    parts = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i : i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model(**enc)
        h = out.last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        pooled = F.normalize(pooled, p=2, dim=1)
        parts.append(pooled.detach().cpu().numpy())
    return np.concatenate(parts, axis=0)


def sem_sim_multi(
    groups: List[Tuple[List[str], List[str]]],
) -> List[float]:
    """Compute mean cosine similarity for multiple (predictions, answers) groups.

    Loads BAAI/bge-m3 once, encodes all texts in a single pass, then returns
    one mean similarity per group. Returns 0.0 for any empty group.

    Args:
        groups: list of (predictions, answers) pairs. Lengths within a pair
                must match.

    Returns:
        List of floats, one per group, in [-1, 1] (typically [0, 1] in practice).
    """
    # Collect non-empty groups and their slice boundaries in the flat text list
    all_preds: List[str] = []
    all_ans: List[str] = []
    slices: List[Tuple[int, int] | None] = []  # (start, end) or None for empty
    for preds, answers in groups:
        if not preds or not answers:
            slices.append(None)
        else:
            start = len(all_preds)
            all_preds.extend(preds)
            all_ans.extend(answers)
            slices.append((start, start + len(preds)))

    if not all_preds:
        return [0.0] * len(groups)

    tokenizer, model = _load_bge_m3()
    try:
        pred_embs = _encode(all_preds, tokenizer, model)
        ans_embs = _encode(all_ans, tokenizer, model)
    finally:
        del tokenizer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Cosine similarity: both arrays are already L2-normalised
    cosine_sims = (pred_embs * ans_embs).sum(axis=1)

    results = []
    for sl in slices:
        if sl is None:
            results.append(0.0)
        else:
            results.append(float(np.mean(cosine_sims[sl[0] : sl[1]])))
    return results
