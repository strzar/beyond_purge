"""Utilities for computing completion length statistics."""
from typing import Dict, List
import numpy as np


def compute_length_stats(texts: List[str], tokenizer) -> Dict[str, float]:
    """Compute min/mean/max token length across a list of completions.
    
    Args:
        texts: List of completion strings.
        tokenizer: Huggingface tokenizer to compute token lengths.
    
    Returns:
        Dict with keys: 'min_length', 'mean_length', 'max_length'.
    """
    if not texts:
        return {'min_length': 0.0, 'mean_length': 0.0, 'max_length': 0.0}
    
    lengths = []
    for text in texts:
        # Only count non-empty, non-NOANSWER completions
        if text and text.strip() and text.strip() != 'NOANSWER':
            tokens = tokenizer.encode(text, add_special_tokens=False)
            lengths.append(len(tokens))
    
    if not lengths:
        return {'min_length': 0.0, 'mean_length': 0.0, 'max_length': 0.0}
    
    return {
        'min_length': float(np.min(lengths)),
        'mean_length': float(np.mean(lengths)),
        'max_length': float(np.max(lengths)),
    }
