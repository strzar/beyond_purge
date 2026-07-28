"""
PageRank-weighted reward function for GRPO unlearning.
Uses a dedicated text embedding model to build a semantic similarity graph
over forget-set terms and weights penalties by personalized PageRank.
"""
import gc
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

_COUNT_CAP = 3  # max times a single term contributes to penalty per completion

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from .base import RewardConfig, RewardFunction


class PageRankWeightedReward(RewardFunction):
    """
    PageRank-weighted reward function.
    
    - If NO forbidden words are found: reward = 1.0 (perfect)
    - If forbidden words are found: reward = 1.0 - sum(penalty_weights for matched words)
    - The penalty is proportional to the PageRank importance of the matched words
    - Result is clamped to [0, 1]
    
    Higher PageRank words (more semantically connected to the main entity) 
    cause larger penalties when generated.
    
    Requires preprocessing to compute embeddings and PageRank weights.
    
    Extra params (via config.extra_params):
        - weight_transform: str ("max", "softmax", or "rank")
        - softmax_temperature: float (used when weight_transform="softmax")
        - embedding_model_id: str
        - embedding_max_length: int
        - top_k_neighbors: int
        - min_similarity: float
        - penalty_scale: float
    """
    _forget_set: List[str] = []
    _penalty_weights: Dict[str, float] = {}
    _matchers: Dict[str, re.Pattern] = {}
    _preprocessed: bool = False
    _config: Optional[RewardConfig] = None

    _embedding_model_id: str = "BAAI/bge-m3"
    _embedding_model_revision: Optional[str] = "6892b95fed65c899a30896eb40d619ae284d0455"
    _embedding_max_length: int = 512
    _embedding_local_files_only: bool = False
    _top_k_neighbors: int = 8
    _min_similarity: float = 0.25
    _penalty_scale: float = 1.0
    _weight_transform: str = "max"
    _softmax_temperature: float = 0.05
    _rank_strategy: str = "linear"
    _exp_rank_lambda: float = 0.1
    _similarity_matrix: Optional[np.ndarray] = None
    _adjacency_matrix: Optional[np.ndarray] = None
    _pagerank_scores: Dict[str, float] = {}
    _graph_stats: Dict[str, Any] = {}

    @classmethod
    def reset(cls) -> None:
        """Reset PageRank-specific class state."""
        super().reset()
        cls._forget_set = []
        cls._penalty_weights = {}
        cls._matchers = {}
        cls._config = None
        cls._embedding_model_id = "BAAI/bge-m3"
        cls._embedding_model_revision = "6892b95fed65c899a30896eb40d619ae284d0455"
        cls._embedding_max_length = 512
        cls._embedding_local_files_only = False
        cls._top_k_neighbors = 8
        cls._min_similarity = 0.25
        cls._penalty_scale = 1.0
        cls._weight_transform = "max"
        cls._softmax_temperature = 0.05
        cls._rank_strategy = "linear"
        cls._exp_rank_lambda = 0.1
        cls._similarity_matrix = None
        cls._adjacency_matrix = None
        cls._pagerank_scores = {}
        cls._graph_stats = {}

    @classmethod
    def _sanitize_forget_set(cls, terms: List[str]) -> List[str]:
        """Normalize forget-set terms while preserving order."""
        cleaned_terms: List[str] = []
        seen = set()
        for term in terms:
            normalized = term.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            cleaned_terms.append(normalized)
        return cleaned_terms

    @classmethod
    def _format_embedding_text(cls, text: str) -> str:
        """Return text as-is for BGE-M3 dense embedding."""
        return text

    @staticmethod
    def _compile_matcher(term: str) -> re.Pattern:
        """
        Compile a case-insensitive matcher that respects token boundaries
        without relying on \\b, which is brittle for punctuation-heavy terms.
        """
        escaped = re.escape(term)
        return re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE)

    @classmethod
    def _load_embedding_model(cls, model_id: str) -> tuple[Any, Any]:
        """Load the dedicated embedding model and tokenizer."""
        model_source = model_id
        if cls._embedding_local_files_only:
            try:
                snapshot_dir = Path(
                    snapshot_download(
                        repo_id=model_id,
                        revision=cls._embedding_model_revision,
                        local_files_only=True,
                    )
                )
                if not (
                    (snapshot_dir / "model.safetensors").exists()
                    or (snapshot_dir / "model.safetensors.index.json").exists()
                ):
                    snapshots_root = snapshot_dir.parent
                    candidates = [
                        candidate
                        for candidate in snapshots_root.iterdir()
                        if candidate.is_dir()
                        and (
                            (candidate / "model.safetensors").exists()
                            or (candidate / "model.safetensors.index.json").exists()
                        )
                    ]
                    if not candidates:
                        raise OSError(
                            f"Local Hugging Face cache for {model_id} exists, but no "
                            "snapshot with model.safetensors was found."
                        )
                    snapshot_dir = max(candidates, key=lambda path: path.stat().st_mtime)
                model_source = str(snapshot_dir)
            except Exception as exc:
                raise OSError(
                    f"Local Hugging Face cache for {model_id} was not found. "
                    "Run the download step on a machine with network access first."
                ) from exc
        else:
            model_source = snapshot_download(
                repo_id=model_id,
                revision=cls._embedding_model_revision,
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            padding_side="left",
            local_files_only=cls._embedding_local_files_only,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        try:
            model = AutoModel.from_pretrained(
                model_source,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                local_files_only=cls._embedding_local_files_only,
            )
        except OSError as exc:
            if "use_safetensors=True" not in str(exc):
                raise
            raise OSError(
                f"{model_id} could not be loaded with safetensors in this environment. "
                "On CUDA 11.2 / older-PyTorch setups we intentionally avoid torch.load; "
                "please refresh the Hugging Face cache with a safetensors-capable copy of "
                "the model or pin a revision that ships `model.safetensors`."
            ) from exc
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return tokenizer, model

    @classmethod
    def _encode_texts(
        cls,
        texts: List[str],
        tokenizer: Any,
        model: Any,
    ) -> np.ndarray:
        """Batch-encode texts with masked mean pooling and L2 normalization."""
        device = next(model.parameters()).device
        encoded = tokenizer(
            [cls._format_embedding_text(text) for text in texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cls._embedding_max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            outputs = model(**encoded)

        hidden_state = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).to(hidden_state.dtype)
        summed = (hidden_state * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        pooled = F.normalize(pooled, p=2, dim=1)
        return pooled.detach().cpu().numpy()

    @classmethod
    def _calculate_similarity_matrix(cls, embeddings: List[np.ndarray]) -> np.ndarray:
        """Calculate cosine similarity for the forget-set embeddings."""
        embeddings_array = np.asarray(embeddings, dtype=np.float32)
        similarity_matrix = cosine_similarity(embeddings_array)
        np.fill_diagonal(similarity_matrix, -np.inf)
        return similarity_matrix

    @staticmethod
    def _compute_graph_stats(adjacency: np.ndarray) -> Dict[str, Any]:
        """Compute graph-level diagnostics from a weighted adjacency matrix."""
        if adjacency.size == 0:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "density": 0.0,
                "connected_components": 0,
                "largest_component_size": 0,
                "component_sizes": [],
            }

        graph = nx.from_numpy_array(adjacency)
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        density = nx.density(graph) if num_nodes > 1 else 0.0
        components = [len(component) for component in nx.connected_components(graph)]
        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": density,
            "connected_components": len(components),
            "largest_component_size": max(components) if components else 0,
            "component_sizes": sorted(components, reverse=True),
        }

    @classmethod
    def _build_weighted_knn_adjacency(cls, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Build a symmetric weighted kNN adjacency matrix from cosine similarities.

        Only positive similarities above the configured threshold are retained.
        """
        num_nodes = similarity_matrix.shape[0]
        adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        if num_nodes <= 1:
            return adjacency

        k = max(1, min(cls._top_k_neighbors, num_nodes - 1))

        for i in range(num_nodes):
            row = similarity_matrix[i].copy()
            row[i] = -np.inf

            candidates = np.where(row >= cls._min_similarity)[0]
            if candidates.size == 0:
                continue

            sorted_candidates = candidates[np.argsort(row[candidates])[::-1][:k]]
            for j in sorted_candidates:
                weight = float(row[j])
                if weight <= 0.0:
                    continue
                adjacency[i, j] = max(adjacency[i, j], weight)
                adjacency[j, i] = max(adjacency[j, i], weight)

        return adjacency

    @classmethod
    def _transform_pagerank_scores(cls, pagerank_scores: Dict[str, float]) -> Dict[str, float]:
        """Transform raw PageRank scores into penalty weights."""
        if not pagerank_scores:
            return {}

        if cls._weight_transform == "max":
            max_score = max(pagerank_scores.values())
            if max_score <= 0.0:
                return {term: 0.0 for term in pagerank_scores}
            return {term: score / max_score for term, score in pagerank_scores.items()}

        if cls._weight_transform == "softmax":
            temperature = max(cls._softmax_temperature, 1e-8)
            scores = np.asarray(list(pagerank_scores.values()), dtype=np.float64)
            logits = scores / temperature
            logits = logits - np.max(logits)
            probs = np.exp(logits)
            denom = float(np.sum(probs))
            if denom <= 0.0:
                return {term: 0.0 for term in pagerank_scores}
            return {
                term: float(prob / denom)
                for term, prob in zip(pagerank_scores.keys(), probs, strict=False)
            }

        if cls._weight_transform == "rank":
            ordered_terms = sorted(
                pagerank_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if len(ordered_terms) == 1:
                return {ordered_terms[0][0]: 1.0}
            denom = float(len(ordered_terms) - 1)
            return {
                term: 1.0 - (rank / denom)
                for rank, (term, _) in enumerate(ordered_terms)
            }

        if cls._weight_transform == "exp_rank":
            # Assign exp(-λ·rank) by PageRank order, max-normalised so top term = 1.
            ordered_terms = sorted(
                pagerank_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            lam = cls._exp_rank_lambda
            weights = {term: float(np.exp(-lam * rank)) for rank, (term, _) in enumerate(ordered_terms)}
            mx = max(weights.values())
            return {t: v / mx for t, v in weights.items()}

        if cls._weight_transform == "argmax":
            # Weight 1 for the highest-PageRank term, 0 for all others.
            top_term = max(pagerank_scores, key=lambda t: pagerank_scores[t])
            return {term: 1.0 if term == top_term else 0.0 for term in pagerank_scores}

        raise ValueError(f"Unknown weight_transform: {cls._weight_transform}")

    @classmethod
    def _compute_pagerank_weights(cls, similarity_matrix: np.ndarray) -> Dict[str, float]:
        """
        Compute personalized PageRank weights over the weighted similarity graph.
        """
        adjacency = cls._build_weighted_knn_adjacency(similarity_matrix)
        cls._adjacency_matrix = adjacency
        cls._graph_stats = cls._compute_graph_stats(adjacency)
        G = nx.from_numpy_array(adjacency)

        mapping = {i: cls._forget_set[i] for i in range(len(cls._forget_set))}
        G = nx.relabel_nodes(G, mapping)

        start_node = cls._forget_set[0]
        if start_node not in G.nodes():
            raise ValueError(f"Start node '{start_node}' not in graph")

        personalization = {term: 0.0 for term in cls._forget_set}
        personalization[start_node] = 1.0

        pagerank_scores = nx.pagerank(G, weight="weight", personalization=personalization)
        cls._pagerank_scores = dict(pagerank_scores)
        return cls._transform_pagerank_scores(pagerank_scores)

    @classmethod
    def preprocess(cls, config: RewardConfig) -> None:
        """
        Compute PageRank weights from forget-set embeddings.
        """
        cls.reset()
        cls._config = config

        extra_params = config.extra_params or {}
        cls._embedding_model_id = extra_params.get("embedding_model_id", cls._embedding_model_id)
        cls._embedding_model_revision = extra_params.get(
            "embedding_model_revision",
            cls._embedding_model_revision,
        )
        cls._embedding_max_length = int(extra_params.get("embedding_max_length", cls._embedding_max_length))
        cls._embedding_local_files_only = bool(
            extra_params.get("embedding_local_files_only", cls._embedding_local_files_only)
        )
        cls._top_k_neighbors = int(extra_params.get("top_k_neighbors", cls._top_k_neighbors))
        cls._min_similarity = float(extra_params.get("min_similarity", cls._min_similarity))
        cls._penalty_scale = float(extra_params.get("penalty_scale", cls._penalty_scale))
        cls._weight_transform = str(extra_params.get("weight_transform", cls._weight_transform))
        cls._softmax_temperature = float(
            extra_params.get("softmax_temperature", cls._softmax_temperature)
        )
        cls._rank_strategy = str(extra_params.get("rank_strategy", cls._rank_strategy))
        cls._exp_rank_lambda = float(extra_params.get("exp_rank_lambda", cls._exp_rank_lambda))

        if not cls._embedding_model_id:
            raise ValueError("embedding_model_id must be a non-empty string")
        if cls._embedding_model_revision is not None and not str(cls._embedding_model_revision).strip():
            raise ValueError("embedding_model_revision must be a non-empty string or null")
        if cls._embedding_max_length <= 0:
            raise ValueError("embedding_max_length must be positive")
        if not isinstance(cls._embedding_local_files_only, bool):
            raise ValueError("embedding_local_files_only must be a boolean")
        if cls._top_k_neighbors <= 0:
            raise ValueError("top_k_neighbors must be positive")
        if cls._min_similarity < 0.0:
            raise ValueError("min_similarity must be non-negative")
        if cls._penalty_scale <= 0.0:
            raise ValueError("penalty_scale must be positive")
        if cls._weight_transform not in {"max", "softmax", "rank", "exp_rank", "argmax"}:
            raise ValueError("weight_transform must be one of: max, softmax, rank, exp_rank, argmax")
        if cls._weight_transform == "softmax" and cls._softmax_temperature <= 0.0:
            raise ValueError("softmax_temperature must be positive")
        if cls._weight_transform == "rank" and cls._rank_strategy != "linear":
            raise ValueError("rank_strategy currently only supports 'linear'")
        if cls._weight_transform == "exp_rank" and cls._exp_rank_lambda <= 0.0:
            raise ValueError("exp_rank_lambda must be positive")

        forget_set = cls.load_forget_set(config.forget_words_file, config.target_entity, config.forget_set_filter)
        cls._forget_set = cls._sanitize_forget_set(forget_set)
        if not cls._forget_set:
            raise ValueError("Forget set is empty after sanitization")

        print(
            f"[PageRankWeightedReward] Loading embedding model: {cls._embedding_model_id}"
        )
        tokenizer, model = cls._load_embedding_model(cls._embedding_model_id)

        try:
            print(
                f"[PageRankWeightedReward] Computing embeddings for {len(cls._forget_set)} terms..."
            )
            embeddings = cls._encode_texts(cls._forget_set, tokenizer, model)

            print("[PageRankWeightedReward] Building similarity matrix...")
            similarity_matrix = cls._calculate_similarity_matrix(list(embeddings))
            cls._similarity_matrix = similarity_matrix

            print("[PageRankWeightedReward] Computing PageRank weights...")
            cls._penalty_weights = cls._compute_pagerank_weights(similarity_matrix)
            cls._matchers = {
                term: cls._compile_matcher(term)
                for term in cls._forget_set
            }
            cls._preprocessed = True

            print("[PageRankWeightedReward] Preprocessing complete. Top 5 penalty weights:")
            sorted_weights = sorted(cls._penalty_weights.items(), key=lambda item: item[1], reverse=True)
            for term, weight in sorted_weights[:5]:
                print(f"  {term}: {weight:.4f}")
        finally:
            del tokenizer
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def calc_reward(completions: List[str], **kwargs) -> List[float]:
        """
        Smooth PageRank-weighted reward function for GRPO unlearning.
        """
        if not PageRankWeightedReward._preprocessed:
            raise RuntimeError("PageRankWeightedReward.preprocess() must be called before calc_reward()")

        rewards = []
        forget_set = PageRankWeightedReward._forget_set
        penalty_weights = PageRankWeightedReward._penalty_weights
        matchers = PageRankWeightedReward._matchers
        penalty_scale = PageRankWeightedReward._penalty_scale

        for completion in completions:
            total_penalty = 0.0
            for term in forget_set:
                matcher = matchers.get(term)
                if matcher is not None:
                    count = min(len(matcher.findall(completion)), _COUNT_CAP)
                    if count > 0:
                        total_penalty += penalty_weights.get(term, 0.0) * count

            reward = math.exp(-penalty_scale * total_penalty)
            rewards.append(float(reward))

        return rewards

    @classmethod
    def get_weight_diagnostics(cls) -> Dict[str, Any]:
        """Return PageRank weight and graph diagnostics for the current state."""
        if not cls._preprocessed:
            return {
                "status": "n/a",
                "reason": "preprocess_not_run",
            }

        weights = np.asarray(list(cls._penalty_weights.values()), dtype=np.float64)
        if weights.size == 0:
            return {
                "status": "empty",
                "weight_transform": cls._weight_transform,
                "graph": cls._graph_stats,
            }

        sorted_weights = np.sort(weights)
        top_k = min(5, sorted_weights.size)
        top_k_mass = float(np.sum(sorted_weights[-top_k:]) / np.sum(sorted_weights)) if np.sum(sorted_weights) > 0 else 0.0
        probs = weights / np.sum(weights) if np.sum(weights) > 0 else np.zeros_like(weights)
        entropy = float(-np.sum(probs * np.log(probs + 1e-12))) if probs.size else 0.0
        gini = 0.0
        if weights.size > 1 and np.sum(weights) > 0:
            ordered = np.sort(weights)
            n = ordered.size
            index = np.arange(1, n + 1)
            gini = float((np.sum((2 * index - n - 1) * ordered)) / (n * np.sum(ordered)))

        quantiles = {
            "q05": float(np.quantile(weights, 0.05)),
            "q25": float(np.quantile(weights, 0.25)),
            "q50": float(np.quantile(weights, 0.50)),
            "q75": float(np.quantile(weights, 0.75)),
            "q95": float(np.quantile(weights, 0.95)),
        }

        return {
            "status": "ok",
            "weight_transform": cls._weight_transform,
            "min": float(np.min(weights)),
            "max": float(np.max(weights)),
            "mean": float(np.mean(weights)),
            "std": float(np.std(weights)),
            "entropy": entropy,
            "gini": gini,
            "top_5_mass": top_k_mass,
            "quantiles": quantiles,
            "graph": cls._graph_stats,
        }
