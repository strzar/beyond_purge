"""
Abstract base class for reward functions used in GRPO unlearning.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Set, Optional, Any, Dict
import json


@dataclass
class RewardConfig:
    """Configuration for reward functions."""
    target_entity: str
    forget_words_file: str
    forget_dataset_file: str
    model: Any = None
    tokenizer: Any = None
    extra_params: Optional[Dict[str, Any]] = None
    # Controls which subset of fts.json is used as the forget set:
    #   "full"        — all terms (default)
    #   "entity_only" — only the first term (entity name)
    #   "no_entity"   — all terms except the first
    forget_set_filter: str = "full"


class RewardFunction(ABC):
    """
    Abstract base class for reward functions.
    
    Subclasses must implement:
    - preprocess(): Class method to compute any required preprocessing (e.g., PageRank)
    - calc_reward(): Static method that computes rewards for completions
    
    Usage with GRPOTrainer:
        reward_class = PageRankWeightedReward
        reward_class.preprocess(config)
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_class.calc_reward,
            ...
        )
    """
    
    # Class-level state that gets set during preprocess()
    _forget_words: Set[str] = set()
    _forget_set: List[str] = []
    _config: Optional[RewardConfig] = None
    _preprocessed: bool = False
    
    @classmethod
    def _apply_filter(cls, fts: List[str], target_entity: str, forget_set_filter: str) -> List[str]:
        """Apply forget_set_filter to a raw fts list (entity name is always fts[0])."""
        if forget_set_filter == "entity_only":
            return [target_entity] if target_entity in fts else fts[:1]
        if forget_set_filter == "no_entity":
            return [t for t in fts if t != target_entity]
        # "full" or unrecognised: entity first, then the rest de-duplicated
        rest = [t for t in fts if t != target_entity]
        return [target_entity] + rest

    @classmethod
    def load_forget_words(cls, forget_words_file: str, forget_set_filter: str = "full") -> Set[str]:
        """Load forget words from fts.json, applying forget_set_filter."""
        with open(forget_words_file, 'r') as f:
            fts = json.load(f)
        entity = fts[0] if fts else ""
        return set(cls._apply_filter(fts, entity, forget_set_filter))

    @classmethod
    def load_forget_set(cls, forget_words_file: str, target_entity: str, forget_set_filter: str = "full") -> List[str]:
        """Load forget set from fts.json, applying forget_set_filter, entity placed first."""
        with open(forget_words_file, 'r') as f:
            fts = json.load(f)
        return cls._apply_filter(fts, target_entity, forget_set_filter)
    
    @classmethod
    @abstractmethod
    def preprocess(cls, config: RewardConfig) -> None:
        """
        Preprocess step that runs before training.
        
        This method should compute any required data structures (e.g., PageRank weights)
        and store them as class attributes for use in calc_reward().
        
        Args:
            config: RewardConfig with model, tokenizer, and file paths
        """
        pass
    
    @staticmethod
    @abstractmethod
    def calc_reward(completions: List[str], **kwargs) -> List[float]:
        """
        Calculate rewards for a batch of completions.
        
        This method is passed directly to GRPOTrainer as reward_funcs.
        It should be a static method that can access class-level state
        set during preprocess().
        
        Args:
            completions: List of generated text completions
            **kwargs: Additional arguments passed by GRPOTrainer
            
        Returns:
            List of reward scores (typically in [0, 1] range)
        """
        pass
    
    @classmethod
    def get_reward_func(cls):
        """
        Returns the calc_reward method bound to the class state.
        Use this when you need to pass the reward function to GRPOTrainer.
        """
        if not cls._preprocessed:
            raise RuntimeError(
                f"{cls.__name__}.preprocess() must be called before get_reward_func()"
            )
        return cls.calc_reward
    
    @classmethod
    def reset(cls) -> None:
        """Reset class-level state."""
        cls._forget_words = set()
        cls._forget_set = []
        cls._config = None
        cls._preprocessed = False

    @classmethod
    def get_weight_diagnostics(cls) -> Dict[str, Any]:
        """Default weight diagnostics for reward types without explicit graphs."""
        return {
            "status": "n/a",
            "reason": "not_applicable",
        }
