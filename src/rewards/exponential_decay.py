import math

from rewards import BinaryReward
from rewards.base import RewardConfig
from typing import List
from rewards.base import RewardFunction


class ExponentialDecayReward(RewardFunction):
    """
    Exponential decay reward function.
    """

    _tau: float = 1.00
    _base: float = math.e

    @classmethod
    def preprocess(cls, config: RewardConfig) -> None:
        BinaryReward.preprocess(config)
        cls._config = config
        cls._preprocessed = True

    @staticmethod
    def calc_reward(completions: List[str], **kwargs) -> List[float]:
        pattern = BinaryReward._pattern
        base = ExponentialDecayReward._base
        tau = ExponentialDecayReward._tau
        rewards: List[float] = []
        for completion in completions:
            matches = pattern.findall(completion)
            forget_count = len(matches)
            reward = base ** (-(forget_count / tau)) if base > 0 else 0.0
            rewards.append(reward)
        return rewards
