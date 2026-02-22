from abc import ABC, abstractmethod
import numpy as np


class BaseReward(ABC):
    """Abstract base class for reward functions."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

    @abstractmethod
    def compute(self, model, data, info: dict) -> float:
        """Compute scalar reward from current simulation state."""
        ...

    def reset(self) -> None:
        """Called at episode reset to clear any stateful reward terms."""
        pass
