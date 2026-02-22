from abc import ABC, abstractmethod
import numpy as np
import torch


class BaseAlgorithm(ABC):
    """Abstract base class for RL algorithms."""

    def __init__(self, cfg, obs_dim: int, action_dim: int, device: torch.device) -> None:
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

    @abstractmethod
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Sample an action given an observation."""
        ...

    @abstractmethod
    def update(self, batch: dict) -> dict[str, float]:
        """Perform one gradient update step; return a dict of loss scalars."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model weights to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from disk."""
        ...
