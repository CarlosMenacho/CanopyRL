from abc import ABC, abstractmethod
import numpy as np


class BaseEnv(ABC):
    """Abstract base class for all RL environments."""

    @abstractmethod
    def reset(self) -> tuple[dict, dict]:
        """Reset env and return (obs, info)."""
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Apply action and return (obs, reward, terminated, truncated, info)."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources."""
        ...

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Observation space dimensionality."""
        ...

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Action space dimensionality."""
        ...

    @property
    @abstractmethod
    def img_shape(self) -> tuple[int, int, int]:
        """Image observation shape (H, W, C)."""
        ...
