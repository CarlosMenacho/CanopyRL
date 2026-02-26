from omegaconf import DictConfig

from .base import BaseReward
from .reach import ReachReward
from .grasp import GraspReward
from .picking import PickingReward

_REGISTRY: dict[str, type[BaseReward]] = {
    "reach": ReachReward,
    "grasp": GraspReward,
    "picking": PickingReward,
}


def build_reward(cfg: DictConfig) -> BaseReward:
    """Instantiate a reward function from config.

    Config example:
        rewards:
          type: reach
          w_dist: 1.0
          w_vel: 0.01
          target_body: tomato_a
    """
    reward_type = cfg.get("type", "reach")
    cls = _REGISTRY.get(reward_type)
    if cls is None:
        raise ValueError(f"Unknown reward type '{reward_type}'. Available: {list(_REGISTRY)}")
    return cls(cfg)
