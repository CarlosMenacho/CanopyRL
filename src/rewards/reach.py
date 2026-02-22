import mujoco
import numpy as np

from .base import BaseReward


class ReachReward(BaseReward):
    """Reward for minimising distance between end-effector and target fruit."""

    def compute(self, model, data, info: dict) -> float:
        w_dist = self.cfg.get("w_dist", 1.0)
        w_vel = self.cfg.get("w_vel", 0.01)
        target = self.cfg.get("target_body", "tomato_a")

        target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target)
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")

        dist = float(np.linalg.norm(data.xpos[ee_id] - data.xpos[target_id]))
        vel_penalty = float(np.linalg.norm(data.qvel))

        return -w_dist * dist - w_vel * vel_penalty
