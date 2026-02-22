import mujoco
import numpy as np

from .base import BaseReward


class GraspReward(BaseReward):
    """Reward for a successful grasp: low distance + gripper contact with fruit."""

    def compute(self, model, data, info: dict) -> float:
        w_dist = self.cfg.get("w_dist", 1.0)
        w_contact = self.cfg.get("w_contact", 5.0)
        target = self.cfg.get("target_body", "tomato_a")

        target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target)
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")

        dist = float(np.linalg.norm(data.xpos[ee_id] - data.xpos[target_id]))
        reward = -w_dist * dist

        # Bonus for each contact involving the target body
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = model.geom_bodyid[c.geom1]
            b2 = model.geom_bodyid[c.geom2]
            if target_id in (b1, b2):
                reward += w_contact

        return reward
