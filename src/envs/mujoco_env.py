import mujoco
import numpy as np
from omegaconf import DictConfig

from .base import BaseEnv
from src.randomizers.factory import build_randomisers


class MujocoEnv(BaseEnv):
    """MuJoCo environment wrapper compatible with the CanopyRL pipeline."""

    def __init__(self, cfg: DictConfig, xml_path: str) -> None:
        self.cfg = cfg
        self.xml_path = xml_path
        self._rng = np.random.default_rng(cfg.get("seed", 0))

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        rand_cfg = cfg.get("randomization", {})
        self.randomizers = build_randomisers(rand_cfg, xml_dir=str(xml_path).rsplit("/", 1)[0])

        self._step_count = 0
        self._max_steps = cfg.env.get("max_episode_steps", 200)
        self._frame_skip = cfg.env.get("frame_skip", 1)

        # IDs cached for action application
        _gripper_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        self._gripper_ctrl_max = float(self.model.actuator_ctrlrange[_gripper_act_id, 1])  # 255.0

        # IDs cached for observation computation
        self._base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link_base")
        self._tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "link_tcp")
        _driver_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint")
        self._driver_qpos_addr = self.model.jnt_qposadr[_driver_jnt_id]
        _jnt_range = self.model.jnt_range[_driver_jnt_id]
        self._driver_range = float(_jnt_range[1] - _jnt_range[0])  # 0.85 for xarm gripper

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple[np.ndarray, dict]:
        mujoco.mj_resetData(self.model, self.data)
        self._apply_randomizers()
        self._step_count = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._apply_action(action)
        for _ in range(self._frame_skip):
            mujoco.mj_step(self.model, self.data)  # type: ignore[attr-defined]
        self._step_count += 1

        obs = self._get_obs()
        info: dict = {}
        terminated = False
        truncated = self._step_count >= self._max_steps
        reward = 0.0  # reward injected externally by Trainer via RewardFn

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        pass

    @property
    def obs_dim(self) -> int:
        # qpos + qvel + tcp_pos_rel (3) + tcp_quat_rel (4) + gripper_state (1)
        return self.model.nq + self.model.nv + 8

    @property
    def action_dim(self) -> int:
        return self.model.nu

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        # --- joint state ---
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        # --- TCP pose relative to robot base ---
        pos_base = self.data.xpos[self._base_body_id]
        rot_base = self.data.xmat[self._base_body_id].reshape(3, 3)

        pos_tcp = self.data.site_xpos[self._tcp_site_id]
        rot_tcp = self.data.site_xmat[self._tcp_site_id].reshape(3, 3)

        # position in base frame
        tcp_pos_rel = rot_base.T @ (pos_tcp - pos_base)

        # orientation relative to base (rotation matrix → quaternion)
        rot_rel = rot_base.T @ rot_tcp
        tcp_quat_rel = np.empty(4)
        mujoco.mju_mat2Quat(tcp_quat_rel, rot_rel.flatten())

        # --- gripper state: 0 = open, 1 = closed ---
        gripper_state = np.array(
            [self.data.qpos[self._driver_qpos_addr] / self._driver_range]
        )

        return np.concatenate([qpos, qvel, tcp_pos_rel, tcp_quat_rel, gripper_state])

    def _apply_action(self, action: np.ndarray) -> None:
        # Arm joints (first nu-1): expected in [-1, 1]
        self.data.ctrl[:-1] = np.clip(action[:-1], -1.0, 1.0)
        # Gripper (last dim): expected in [0, 1] → scaled to [0, 255]
        self.data.ctrl[-1] = np.clip(action[-1], 0.0, 1.0) * self._gripper_ctrl_max

    def _apply_randomizers(self) -> None:
        for r in self.randomizers:
            r.apply(
                spec=None,
                model=self.model,
                data=self.data,
                rng=self._rng,
            )
