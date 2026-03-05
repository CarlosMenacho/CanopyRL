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

        # Camera / renderer
        self._cam_name = cfg.env.get("cam_name", "eye_in_hand")
        self._img_h = cfg.env.get("img_height", 128)
        self._img_w = cfg.env.get("img_width", 128)
        self._renderer = mujoco.Renderer(self.model, height=self._img_h, width=self._img_w)

        # IDs cached for action application
        _gripper_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        self._gripper_ctrl_max = float(self.model.actuator_ctrlrange[_gripper_act_id, 1])  # 255.0

        # IDs cached for observation computation
        _driver_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint")
        self._driver_qpos_addr = self.model.jnt_qposadr[_driver_jnt_id]
        _jnt_range = self.model.jnt_range[_driver_jnt_id]
        self._driver_range = float(_jnt_range[1] - _jnt_range[0])  # 0.85 for xarm gripper


    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple[dict, dict]:
        mujoco.mj_resetData(self.model, self.data) 
        self._apply_randomizers()
        self._step_count = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        self._apply_action(action)
        for _ in range(self._frame_skip):
            mujoco.mj_step(self.model, self.data) 
        self._step_count += 1

        obs = self._get_obs()
        info: dict = {}
        terminated = False
        truncated = self._step_count >= self._max_steps
        reward = 0.0  # reward injected externally by Trainer via RewardFn

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._renderer.close()

    @property
    def obs_dim(self) -> int:
        # qpos + qvel + gripper_state (1)
        return self.model.nq + self.model.nv + 1

    @property
    def action_dim(self) -> int:
        return self.model.nu

    @property
    def img_shape(self) -> tuple[int, int, int]:
        return (self._img_h, self._img_w, 3)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict:
        # --- joint state ---
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()

        # --- gripper state: 0 = open, 1 = closed ---
        gripper_state = np.array(
            [self.data.qpos[self._driver_qpos_addr] / self._driver_range]
        )

        return {
            "state": np.concatenate([qpos, qvel, gripper_state]),
            "pixels": self._render_pixels(),
        }

    def _render_pixels(self) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=self._cam_name)
        return self._renderer.render().copy()  # uint8 HxWx3

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
