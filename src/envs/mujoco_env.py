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
        return self.model.nq + self.model.nv

    @property
    def action_dim(self) -> int:
        return self.model.nu

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    def _apply_action(self, action: np.ndarray) -> None:
        np.clip(action, -1.0, 1.0, out=action)
        self.data.ctrl[:] = action

    def _apply_randomizers(self) -> None:
        for r in self.randomizers:
            r.apply(
                spec=None,
                model=self.model,
                data=self.data,
                rng=self._rng,
            )
