from __future__ import annotations

from typing import Optional, Sequence, Tuple

import mujoco
import numpy as np

from .base import Randomizer

__all__ = ["CameraPoseRandomizer"]


def _axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), *(s * axis)], dtype=float)


def _quat_mul(q2: np.ndarray, q1: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
            w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,
            w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,
            w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,
        ],
        dtype=float,
    )


class CameraPoseRandomizer(Randomizer):
    """Randomises the camera mount offset and orientation (eye-in-hand).

    Simulates physical uncertainty in camera mounting position and tilt,
    helping the policy generalise across small calibration errors.

    Parameters
    ----------
    camera_name:
        Name of the MuJoCo camera to perturb.
    pos_delta_lo / pos_delta_hi:
        XYZ offset range (metres) relative to the camera's compiled pose.
    rot_enabled:
        Whether to also add a small random rotation.
    ang_range:
        Min / max perturbation angle (radians) when rot_enabled is True.
    """

    affects_spec: bool = False
    needs_ctx: bool = False

    def __init__(
        self,
        camera_name: str = "eye_in_hand",
        pos_delta_lo: Sequence[float] = (-0.005, -0.005, -0.003),
        pos_delta_hi: Sequence[float] = (0.005, 0.005, 0.003),
        rot_enabled: bool = True,
        ang_range: Tuple[float, float] = (-0.05, 0.05),
    ) -> None:
        self.camera_name = camera_name
        self.pos_lo = np.asarray(pos_delta_lo, dtype=float)
        self.pos_hi = np.asarray(pos_delta_hi, dtype=float)
        self.rot_enabled = rot_enabled
        self.ang_lo, self.ang_hi = ang_range
        # Base pose caches — populated on first apply so we always jitter
        # relative to the compiled model, not the previous episode.
        self._base_pos: Optional[np.ndarray] = None
        self._base_quat: Optional[np.ndarray] = None

    def apply(
        self,
        *,
        spec,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        rng: np.random.Generator,
        ext=None,
    ) -> None:
        cid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name
        )
        if cid == -1:
            return

        if self._base_pos is None:
            self._base_pos = model.cam_pos[cid].copy()
            self._base_quat = model.cam_quat[cid].copy()

        model.cam_pos[cid] = self._base_pos + rng.uniform(self.pos_lo, self.pos_hi)

        if not self.rot_enabled:
            return

        axis = rng.normal(size=3)
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            return
        axis /= norm
        angle = rng.uniform(self.ang_lo, self.ang_hi)
        dq = _axis_angle_to_quat(axis, angle)
        q = _quat_mul(dq, self._base_quat)
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-9:
            model.cam_quat[cid] = q / q_norm
