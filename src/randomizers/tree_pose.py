from __future__ import annotations

from typing import Optional, Sequence, Tuple

import mujoco
import numpy as np

from .base import Randomizer

__all__ = ["TreePoseRandomizer"]


def _quat_mul(q2: np.ndarray, q1: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
        w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,
        w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,
        w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,
    ], dtype=float)


def _yaw_quat(angle: float) -> np.ndarray:
    """Quaternion for a rotation of *angle* radians around Z."""
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), 0.0, 0.0, s], dtype=float)


class TreePoseRandomizer(Randomizer):
    """Randomises the tree body's XY position and yaw each episode.

    Applies small per-episode offsets to the tree's compiled world pose so
    the robot sees the canopy at slightly different locations and angles.
    Both position and yaw deltas are sampled relative to the compiled
    defaults (cached on the first call), so values never accumulate.

    Parameters
    ----------
    body_name:
        MuJoCo body name of the tree root.
    pos_delta_lo / pos_delta_hi:
        XY (and optionally Z) offset range in metres.  The default keeps
        the tree on the ground (zero Z range).
    yaw_enabled:
        Whether to also apply a random yaw perturbation around Z.
    yaw_range:
        (min, max) yaw delta in radians.  Applied on top of the compiled
        rotation (e.g. the 180° flip set in world.xml).
    """

    affects_spec: bool = False
    needs_ctx: bool = False

    def __init__(
        self,
        body_name: str = "tree",
        pos_delta_lo: Sequence[float] = (-0.05, -0.08, 0.0),
        pos_delta_hi: Sequence[float] = (0.05,  0.08, 0.0),
        yaw_enabled: bool = True,
        yaw_range: Tuple[float, float] = (-0.20, 0.20),
    ) -> None:
        self.body_name = body_name
        self.pos_lo = np.asarray(pos_delta_lo, dtype=float)
        self.pos_hi = np.asarray(pos_delta_hi, dtype=float)
        self.yaw_enabled = yaw_enabled
        self.yaw_lo, self.yaw_hi = yaw_range
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
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        if bid == -1:
            return

        if self._base_pos is None:
            self._base_pos = model.body_pos[bid].copy()
            self._base_quat = model.body_quat[bid].copy()

        # ── position ──────────────────────────────────────────────────────
        model.body_pos[bid] = self._base_pos + rng.uniform(self.pos_lo, self.pos_hi)

        # ── yaw ───────────────────────────────────────────────────────────
        if self.yaw_enabled:
            dq = _yaw_quat(float(rng.uniform(self.yaw_lo, self.yaw_hi)))
            q = _quat_mul(dq, self._base_quat)
            norm = np.linalg.norm(q)
            if norm > 1e-9:
                model.body_quat[bid] = q / norm
