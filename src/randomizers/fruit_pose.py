from __future__ import annotations

from typing import Optional, Sequence, Tuple

import mujoco
import numpy as np

from .base import Randomizer

__all__ = ["FruitPoseRandomizer"]


class FruitPoseRandomizer(Randomizer):
    """Randomises the vine body's world position each episode.

    Applies small per-episode XYZ offsets to the vine's compiled pose so
    the fruit cluster appears at slightly different locations relative to
    the robot.  Deltas are always relative to the compiled default (cached
    on first call) so they never accumulate across episodes.

    Parameters
    ----------
    body_name:
        Name of the vine/fruit-cluster body to perturb.
    pos_delta_lo / pos_delta_hi:
        Per-axis (X, Y, Z) offset range in metres.
    """

    affects_spec: bool = False
    needs_ctx: bool = False

    def __init__(
        self,
        body_name: str = "vine",
        pos_delta_lo: Sequence[float] = (-0.05, -0.10, -0.05),
        pos_delta_hi: Sequence[float] = (0.05,  0.10,  0.05),
    ) -> None:
        self.body_name = body_name
        self.pos_lo = np.asarray(pos_delta_lo, dtype=float)
        self.pos_hi = np.asarray(pos_delta_hi, dtype=float)
        self._base_pos: Optional[np.ndarray] = None

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

        model.body_pos[bid] = self._base_pos + rng.uniform(self.pos_lo, self.pos_hi)
