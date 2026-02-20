from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import mujoco
import numpy as np

from .base import Randomizer

__all__ = ["FruitPoseRandomizer"]


class FruitPoseRandomizer(Randomizer):
    """Randomises apple body positions relative to their parent branches.

    Applies a per-episode random offset to each fruit body's local
    translation so the trained policy sees fruits at slightly different
    positions on the branches, improving generalisation.

    Parameters
    ----------
    fruit_body_names:
        MuJoCo body names for each fruit.  Defaults to the three apples
        defined in world.xml.
    pos_delta_lo / pos_delta_hi:
        Minimum and maximum XYZ offsets (metres) sampled each episode.
    """

    affects_spec: bool = False
    needs_ctx: bool = False

    def __init__(
        self,
        fruit_body_names: Optional[List[str]] = None,
        pos_delta_lo: Sequence[float] = (-0.02, -0.02, -0.01),
        pos_delta_hi: Sequence[float] = (0.02, 0.02, 0.01),
    ) -> None:
        if fruit_body_names is None:
            fruit_body_names = ["apple_a", "apple_b", "apple_occluded"]
        self.fruit_names: List[str] = list(fruit_body_names)
        self.pos_lo = np.asarray(pos_delta_lo, dtype=float)
        self.pos_hi = np.asarray(pos_delta_hi, dtype=float)
        # Cache original body positions so we always jitter from the
        # compiled pose rather than accumulating across episodes.
        self._base_pos: Dict[int, np.ndarray] = {}

    def apply(
        self,
        *,
        spec,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        rng: np.random.Generator,
        ext=None,
    ) -> None:
        for name in self.fruit_names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid == -1:
                continue

            # Cache compiled default on first call.
            if bid not in self._base_pos:
                self._base_pos[bid] = model.body_pos[bid].copy()

            delta = rng.uniform(self.pos_lo, self.pos_hi)
            model.body_pos[bid] = self._base_pos[bid] + delta
