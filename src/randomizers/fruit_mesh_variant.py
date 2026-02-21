from __future__ import annotations

from typing import Optional, Sequence, Tuple

import mujoco
import numpy as np

from .base import Randomizer

__all__ = ["FruitMeshVariantRandomizer"]


class FruitMeshVariantRandomizer(Randomizer):
    """Randomises the size of fruit sphere geoms each episode.

    Independently scales each listed geom's radius by a factor drawn from
    ``scale_range``, simulating natural fruit-size variation.  The base
    radius is cached on the first call so scales are always applied to the
    compiled default, not the previous episode's value.

    Parameters
    ----------
    geom_names:
        Names of the fruit geoms to rescale.
    scale_range:
        (min, max) multiplicative factor applied to each geom's radius.
    """

    affects_spec: bool = False
    needs_ctx: bool = False

    def __init__(
        self,
        geom_names: Sequence[str] = ("tomato_geom_a", "tomato_geom_b", "tomato_geom_c"),
        scale_range: Tuple[float, float] = (0.85, 1.15),
    ) -> None:
        self.geom_names = list(geom_names)
        self.scale_lo, self.scale_hi = scale_range
        self._base_sizes: Optional[dict] = None

    def apply(
        self,
        *,
        spec,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        rng: np.random.Generator,
        ext=None,
    ) -> None:
        if self._base_sizes is None:
            self._base_sizes = {}
            for name in self.geom_names:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid != -1:
                    self._base_sizes[name] = model.geom_size[gid].copy()

        for name in self.geom_names:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid == -1:
                continue
            scale = float(rng.uniform(self.scale_lo, self.scale_hi))
            model.geom_size[gid] = self._base_sizes[name] * scale
