from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from .base import Randomizer

__all__ = ["FruitMeshVariantRandomizer"]


class FruitMeshVariantRandomizer(Randomizer):
    """Randomises fruit geometry by perturbing sphere radius each episode.

    Modifies ``model.geom_size`` for the geom(s) belonging to each fruit
    body.  The radius is sampled uniformly from [base * lo_factor,
    base * hi_factor] so the change is always relative to the compiled size.

    This simulates natural variation in fruit sizes and helps the visual
    detector generalise across different fruit scales.

    Parameters
    ----------
    fruit_body_names:
        MuJoCo body names for each fruit.
    scale_range:
        (min, max) multiplicative factor applied to the compiled sphere
        radius.  E.g. (0.8, 1.2) produces ±20 % size variation.
    """

    affects_spec: bool = False
    needs_ctx: bool = False

    def __init__(
        self,
        fruit_body_names: Optional[List[str]] = None,
        scale_range: Tuple[float, float] = (0.8, 1.2),
    ) -> None:
        if fruit_body_names is None:
            fruit_body_names = ["apple_a", "apple_b", "apple_occluded"]
        self.fruit_names: List[str] = list(fruit_body_names)
        self.scale_lo, self.scale_hi = scale_range
        # Cache compiled sizes keyed by geom id.
        self._base_size: Dict[int, np.ndarray] = {}

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

            geomadr = int(model.body_geomadr[bid])
            geomnum = int(model.body_geomnum[bid])

            for gi in range(geomnum):
                gid = geomadr + gi
                # Only resize sphere / capsule geoms.
                gtype = int(model.geom_type[gid])
                if gtype not in (
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                ):
                    continue

                if gid not in self._base_size:
                    self._base_size[gid] = model.geom_size[gid].copy()

                sf = float(rng.uniform(self.scale_lo, self.scale_hi))
                model.geom_size[gid] = self._base_size[gid] * sf
