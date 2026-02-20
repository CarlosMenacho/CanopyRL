from __future__ import annotations

import colorsys
from typing import List, Optional, Tuple

import mujoco
import numpy as np

from .base import Randomizer

__all__ = ["FruitColorRandomizer"]


class FruitColorRandomizer(Randomizer):
    """Randomises fruit colour by sampling in HSV colour space.

    Keeps colours perceptually similar to real apples (red/yellow/green)
    while providing lighting-robust variation for training.  Each episode
    every fruit body receives an independently sampled colour.

    Parameters
    ----------
    fruit_body_names:
        MuJoCo *body* names whose geoms will be recoloured.
    hue_center:
        Central hue value in [0, 1].  0.02 ≈ red.
    hue_spread:
        ± half-width of the hue window.  0.12 reaches yellow-green.
    saturation_range:
        (min, max) saturation in [0, 1].
    value_range:
        (min, max) brightness in [0, 1].
    """

    affects_spec: bool = False
    needs_ctx: bool = False

    def __init__(
        self,
        fruit_body_names: Optional[List[str]] = None,
        hue_center: float = 0.02,
        hue_spread: float = 0.12,
        saturation_range: Tuple[float, float] = (0.75, 1.0),
        value_range: Tuple[float, float] = (0.65, 1.0),
    ) -> None:
        if fruit_body_names is None:
            fruit_body_names = ["apple_a", "apple_b", "apple_occluded"]
        self.fruit_names: List[str] = list(fruit_body_names)
        self.hue_center = hue_center
        self.hue_spread = hue_spread
        self.sat_lo, self.sat_hi = saturation_range
        self.val_lo, self.val_hi = value_range

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
            if geomnum == 0:
                continue

            h = (self.hue_center + rng.uniform(-self.hue_spread, self.hue_spread)) % 1.0
            s = float(rng.uniform(self.sat_lo, self.sat_hi))
            v = float(rng.uniform(self.val_lo, self.val_hi))
            r, g, b = colorsys.hsv_to_rgb(h, s, v)

            for gi in range(geomnum):
                model.geom_rgba[geomadr + gi, :3] = [r, g, b]
