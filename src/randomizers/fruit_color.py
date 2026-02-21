from __future__ import annotations

from typing import Optional, Sequence, Tuple

import mujoco
import numpy as np

from .base import Randomizer

__all__ = ["FruitColorRandomizer"]


class FruitColorRandomizer(Randomizer):
    """Randomises the RGBA colour of fruit geoms each episode.

    Samples a colour for each listed geom from a hue range that spans
    unripe (green) through ripening (yellow/orange) to fully ripe (red),
    keeping the alpha channel fixed at 1.

    Parameters
    ----------
    geom_names:
        Names of the fruit geoms to recolour.
    hue_range:
        (min, max) hue in [0, 1].  Defaults to red–orange–yellow range.
    sat_range:
        (min, max) HSV saturation.
    val_range:
        (min, max) HSV value (brightness).
    """

    affects_spec: bool = False
    needs_ctx: bool = False

    def __init__(
        self,
        geom_names: Sequence[str] = ("tomato_geom_a", "tomato_geom_b", "tomato_geom_c"),
        hue_range: Tuple[float, float] = (0.00, 0.18),
        sat_range: Tuple[float, float] = (0.70, 1.00),
        val_range: Tuple[float, float] = (0.75, 1.00),
    ) -> None:
        self.geom_names = list(geom_names)
        self.hue_lo, self.hue_hi = hue_range
        self.sat_lo, self.sat_hi = sat_range
        self.val_lo, self.val_hi = val_range

    # ------------------------------------------------------------------

    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float) -> np.ndarray:
        """Convert HSV (all in [0,1]) to RGB array."""
        if s == 0.0:
            return np.array([v, v, v], dtype=float)
        i = int(h * 6.0)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i %= 6
        if i == 0:
            return np.array([v, t, p], dtype=float)
        if i == 1:
            return np.array([q, v, p], dtype=float)
        if i == 2:
            return np.array([p, v, t], dtype=float)
        if i == 3:
            return np.array([p, q, v], dtype=float)
        if i == 4:
            return np.array([t, p, v], dtype=float)
        return np.array([v, p, q], dtype=float)

    def apply(
        self,
        *,
        spec,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        rng: np.random.Generator,
        ext=None,
    ) -> None:
        for name in self.geom_names:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid == -1:
                continue
            h = float(rng.uniform(self.hue_lo, self.hue_hi))
            s = float(rng.uniform(self.sat_lo, self.sat_hi))
            v = float(rng.uniform(self.val_lo, self.val_hi))
            rgb = self._hsv_to_rgb(h, s, v)
            model.geom_rgba[gid, :3] = rgb
            model.geom_rgba[gid, 3] = 1.0
