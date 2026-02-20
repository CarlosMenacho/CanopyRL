from .base import Randomizer
import numpy as np
from typing import Sequence, Optional


class LightingRandomizer(Randomizer):
    affects_spec = False
    needs_ctx = False

    def __init__(self,
                 pos_range_low: Sequence[float] = (-0.8, 0.5, -0.05),
                 pos_range_high: Sequence[float] = (1.2, 0.5, 0.2),
                 diffuse_range: Sequence[float] = (0.05, 0.3),
                 ambient_range: Sequence[float] = (0.0, 0.2),
                 specular_range: Sequence[float] = (0.0, 0.5)):

        self.pos_low = np.asarray(pos_range_low, dtype=float)
        self.pos_high = np.asarray(pos_range_high, dtype=float)
        self.diffuse_rng = diffuse_range
        self.ambient_rng = ambient_range
        self.specular_rng = specular_range

    def apply(self, *, spec, model, data, rng, ext=None):

        def _rand_rgb(lo_hi):
            lo, hi = lo_hi
            return rng.uniform(lo, hi, size=3)

        # Randomise all lights in the scene (world.xml defines 3 unnamed lights).
        for i in range(model.nlight):
            model.light_pos[i] = rng.uniform(self.pos_low, self.pos_high)
            model.light_diffuse[i] = _rand_rgb(self.diffuse_rng)
            model.light_ambient[i] = _rand_rgb(self.ambient_rng)
            model.light_specular[i] = _rand_rgb(self.specular_rng)

        # Headlight: set (not +=) to avoid accumulation across episodes.
        jitter = rng.uniform(-0.1, 0.1, size=9)
        model.vis.headlight.diffuse[:] = np.clip(0.6 + jitter[:3], 0.0, 1.0)
        model.vis.headlight.ambient[:] = np.clip(0.4 + jitter[3:6], 0.0, 1.0)
        model.vis.headlight.specular[:] = np.clip(0.2 + jitter[6:9], 0.0, 1.0)
