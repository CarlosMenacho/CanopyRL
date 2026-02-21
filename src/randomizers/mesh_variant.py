from __future__ import annotations

from typing import Optional, Sequence, Tuple

import mujoco
import numpy as np

from .base import Randomizer

__all__ = ["MeshVariantRandomizer"]


class MeshVariantRandomizer(Randomizer):
    """Spec-level randomiser that varies the vine capsule radius.

    Modifies the MjSpec before compilation so the vine stem and peduncle
    geometry changes shape each episode.  Because it requires an MjSpec
    this randomiser is spec-modifying (``affects_spec = True``) and must
    be applied before ``spec.compile()``.

    Parameters
    ----------
    vine_geom_name:
        Name of the main vine capsule geom in the spec.
    radius_range:
        (min, max) radius in metres for the vine capsule.
    peduncle_geom_names:
        Names of proximal peduncle capsule geoms to co-vary.
    peduncle_radius_range:
        (min, max) radius for each peduncle capsule.
    """

    affects_spec: bool = True
    needs_ctx: bool = False

    def __init__(
        self,
        vine_geom_name: str = "vine_capsule",
        radius_range: Tuple[float, float] = (0.008, 0.018),
        peduncle_geom_names: Sequence[str] = (
            "peduncle1_a_caps",
            "peduncle1_b_caps",
            "peduncle1_c_caps",
        ),
        peduncle_radius_range: Tuple[float, float] = (0.007, 0.013),
    ) -> None:
        self.vine_geom_name = vine_geom_name
        self.radius_lo, self.radius_hi = radius_range
        self.peduncle_geom_names = list(peduncle_geom_names)
        self.ped_lo, self.ped_hi = peduncle_radius_range

    def apply(
        self,
        *,
        spec: Optional[mujoco.MjSpec],
        model: mujoco.MjModel,
        data: mujoco.MjData,
        rng: np.random.Generator,
        ext=None,
    ) -> None:
        if spec is None:
            return

        vine_r = float(rng.uniform(self.radius_lo, self.radius_hi))
        ped_r = float(rng.uniform(self.ped_lo, self.ped_hi))

        for geom in spec.worldbody.find_all("geom"):
            if geom.name == self.vine_geom_name:
                geom.size[0] = vine_r
            elif geom.name in self.peduncle_geom_names:
                geom.size[0] = ped_r
