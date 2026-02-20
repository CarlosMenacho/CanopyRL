from __future__ import annotations

from typing import List, Optional, Tuple

import mujoco
import numpy as np

from .base import Randomizer

__all__ = ["MeshVariantRandomizer"]


class MeshVariantRandomizer(Randomizer):
    """Randomises mesh asset scales in the MjSpec for sim-to-real robustness.

    Applies a uniform scale perturbation to named mesh assets before the
    model is (re-)compiled, simulating manufacturing tolerances or
    small shape differences between the real robot and its CAD model.

    Because mesh assets are baked in at compile time, this randomiser sets
    ``affects_spec = True``.  The caller **must** recompile the model after
    invoking ``apply``.  See ``factory.build_randomisers`` and the demo in
    ``test.py`` for the recommended usage pattern.

    Parameters
    ----------
    mesh_names:
        Names of the mesh assets to perturb.  When *None* every mesh in the
        spec is perturbed.
    scale_range:
        (min, max) multiplicative scale factor.  E.g. (0.98, 1.02) adds
        ±2 % size variation.
    per_axis:
        When True each axis (X, Y, Z) gets an independent scale factor,
        producing non-uniform stretching.  When False a single scalar is
        sampled and applied to all three axes.
    """

    affects_spec: bool = True
    needs_ctx: bool = False

    def __init__(
        self,
        mesh_names: Optional[List[str]] = None,
        scale_range: Tuple[float, float] = (0.98, 1.02),
        per_axis: bool = False,
    ) -> None:
        self.mesh_names: Optional[set] = set(mesh_names) if mesh_names else None
        self.scale_lo, self.scale_hi = scale_range
        self.per_axis = per_axis

    def apply(
        self,
        *,
        spec: mujoco.MjSpec,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        rng: np.random.Generator,
        ext=None,
    ) -> None:
        if spec is None:
            return

        mesh = spec.first_mesh()
        while mesh is not None:
            if self.mesh_names is None or mesh.name in self.mesh_names:
                cur = np.array(mesh.scale, dtype=float)
                # Treat an all-zero default scale as identity (1, 1, 1).
                if np.all(cur == 0.0):
                    cur = np.ones(3, dtype=float)

                if self.per_axis:
                    sf = rng.uniform(self.scale_lo, self.scale_hi, size=3)
                else:
                    sf = float(rng.uniform(self.scale_lo, self.scale_hi))

                mesh.scale = (cur * sf).tolist()
            mesh = mesh.next()
