from __future__ import annotations

from typing import Optional, Tuple

import mujoco
import numpy as np

from .base import Randomizer

__all__ = ["MujocoTableRandomizer"]


class MujocoTableRandomizer(Randomizer):
    """Randomises the floor/ground appearance by recolouring its texture.

    The ground in ``world.xml`` uses a built-in checker texture
    (``ground_tex``) applied via ``ground_mat``.  This randomiser rewrites
    the two checker colours in the texture buffer every episode, producing
    different ground appearances without recompiling the model.

    It also randomises the material RGBA tint, which multiplies with the
    texture and provides a quick global brightness / hue shift.

    When an MjrContext is provided via ``ext`` the updated texture is
    uploaded to the GPU immediately.

    Parameters
    ----------
    material_name:
        Name of the MuJoCo material to perturb.
    color1_range / color2_range:
        (min, max) range sampled per channel for each checker colour.
        Defaults keep an earthy green-brown palette.
    tint_range:
        (min, max) multiplicative tint applied to the material RGBA.
    """

    affects_spec: bool = False
    needs_ctx: bool = True

    def __init__(
        self,
        material_name: str = "ground_mat",
        color1_range: Tuple[float, float] = (0.10, 0.50),
        color2_range: Tuple[float, float] = (0.05, 0.35),
        tint_range: Tuple[float, float] = (0.7, 1.3),
    ) -> None:
        self.material_name = material_name
        self.c1_lo, self.c1_hi = color1_range
        self.c2_lo, self.c2_hi = color2_range
        self.tint_lo, self.tint_hi = tint_range
        self._tex_tid: int = -2   # -2 = not looked up yet; -1 = not found
        self._base_tex: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_tex(self, model: mujoco.MjModel) -> int:
        """Return the texture ID bound to *material_name*, or -1."""
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, self.material_name)
        if mid == -1:
            return -1
        # mat_texid layout differs between MuJoCo versions.
        texid_raw = model.mat_texid[mid]
        tid = int(texid_raw.flat[0]) if hasattr(texid_raw, "flat") else int(texid_raw)
        return tid

    # ------------------------------------------------------------------
    # Randomizer interface
    # ------------------------------------------------------------------

    def apply(
        self,
        *,
        spec,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        rng: np.random.Generator,
        ext=None,
    ) -> None:
        # ---- material RGBA tint ----
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, self.material_name)
        if mid != -1:
            tint = rng.uniform(self.tint_lo, self.tint_hi, size=3)
            model.mat_rgba[mid, :3] = np.clip(tint, 0.0, 1.0)

        # ---- texture checker recolouring ----
        if self._tex_tid == -2:
            self._tex_tid = self._find_tex(model)
        if self._tex_tid < 0:
            return

        tid = self._tex_tid
        adr = int(model.tex_adr[tid])
        h = int(model.tex_height[tid])
        w = int(model.tex_width[tid])
        nc = int(model.tex_nchannel[tid]) if hasattr(model, "tex_nchannel") else 3
        n = h * w * nc

        if self._base_tex is None:
            self._base_tex = model.tex_data[adr : adr + n].copy()

        base = self._base_tex.reshape(h, w, nc).astype(np.float32)
        # Determine which pixels belong to each checker square via brightness.
        brightness = base[..., :3].mean(axis=-1)
        threshold = brightness.max() * 0.5

        c1 = rng.uniform(self.c1_lo, self.c1_hi, size=3) * 255.0
        c2 = rng.uniform(self.c2_lo, self.c2_hi, size=3) * 255.0

        new_tex = np.empty_like(base)
        mask = brightness >= threshold
        new_tex[mask, :3] = c1
        new_tex[~mask, :3] = c2
        if nc == 4:
            new_tex[..., 3] = 255.0

        np.clip(new_tex, 0, 255, out=new_tex)
        model.tex_data[adr : adr + n] = new_tex.reshape(-1).astype(np.uint8)

        if ext is not None:
            mujoco.mjr_uploadTexture(model, ext, tid)
