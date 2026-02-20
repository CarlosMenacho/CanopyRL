from __future__ import annotations

from typing import Optional, Tuple

import mujoco
import numpy as np

from .base import Randomizer

__all__ = ["SkyboxRandomizer"]


class SkyboxRandomizer(Randomizer):
    """Randomises the skybox texture by applying a random colour tint.

    On the first call the original pixel data is cached.  Subsequent calls
    multiply that cached data by a per-channel tint factor sampled from
    ``tint_range``, giving a different ambient colour each episode without
    needing additional image files.

    When a rendering context is provided (``ext`` argument) the updated
    texture is uploaded to the GPU immediately.  If ``ext`` is *None* the
    CPU-side data is updated and the renderer will pick it up on the next
    scene update.

    Parameters
    ----------
    tint_range:
        (min, max) multiplicative factor applied independently to each of
        the R, G, B channels.  Values > 1 brighten, < 1 darken.
    """

    affects_spec: bool = False
    needs_ctx: bool = True

    def __init__(
        self,
        tint_range: Tuple[float, float] = (0.5, 1.5),
    ) -> None:
        self.tint_lo, self.tint_hi = tint_range
        self._skybox_tid: int = -1
        self._base_data: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_skybox(self, model: mujoco.MjModel) -> int:
        for tid in range(model.ntex):
            if model.tex_type[tid] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                return tid
        return -1

    def _tex_slice(self, model: mujoco.MjModel, tid: int):
        """Return (adr, h, w, nc, slice) for texture *tid*."""
        adr = int(model.tex_adr[tid])
        h = int(model.tex_height[tid])
        w = int(model.tex_width[tid])
        # nchannel available in MuJoCo ≥ 3.1; fall back to 3 for older builds.
        nc = int(model.tex_nchannel[tid]) if hasattr(model, "tex_nchannel") else 3
        n = h * w * nc
        return adr, h, w, nc, model.tex_data[adr : adr + n]

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
        if self._skybox_tid == -1:
            self._skybox_tid = self._find_skybox(model)
        if self._skybox_tid == -1:
            return

        tid = self._skybox_tid
        adr, h, w, nc, tex_slice = self._tex_slice(model, tid)

        if self._base_data is None:
            self._base_data = tex_slice.copy()

        tint = rng.uniform(self.tint_lo, self.tint_hi, size=min(nc, 3))
        base = self._base_data.reshape(h, w, nc).astype(np.float32)
        result = base.copy()
        result[..., : len(tint)] *= tint
        np.clip(result, 0, 255, out=result)
        model.tex_data[adr : adr + h * w * nc] = result.reshape(-1).astype(np.uint8)

        if ext is not None:
            mujoco.mjr_uploadTexture(model, ext, tid)
