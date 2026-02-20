from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import mujoco
import numpy as np

from .base import Randomizer

__all__ = ["BackgroundImageRandomizer"]

_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


class BackgroundImageRandomizer(Randomizer):
    """Swaps the skybox texture with a randomly selected image file each episode.

    At each ``apply`` call one image is chosen uniformly at random from
    ``image_dir`` (or from the explicit ``image_paths`` list) and written
    directly into the skybox texture buffer.  The image is resized to match
    the compiled texture dimensions so no recompile is needed.

    Requires *opencv-python* (``cv2``) which is already listed in
    ``requirements.txt``.

    When an MjrContext is supplied via ``ext`` the texture is immediately
    uploaded to the GPU; otherwise the change will be visible on the next
    renderer update.

    Parameters
    ----------
    image_dir:
        Directory that will be scanned (non-recursively) for image files.
        Mutually exclusive with *image_paths*.
    image_paths:
        Explicit list of image file paths.  Mutually exclusive with
        *image_dir*.
    extensions:
        File extensions recognised when scanning *image_dir*.
    fallback_tint_range:
        When no images are available the skybox is tinted with a random
        per-channel factor in this range (same behaviour as
        :class:`SkyboxRandomizer`).
    """

    affects_spec: bool = False
    needs_ctx: bool = True

    def __init__(
        self,
        image_dir: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
        fallback_tint_range: Tuple[float, float] = (0.5, 1.5),
    ) -> None:
        if image_dir is not None and image_paths is not None:
            raise ValueError("Provide either image_dir or image_paths, not both.")

        if image_dir is not None:
            d = Path(image_dir)
            self._images: List[Path] = sorted(
                p for p in d.iterdir() if p.suffix.lower() in set(extensions)
            ) if d.exists() else []
        elif image_paths is not None:
            self._images = [Path(p) for p in image_paths]
        else:
            self._images = []

        self.tint_lo, self.tint_hi = fallback_tint_range
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
        import cv2  # local import — keeps module importable without cv2

        if self._skybox_tid == -1:
            self._skybox_tid = self._find_skybox(model)
        if self._skybox_tid == -1:
            return

        tid = self._skybox_tid
        adr = int(model.tex_adr[tid])
        h = int(model.tex_height[tid])
        w = int(model.tex_width[tid])
        nc = int(model.tex_nchannel[tid]) if hasattr(model, "tex_nchannel") else 3
        n = h * w * nc

        if self._images:
            idx = int(rng.integers(len(self._images)))
            img_bgr = cv2.imread(str(self._images[idx]))
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
                if nc == 4:
                    img_out = np.concatenate(
                        [img_rgb, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1
                    )
                else:
                    img_out = img_rgb
                model.tex_data[adr : adr + n] = img_out.reshape(-1).astype(np.uint8)
        else:
            # Fallback: tint the original texture data.
            if self._base_data is None:
                self._base_data = model.tex_data[adr : adr + n].copy()
            tint = rng.uniform(self.tint_lo, self.tint_hi, size=min(nc, 3))
            base = self._base_data.reshape(h, w, nc).astype(np.float32)
            result = base.copy()
            result[..., : len(tint)] *= tint
            np.clip(result, 0, 255, out=result)
            model.tex_data[adr : adr + n] = result.reshape(-1).astype(np.uint8)

        if ext is not None:
            mujoco.mjr_uploadTexture(model, ext, tid)
