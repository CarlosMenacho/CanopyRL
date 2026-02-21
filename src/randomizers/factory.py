from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

from src.randomizers import (
    BackgroundImageRandomizer,
    CameraPoseRandomizer,
    FruitColorRandomizer,
    FruitMeshVariantRandomizer,
    FruitPoseRandomizer,
    LightingRandomizer,
    MeshVariantRandomizer,
    MujocoTableRandomizer,
    Randomizer,
    RobotPoseRandomizer,
    SkyboxRandomizer,
    TreePoseRandomizer,
)

__all__ = ["build_randomisers"]

# Maps config key → Randomizer class.
_REGISTRY: Dict[str, type] = {
    "lights": LightingRandomizer,
    "robot_pose": RobotPoseRandomizer,
    "fruit_pose": FruitPoseRandomizer,
    "camera_pose": CameraPoseRandomizer,
    "fruit_color": FruitColorRandomizer,
    "mesh_variant": MeshVariantRandomizer,
    "skybox": SkyboxRandomizer,
    "fruit_mesh_variant": FruitMeshVariantRandomizer,
    "background_image": BackgroundImageRandomizer,
    "table": MujocoTableRandomizer,
    "tree_pose": TreePoseRandomizer,
}


def _to_dict(value: Any) -> Dict[str, Any]:
    """Coerce a config value to a plain Python dict.

    Handles both plain ``dict`` and OmegaConf ``DictConfig`` so the factory
    works with raw Python configs and with Hydra.
    """
    if isinstance(value, dict):
        return dict(value)
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return dict(OmegaConf.to_container(value, resolve=True))
    except ImportError:
        pass
    return {}


def build_randomisers(
    cfg: Mapping,
    *,
    xml_dir: Path | str = "",
) -> List[Randomizer]:
    """Build a list of :class:`~src.randomizers.base.Randomizer` instances
    from a configuration mapping.

    Each key in *cfg* maps to an entry in the randomiser registry.  The
    value controls whether the randomiser is created and with which kwargs:

    * ``"enabled"`` or ``True``  — create with default parameters.
    * ``"disabled"`` or ``False`` — skip entirely.
    * A dict / OmegaConf node    — ``enabled`` key (default ``True``) gates
      creation; remaining keys are forwarded as ``__init__`` kwargs.

    Parameters
    ----------
    cfg:
        Mapping of randomiser keys to their configurations.
    xml_dir:
        Base directory for resolving relative asset paths (reserved for
        future use).

    Returns
    -------
    list[Randomizer]
        Ordered list of instantiated randomisers.

    Examples
    --------
    Plain-dict usage::

        cfg = {
            "lights":       {"enabled": True, "diffuse_range": [0.05, 0.3]},
            "robot_pose":   "disabled",
        }
        randomisers = build_randomisers(cfg)

    Equivalent YAML for Hydra ``rand_conf.yaml``::

        lights:
          enabled: true
          diffuse_range: [0.05, 0.3]
        robot_pose: disabled
    """
    xml_dir = Path(xml_dir)  # reserved for future asset-path resolution
    out: List[Randomizer] = []

    for key, cls in _REGISTRY.items():
        raw = cfg.get(key)
        if raw is None:
            continue

        # Short-hand: plain string or bool.
        if raw in ("disabled", False):
            continue
        if raw in ("enabled", True):
            out.append(cls())
            continue

        # Full dict / OmegaConf node.
        kw = _to_dict(raw)
        if not kw.get("enabled", True):
            continue
        kw.pop("enabled", None)

        out.append(cls(**kw))

    return out
