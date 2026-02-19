from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Tuple

import numpy as np
from src.randomizers import (Randomizer, LightingRandomizer,
                             RobotPoseRandomizer)

__all__ = ["build_randomizers"]


def build_randomisers(cfg: Mapping,
                      *,
                      xml_dir: Path | str = "") -> List[Randomizer]:
    pass
