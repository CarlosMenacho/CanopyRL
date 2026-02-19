from abc import ABC, abstractmethod
import mujoco
from typing import Optional
import numpy as np


class Randomizer(ABC):

    affects_spec: bool = False
    needs_ctx: bool = False

    @abstractmethod
    def apply(self,
              *,
              spec: Optional[mujoco.MjSpec],
              model: mujoco.MjModel,
              data: mujoco.MjData,
              rng: np.random.Generator,
              ext: Optional[mujoco.MjrContext] = None) -> None:
        ...
