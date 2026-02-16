import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(f"Config: \n{OmegaConf.to_yaml(cfg)}")

if __name__ == "__main__":
    main()