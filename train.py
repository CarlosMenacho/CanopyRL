import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("Config: ")
    logger.info(cfg)

if __name__ == "__main__":
    main()