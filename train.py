"""CanopyRL — main training entry point.

Usage::

    # default run
    python train.py

    # override flags inline (Hydra syntax)
    python train.py total_steps=2_000_000 device=cpu headless_mode=true
    python train.py rewards=grasp gen_video_test=10
    python train.py algorithm=sac algorithm.lr_actor=1e-4
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Defer heavy imports so Hydra tab-completion stays fast
from hydra.utils import get_original_cwd

from src.utils.seeding import set_seed
from src.utils.logger import Logger
from src.envs.mujoco_env import MujocoEnv
from src.rewards.factory import build_reward
from src.algorithms.sac import SAC
from src.trainer.trainer import Trainer

# Ensure 'src' is importable even after Hydra changes the working directory.
sys.path.insert(0, str(Path(__file__).parent))

log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    

    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # ── reproducibility ────────────────────────────────────────────────────
    set_seed(cfg.seed)

    # ── device ────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    log.info("Device: %s", device)

    # ── resolve paths (Hydra moves CWD to outputs/) ───────────────────────
    orig = Path(get_original_cwd())
    xml_path = str(orig / cfg.env.xml_path)

    # ── environment ───────────────────────────────────────────────────────
    env = MujocoEnv(cfg, xml_path=xml_path)
    log.info("Env ready  |  obs_dim=%d  action_dim=%d", env.obs_dim, env.action_dim)

    # ── reward function ───────────────────────────────────────────────────
    reward_fn = build_reward(cfg.rewards)
    log.info("Reward: %s", cfg.rewards.type)

    # ── agent ─────────────────────────────────────────────────────────────
    agent = SAC(
        cfg=cfg.algorithm,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device=device,
    )
    log.info("Agent: SAC  |  hidden_dim=%d", cfg.algorithm.hidden_dim)

    # ── logger ────────────────────────────────────────────────────────────
    logger = Logger(log_dir="tensorboard/")

    # ── trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(cfg=cfg, env=env, agent=agent, reward_fn=reward_fn, logger=logger)

    log.info(
        "Training start  |  total_steps=%d  batch_size=%d  "
        "learning_starts=%d  headless=%s",
        cfg.total_steps,
        cfg.trainer.batch_size,
        cfg.trainer.learning_starts,
        cfg.headless_mode,
    )
    trainer.run()

    # ── post-training video rollout ────────────────────────────────────────
    n_video = cfg.get("gen_video_test", 0)
    if n_video > 0:
        from src.utils.video import record_video

        # Load the best checkpoint before recording
        best_ckpt = Path(cfg.trainer.checkpoint_dir) / "best.pt"
        if best_ckpt.exists():
            agent.load(str(best_ckpt))
            log.info("Loaded best checkpoint for video rollout.")

        log.info("Recording %d test episode(s)…", n_video)
        record_video(
            env=env,
            agent=agent,
            reward_fn=reward_fn,
            n_episodes=n_video,
            output_dir="videos/",
            max_episode_steps=cfg.env.max_episode_steps,
        )


if __name__ == "__main__":
    main()
