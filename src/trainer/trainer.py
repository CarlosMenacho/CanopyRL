import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from src.envs.base import BaseEnv
from src.algorithms.base import BaseAlgorithm
from src.rewards.base import BaseReward
from src.utils.logger import Logger
from .replay_buffer import ReplayBuffer

log = logging.getLogger(__name__)


class Trainer:
    """Central training loop: collect experience → update agent → log → checkpoint."""

    def __init__(
        self,
        cfg: DictConfig,
        env: BaseEnv,
        agent: BaseAlgorithm,
        reward_fn: BaseReward,
        logger: Logger,
    ) -> None:
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.reward_fn = reward_fn
        self.logger = logger

        tcfg = cfg.trainer
        self.buffer = ReplayBuffer(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            capacity=tcfg.buffer_capacity,
        )

        self.batch_size = tcfg.batch_size
        self.learning_starts = tcfg.learning_starts
        self.update_every = tcfg.update_every
        self.gradient_steps = tcfg.gradient_steps
        self.log_every = tcfg.log_every
        self.checkpoint_every = tcfg.checkpoint_every
        self.checkpoint_dir = Path(tcfg.checkpoint_dir)
        self.total_steps = cfg.total_steps
        self.reward_scale = cfg.algorithm.get("reward_scale", 1.0)

    # ------------------------------------------------------------------

    def run(self) -> None:
        obs, _ = self.env.reset()
        self.reward_fn.reset()

        episode_reward = 0.0
        episode_steps = 0
        episode = 0
        best_reward = float("-inf")

        for step in range(1, self.total_steps + 1):

            # ── action selection ──────────────────────────────────────────
            if step < self.learning_starts:
                action = np.random.uniform(-1.0, 1.0, self.env.action_dim)
            else:
                action = self.agent.select_action(obs)

            # ── env step ──────────────────────────────────────────────────
            next_obs, _, terminated, truncated, info = self.env.step(action)
            reward = self.reward_fn.compute(self.env.model, self.env.data, info)
            scaled_reward = reward * self.reward_scale
            done = terminated or truncated

            self.buffer.add(obs, action, scaled_reward, next_obs, float(terminated))
            obs = next_obs
            episode_reward += reward
            episode_steps += 1

            # ── gradient updates ──────────────────────────────────────────
            ready = len(self.buffer) >= self.learning_starts
            if ready and step % self.update_every == 0:
                for _ in range(self.gradient_steps):
                    batch = self.buffer.sample(self.batch_size)
                    metrics = self.agent.update(batch)

                self.logger.log_dict(metrics, step)

            # ── periodic console log ──────────────────────────────────────
            if step % self.log_every == 0:
                log.info(f"[{step:>8d}/{self.total_steps}] "
                         f"ep={episode}  buf={len(self.buffer)}")

            # ── episode end ───────────────────────────────────────────────
            if done:
                episode += 1
                log.info(f"Episode {episode:>5d} | step {step:>8d} | "
                         f"ep_steps {episode_steps:>4d} | reward {episode_reward:+.2f}")
                self.logger.log("train/episode_reward", episode_reward, step)
                self.logger.log("train/episode_steps", episode_steps, step)
                self.logger.log("train/episode", episode, step)

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    self._save("best")

                obs, _ = self.env.reset()
                self.reward_fn.reset()
                episode_reward = 0.0
                episode_steps = 0

            # ── periodic checkpoint ───────────────────────────────────────
            if self.checkpoint_every > 0 and step % self.checkpoint_every == 0:
                self._save(f"step_{step}")

        # ── final save ────────────────────────────────────────────────────
        self._save("final")
        self.env.close()
        self.logger.close()
        log.info("Training complete.")

    # ------------------------------------------------------------------

    def _save(self, tag: str) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = str(self.checkpoint_dir / f"{tag}.pt")
        self.agent.save(path)
        log.info(f"Checkpoint saved → {path}")
