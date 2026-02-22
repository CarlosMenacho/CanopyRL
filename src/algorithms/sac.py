import copy
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig

from .base import BaseAlgorithm
from src.models.actor import Actor
from src.models.critic import Critic


class SAC(BaseAlgorithm):
    """Soft Actor-Critic (Haarnoja et al., 2018)."""

    def __init__(self, cfg: DictConfig, obs_dim: int, action_dim: int, device: torch.device) -> None:
        super().__init__(cfg, obs_dim, action_dim, device)

        hidden = cfg.get("hidden_dim", 256)
        lr_actor = cfg.get("lr_actor", 3e-4)
        lr_critic = cfg.get("lr_critic", 3e-4)
        lr_alpha = cfg.get("lr_alpha", 3e-4)

        self.gamma = cfg.get("gamma", 0.99)
        self.tau = cfg.get("tau", 0.005)

        # target_entropy: use value from config if explicitly set, else -action_dim
        te = cfg.get("target_entropy", None)
        self.target_entropy = float(te) if te is not None else float(-action_dim)

        self.actor = Actor(obs_dim, action_dim, hidden).to(device)
        self.critic = Critic(obs_dim, action_dim, hidden).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # init log_alpha so that alpha starts at init_alpha (default 1.0)
        init_alpha = float(cfg.get("init_alpha", 1.0))
        self.log_alpha = torch.tensor(
            [torch.log(torch.tensor(init_alpha)).item()],
            requires_grad=True,
            device=device,
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

    # ------------------------------------------------------------------

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                mu, _ = self.actor(obs_t)
                action = torch.tanh(mu)
            else:
                action, _ = self.actor.sample(obs_t)
        return action.cpu().numpy().squeeze(0)

    def update(self, batch: dict) -> dict[str, float]:
        obs = torch.FloatTensor(batch["obs"]).to(self.device)
        action = torch.FloatTensor(batch["action"]).to(self.device)
        reward = torch.FloatTensor(batch["reward"]).unsqueeze(1).to(self.device)
        next_obs = torch.FloatTensor(batch["next_obs"]).to(self.device)
        done = torch.FloatTensor(batch["done"]).unsqueeze(1).to(self.device)

        # --- Critic update ---
        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(next_obs)
            q1_t, q2_t = self.critic_target(next_obs, next_action)
            q_target = reward + self.gamma * (1 - done) * (
                torch.min(q1_t, q2_t) - self.alpha * next_log_pi
            )

        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # --- Actor update ---
        pi, log_pi = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, pi)
        actor_loss = (self.alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # --- Alpha update ---
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # --- Soft target update ---
        for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.copy_(self.tau * p.data + (1 - self.tau) * p_t.data)

        return {
            "loss/critic": critic_loss.item(),
            "loss/actor": actor_loss.item(),
            "loss/alpha": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }

    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "log_alpha": self.log_alpha,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.log_alpha = ckpt["log_alpha"]
