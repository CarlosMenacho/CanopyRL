import torch
import torch.nn as nn
from torch.distributions import Normal

from .encoder import ImageEncoder

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class Actor(nn.Module):
    """Gaussian policy network (for SAC) with eye-in-hand image encoder."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 img_h: int = 128, img_w: int = 128, latent_dim: int = 128) -> None:
        super().__init__()
        self.encoder = ImageEncoder(latent_dim=latent_dim, img_h=img_h, img_w=img_w)
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def _encode(self, state: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
        # pixels: (B, H, W, C) uint8 → (B, C, H, W) float
        pix = pixels.permute(0, 3, 1, 2).float()
        return torch.cat([state, self.encoder(pix)], dim=-1)

    def forward(self, state: torch.Tensor, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(self._encode(state, pixels))
        mu = self.mu_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, state: torch.Tensor, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (action, log_prob) via reparameterisation + tanh squashing."""
        mu, log_std = self(state, pixels)
        std = log_std.exp()
        dist = Normal(mu, std)
        x = dist.rsample()
        action = torch.tanh(x)
        log_prob = (dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return action, log_prob
