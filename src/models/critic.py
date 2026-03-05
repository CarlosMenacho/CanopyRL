import torch
import torch.nn as nn

from .encoder import ImageEncoder


class Critic(nn.Module):
    """Twin Q-network (for SAC / TD3) with eye-in-hand image encoder."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 img_h: int = 128, img_w: int = 128, latent_dim: int = 128) -> None:
        super().__init__()
        self.encoder = ImageEncoder(latent_dim=latent_dim, img_h=img_h, img_w=img_w)
        in_dim = state_dim + latent_dim + action_dim

        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _encode(self, state: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
        # pixels: (B, H, W, C) uint8 → (B, C, H, W) float
        pix = pixels.permute(0, 3, 1, 2).float()
        return torch.cat([state, self.encoder(pix)], dim=-1)

    def forward(self, state: torch.Tensor, pixels: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([self._encode(state, pixels), action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, state: torch.Tensor, pixels: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(state, pixels, action)
        return torch.min(q1, q2)
