import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    """Lightweight CNN encoder for eye-in-hand RGB observations."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 128, img_h: int = 84, img_w: int = 84) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Determine flattened size with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_h, img_w)
            flat_dim = self.net(dummy).shape[1]

        self.fc = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x / 255.0))
