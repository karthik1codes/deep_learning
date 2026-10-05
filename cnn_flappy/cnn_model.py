"""CNN architectures for Flappy Bird: policy (BC) and Q-network (DQN)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlappyCNN(nn.Module):
    """
    Classic Atari-style CNN.

    Input:  (B, 4, 84, 84)  — stacked grayscale frames
    Output: (B, n_actions)  — logits (BC) or Q-values (DQN)
    """

    def __init__(self, in_channels: int = 4, n_actions: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64 * 7 * 7, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        return self.head(x)

    def forward_with_features(self, x: torch.Tensor):
        """Return logits plus intermediate maps for CNN visualization."""
        feats = {}
        h1 = F.relu(self.conv1(x))
        feats["conv1"] = h1
        h2 = F.relu(self.conv2(h1))
        feats["conv2"] = h2
        h3 = F.relu(self.conv3(h2))
        feats["conv3"] = h3
        flat = torch.flatten(h3, 1)
        hfc = F.relu(self.fc(flat))
        feats["fc"] = hfc
        out = self.head(hfc)
        return out, feats


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
