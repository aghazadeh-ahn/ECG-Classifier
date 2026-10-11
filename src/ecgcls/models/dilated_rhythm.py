"""Light dilated network plus extrasystole and relative-RR features at the classifier."""

from __future__ import annotations

import torch
from torch import nn

from ecgcls.models.dilated_resnet import DilatedResNet
from ecgcls.rhythm import N_FEATURES


class DilatedRhythmNet(nn.Module):
    def __init__(self, channels: list[int], dilations: list[int], dropout: float) -> None:
        super().__init__()
        self.backbone = DilatedResNet(channels, dilations, dropout)
        self.backbone.head = nn.Identity()
        self.rhythm = nn.Linear(N_FEATURES, 16)
        self.head = nn.Linear(channels[-1] + 16, 4)

    def forward(self, signal: torch.Tensor, mask: torch.Tensor, rhythm: torch.Tensor) -> torch.Tensor:
        pooled = self.backbone.embed(signal, mask)
        joined = torch.cat((pooled, torch.relu(self.rhythm(rhythm))), dim=1)
        return self.head(self.backbone.dropout(joined))
