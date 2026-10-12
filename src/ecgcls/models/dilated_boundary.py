"""Mean-max dilated net with artifact-aware max pooling and Normal/Other boundary features."""

from __future__ import annotations

import torch
from torch import nn

from ecgcls.boundary import N_FEATURES
from ecgcls.models.dilated_resnet import DilatedResNet, _masked_max, _masked_mean


class DilatedBoundary(nn.Module):
    def __init__(self, channels: list[int], dilations: list[int], dropout: float) -> None:
        super().__init__()
        self.backbone = DilatedResNet(channels, dilations, dropout)
        self.backbone.head = nn.Identity()
        self.head = nn.Linear(channels[-1] * 2, 4)
        self.boundary = nn.Linear(N_FEATURES, 2)
        nn.init.zeros_(self.boundary.weight)
        nn.init.zeros_(self.boundary.bias)
        # When artifact is low, trust max; when high, fall back toward mean.
        self.artifact_gain = nn.Parameter(torch.tensor(1.0))
        self.artifact_bias = nn.Parameter(torch.tensor(2.0))

    def forward(self, signal: torch.Tensor, mask: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        values, feature_mask = self.backbone.feature_map(signal, mask)
        mean = _masked_mean(values, feature_mask)
        maximum = _masked_max(values, feature_mask)
        artifact = features[:, 0].unsqueeze(1)
        max_weight = torch.sigmoid(self.artifact_bias - self.artifact_gain * artifact)
        blended = maximum * max_weight + mean * (1.0 - max_weight)
        logits = self.head(self.backbone.dropout(torch.cat((mean, blended), dim=1)))
        normal = logits[:, 0]
        other = logits[:, 2]
        rivals = torch.maximum(logits[:, 1], logits[:, 3])
        gate = torch.sigmoid(normal - other) * torch.sigmoid(1.0 - (normal - other)) * torch.sigmoid(normal - rivals)
        delta = self.boundary(features) * gate.unsqueeze(1)
        adjusted = logits.clone()
        adjusted[:, 0] = logits[:, 0] + delta[:, 0]
        adjusted[:, 2] = logits[:, 2] + delta[:, 1]
        return adjusted
