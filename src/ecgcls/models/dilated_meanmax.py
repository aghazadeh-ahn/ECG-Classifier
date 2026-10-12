"""Light dilated network with mean and max pooling, plus a local-RR adjustment near the Normal/Other boundary."""

from __future__ import annotations

import torch
from torch import nn

from ecgcls.local_rr import N_FEATURES
from ecgcls.models.dilated_resnet import DilatedResNet, _masked_max, _masked_mean


class DilatedMeanMax(nn.Module):
    def __init__(self, channels: list[int], dilations: list[int], dropout: float) -> None:
        super().__init__()
        self.backbone = DilatedResNet(channels, dilations, dropout)
        self.backbone.head = nn.Identity()
        self.head = nn.Linear(channels[-1] * 2, 4)
        self.local = nn.Linear(N_FEATURES, 2)
        nn.init.zeros_(self.local.weight)
        nn.init.zeros_(self.local.bias)

    def forward(self, signal: torch.Tensor, mask: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
        values, feature_mask = self.backbone.feature_map(signal, mask)
        pooled = torch.cat((_masked_mean(values, feature_mask), _masked_max(values, feature_mask)), dim=1)
        logits = self.head(self.backbone.dropout(pooled))
        normal = logits[:, 0]
        other = logits[:, 2]
        rivals = torch.maximum(logits[:, 1], logits[:, 3])
        gate = torch.sigmoid(normal - other) * torch.sigmoid(1.0 - (normal - other)) * torch.sigmoid(normal - rivals)
        delta = self.local(local) * gate.unsqueeze(1)
        adjusted = logits.clone()
        adjusted[:, 0] = logits[:, 0] + delta[:, 0]
        adjusted[:, 2] = logits[:, 2] + delta[:, 1]
        return adjusted
