"""Best dilated model, plus the single beat that least matches the record's median beat."""

from __future__ import annotations

import torch
from torch import nn

from ecgcls.models.dilated_resnet import DilatedResNet, _masked_max, _masked_mean


class DilatedOutlier(nn.Module):
    def __init__(self, channels: list[int], dilations: list[int], dropout: float) -> None:
        super().__init__()
        self.backbone = DilatedResNet(channels, dilations, dropout)
        self.backbone.head = nn.Identity()
        self.beat = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(channels[-1] * 2 + 32 + 1, 4)
        nn.init.zeros_(self.head.weight[:, channels[-1] * 2 :])
        nn.init.zeros_(self.head.bias)
        self.local = nn.Linear(4, 2)
        nn.init.zeros_(self.local.weight)
        nn.init.zeros_(self.local.bias)

    def forward(
        self,
        signal: torch.Tensor,
        mask: torch.Tensor,
        local: torch.Tensor,
        beat: torch.Tensor,
        oddness: torch.Tensor,
    ) -> torch.Tensor:
        values, feature_mask = self.backbone.feature_map(signal, mask)
        pooled = torch.cat((_masked_mean(values, feature_mask), _masked_max(values, feature_mask)), dim=1)
        encoded = self.beat(beat.unsqueeze(1)).mean(dim=-1)
        joined = torch.cat((pooled, encoded, oddness.unsqueeze(1)), dim=1)
        logits = self.head(self.backbone.dropout(joined))
        normal = logits[:, 0]
        other = logits[:, 2]
        rivals = torch.maximum(logits[:, 1], logits[:, 3])
        gate = torch.sigmoid(normal - other) * torch.sigmoid(1.0 - (normal - other)) * torch.sigmoid(normal - rivals)
        delta = self.local(local) * gate.unsqueeze(1)
        adjusted = logits.clone()
        adjusted[:, 0] = logits[:, 0] + delta[:, 0]
        adjusted[:, 2] = logits[:, 2] + delta[:, 1]
        return adjusted
