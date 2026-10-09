"""One-dimensional convolutional network. Its field of view stays local."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Cnn1d(nn.Module):
    def __init__(self, channels: list[int], kernels: list[int], dropout: float) -> None:
        super().__init__()
        if len(channels) != len(kernels):
            raise ValueError("Each convolutional stage needs one kernel size")
        stages: list[nn.Module] = []
        in_channels = 1
        for out_channels, kernel in zip(channels, kernels):
            stages.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel, stride=2, padding=kernel // 2, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*stages)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(channels[-1], 4)

    def forward(self, signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = self.encoder(signal.unsqueeze(1))
        mask_ds = F.interpolate(mask.unsqueeze(1).float(), size=values.shape[-1], mode="nearest").squeeze(1)
        weights = mask_ds.unsqueeze(1)
        pooled = (values * weights).sum(dim=-1) / weights.sum(dim=-1).clamp(min=1.0)
        return self.head(self.dropout(pooled))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
