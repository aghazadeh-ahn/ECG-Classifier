"""Small CNN on each 0.6 s beat, then the mean and the max across beats."""

from __future__ import annotations

import torch
from torch import nn


class BeatCnn(nn.Module):
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
        self.head = nn.Linear(channels[-1] * 2, 4)

    def forward(self, beats: torch.Tensor, beat_mask: torch.Tensor) -> torch.Tensor:
        batch, count, length = beats.shape
        values = self.encoder(beats.reshape(batch * count, 1, length))
        embedding = values.mean(dim=-1).view(batch, count, -1)
        weights = beat_mask.unsqueeze(-1).to(dtype=embedding.dtype)
        mean = (embedding * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        filled = embedding.masked_fill(beat_mask.unsqueeze(-1) < 0.5, torch.finfo(embedding.dtype).min)
        maximum = filled.max(dim=1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.head(self.dropout(torch.cat((mean, maximum), dim=1)))
