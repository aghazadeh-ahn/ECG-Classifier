"""Dilated residual network. Dilation is what sees rhythm across the recording."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ResidualStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dilation: int) -> None:
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.skip = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.skip(values)
        output = self.relu(self.bn1(self.conv1(values)))
        output = self.bn2(self.conv2(output))
        length = min(output.shape[-1], residual.shape[-1])
        output = self.relu(output[..., :length] + residual[..., :length])
        return output, _resize_mask(mask, length)


class DilatedResNet(nn.Module):
    def __init__(self, channels: list[int], dilations: list[int], dropout: float) -> None:
        super().__init__()
        if len(channels) != len(dilations):
            raise ValueError("Each stage needs one dilation")
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )
        stages: list[ResidualStage] = []
        in_channels = channels[0]
        for out_channels, dilation in zip(channels, dilations):
            stages.append(ResidualStage(in_channels, out_channels, stride=2, dilation=dilation))
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(channels[-1], 4)

    def embed(self, signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = self.stem(signal.unsqueeze(1))
        mask = _resize_mask(mask, values.shape[-1])
        for stage in self.stages:
            values, mask = stage(values, mask)
        return _masked_mean(values, mask)

    def forward(self, signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.embed(signal, mask)))


def _resize_mask(mask: torch.Tensor, length: int) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(1)
    if mask.shape[-1] == length:
        return mask
    return F.adaptive_max_pool1d(mask.float(), length)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(1)
    weights = _resize_mask(mask, values.shape[-1]).to(dtype=values.dtype)
    return (values * weights).sum(dim=-1) / weights.sum(dim=-1).clamp(min=1.0)
