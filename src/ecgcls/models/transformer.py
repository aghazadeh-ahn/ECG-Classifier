"""Transformer encoder on a CNN stem. Same signal and mask as the dilated network."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class SignalTransformer(nn.Module):
    def __init__(
        self,
        stem_channels: list[int],
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        stages: list[nn.Module] = []
        in_channels = 1
        for out_channels in stem_channels:
            stages.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
        self.stem = nn.Sequential(*stages)
        self.project = nn.Conv1d(stem_channels[-1], d_model, kernel_size=1, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 4)
        self.d_model = d_model

    def forward(self, signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = self.project(self.stem(signal.unsqueeze(1)))
        mask = _resize_mask(mask, values.shape[-1])
        tokens = values.transpose(1, 2)
        tokens = tokens + _sinusoidal_positions(tokens.shape[1], self.d_model, tokens.device, tokens.dtype)
        padding = mask.squeeze(1) < 0.5
        encoded = self.encoder(tokens, src_key_padding_mask=padding)
        weights = (~padding).unsqueeze(-1).to(dtype=encoded.dtype)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.head(self.dropout(pooled))


def _sinusoidal_positions(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim))
    encoding = torch.zeros(length, dim, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(position * div)
    encoding[:, 1::2] = torch.cos(position * div)
    return encoding.unsqueeze(0)


def _resize_mask(mask: torch.Tensor, length: int) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(1)
    if mask.shape[-1] == length:
        return mask
    return F.adaptive_max_pool1d(mask.float(), length)
