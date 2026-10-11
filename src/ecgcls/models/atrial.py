"""CNN on each pre-QRS window, LSTM over beats, then mean and max."""

from __future__ import annotations

import torch
from torch import nn


class AtrialCnnLstm(nn.Module):
    def __init__(
        self,
        channels: list[int],
        kernels: list[int],
        lstm_hidden: int,
        lstm_layers: int,
        dropout: float,
    ) -> None:
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
        self.rr_scale = nn.Linear(1, channels[-1])
        self.lstm = nn.LSTM(
            input_size=channels[-1] * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden * 2, 4)

    def forward(self, atrial: torch.Tensor, rr_before: torch.Tensor, beat_mask: torch.Tensor) -> torch.Tensor:
        batch, count, length = atrial.shape
        values = self.encoder(atrial.reshape(batch * count, 1, length))
        shape = values.mean(dim=-1).view(batch, count, -1)
        rhythm = self.rr_scale(rr_before.unsqueeze(-1))
        tokens = torch.cat((shape, rhythm), dim=-1)
        lengths = beat_mask.sum(dim=1).clamp(min=1).to(dtype=torch.long).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(tokens, lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.lstm(packed)
        sequence, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=count)
        weights = beat_mask.unsqueeze(-1).to(dtype=sequence.dtype)
        mean = (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        filled = sequence.masked_fill(beat_mask.unsqueeze(-1) < 0.5, torch.finfo(sequence.dtype).min)
        maximum = filled.max(dim=1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.head(self.dropout(torch.cat((mean, maximum), dim=1)))
