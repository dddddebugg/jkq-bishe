import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import Dataset


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.positional(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.head(x)


class SlidingWindowDataset(Dataset):
    def __init__(self, series: torch.Tensor, input_window: int, horizon: int):
        self.series = series
        self.input_window = input_window
        self.horizon = horizon

    def __len__(self) -> int:
        return self.series.shape[0] - self.input_window - self.horizon + 1

    def __getitem__(self, idx: int):
        x = self.series[idx : idx + self.input_window]
        y = self.series[
            idx + self.input_window : idx + self.input_window + self.horizon
        ]
        return x, y.view(-1)


@dataclass
class ModelConfig:
    input_dim: int = 1
    output_dim: int = 1
    input_window: int = 48
    horizon: int = 1
    d_model: int = 64
    nhead: int = 8
    num_layers: int = 3
    dim_feedforward: int = 128
    dropout: float = 0.1
