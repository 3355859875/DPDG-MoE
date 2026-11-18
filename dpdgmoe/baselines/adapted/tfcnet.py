"""TFCNet adapted to the benchmark protocol.

Endogenous power patches and exogenous NWP tokens are encoded jointly in time,
a frequency-domain MLP filters the power history, and a gate fuses both. The
original KAN output head is replaced by an MLP.
"""

from __future__ import annotations

import torch
from torch import nn

from .common import FutureHead


class _FrequencyMLP(nn.Module):
    def __init__(self, length: int, hidden: int):
        super().__init__()
        self.real = nn.Linear(length // 2 + 1, length // 2 + 1)
        self.imag = nn.Linear(length // 2 + 1, length // 2 + 1)
        self.projection = nn.Sequential(nn.Linear(length, hidden), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x, dim=-1, norm="ortho")
        real = torch.relu(self.real(spectrum.real))
        imag = torch.relu(self.imag(spectrum.imag))
        filtered = torch.complex(real, imag)
        return self.projection(torch.fft.irfft(filtered, n=x.shape[-1], dim=-1, norm="ortho"))


class TFCNet(nn.Module):
    """TFCNet-style endogenous/exogenous time-frequency collaboration."""

    def __init__(self, history_dim: int, weather_dim: int, history_len: int, horizon: int,
                 d_model: int = 64, heads: int = 4, layers: int = 2, dropout: float = 0.05,
                 future_dim: int = 16, patch_len: int = 24, stride: int = 24):
        super().__init__()
        self.history_dim = int(history_dim)
        self.weather_dim = int(weather_dim)
        self.history_len = int(history_len)
        self.horizon = int(horizon)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.patch_count = (history_len - patch_len) // stride + 1
        self.endogenous_patch = nn.Linear(patch_len, d_model)
        self.exogenous_token = nn.Linear(history_len, d_model)
        self.token_position = nn.Parameter(
            torch.randn(1, self.patch_count + weather_dim, d_model) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(encoder_layer, layers)
        self.frequency = _FrequencyMLP(history_len, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, horizon))
        self.future_head = FutureHead(future_dim)

    def forward(self, x: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.history_len, self.history_dim + self.weather_dim):
            raise ValueError(
                f"TFCNet expected [B,{self.history_len},{self.history_dim + self.weather_dim}], got {tuple(x.shape)}"
            )
        endogenous = x[:, :, self.history_dim - 1:self.history_dim]
        exogenous = x[:, :, self.history_dim:self.history_dim + self.weather_dim]
        endogenous_tokens = self.endogenous_patch(
            endogenous.squeeze(-1).unfold(-1, self.patch_len, self.stride)
        )
        exogenous_tokens = self.exogenous_token(exogenous.transpose(1, 2))
        temporal_tokens = torch.cat([endogenous_tokens, exogenous_tokens], dim=1)
        temporal = self.temporal(temporal_tokens + self.token_position).mean(dim=1)
        frequency = self.frequency(endogenous.squeeze(-1))
        fused = temporal * self.gate(torch.cat([temporal, frequency], dim=-1))
        return self.head(fused) + self.future_head(future)
