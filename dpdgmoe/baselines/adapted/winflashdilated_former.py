"""WinFlashDilated-former adapted to the benchmark protocol.

Exact multi-scale window attention (in place of the FlashAttention-3 kernel)
and dilated-convolution feed-forward layers.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .common import FutureConditioner


class _MultiScaleWindowAttention(nn.Module):
    def __init__(
        self, d_model: int, heads: int, window_sizes: tuple[int, ...], dropout: float
    ):
        super().__init__()
        self.window_sizes = tuple(int(size) for size in window_sizes)
        self.attention = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
                for _ in self.window_sizes
            ]
        )
        self.scale_logits = nn.Parameter(torch.zeros(len(self.window_sizes)))

    @staticmethod
    def _apply_window_attention(
        x: torch.Tensor, size: int, attention: nn.MultiheadAttention
    ) -> torch.Tensor:
        batch, length, dim = x.shape
        pad = (-length) % size
        padded = F.pad(x, (0, 0, 0, pad)) if pad else x
        window_count = padded.shape[1] // size
        windows = padded.reshape(batch, window_count, size, dim).reshape(
            batch * window_count, size, dim
        )
        attended, _ = attention(windows, windows, windows, need_weights=False)
        restored = attended.reshape(batch, window_count, size, dim).reshape(
            batch, padded.shape[1], dim
        )
        return restored[:, :length]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.scale_logits, dim=0)
        outputs = [
            weight * self._apply_window_attention(x, size, attention)
            for weight, size, attention in zip(
                weights, self.window_sizes, self.attention
            )
        ]
        return torch.stack(outputs, dim=0).sum(dim=0)


class _DilatedEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        window_sizes: tuple[int, ...],
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = _MultiScaleWindowAttention(
            d_model, heads, window_sizes, dropout
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, 4 * d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            4 * d_model,
            d_model,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x)))
        residual = x
        y = self.norm2(x).transpose(1, 2)
        y = self.conv2(self.dropout(F.gelu(self.conv1(y)))).transpose(1, 2)
        return residual + self.dropout(y)


def _sinusoidal_position(length: int, d_model: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    encoding = torch.zeros(length, d_model, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * divisor)
    encoding[:, 1::2] = torch.cos(position * divisor[: encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0)


class WinFlashDilatedFormer(nn.Module):
    """WinFlashDilated-former-inspired multi-scale exact-attention adapter."""

    def __init__(
        self,
        input_dim: int,
        history_len: int,
        horizon: int,
        d_model: int = 64,
        heads: int = 4,
        layers: int = 2,
        window_sizes: tuple[int, ...] = (8, 16, 32, 64),
        dilation: int = 4,
        dropout: float = 0.05,
        future_dim: int = 16,
        pooled_steps: int = 24,
    ):
        super().__init__()
        valid_windows = tuple(size for size in window_sizes if 1 <= size <= history_len)
        if not valid_windows:
            raise ValueError("At least one window size must fit inside history_len")
        self.input_dim = int(input_dim)
        self.history_len = int(history_len)
        self.horizon = int(horizon)
        self.input_projection = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
        self.register_buffer(
            "position", _sinusoidal_position(history_len, d_model), persistent=False
        )
        self.layers = nn.ModuleList(
            [
                _DilatedEncoderLayer(d_model, heads, valid_windows, dilation, dropout)
                for _ in range(layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.pooled_steps = min(int(pooled_steps), history_len)
        self.pool = nn.AdaptiveAvgPool1d(self.pooled_steps)
        self.head = nn.Linear(d_model * self.pooled_steps, horizon)
        self.future = FutureConditioner(future_dim, d_model)

    def forward(self, x: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        expected = (self.history_len, self.input_dim)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"WinFlashDilated-former expected [B,{expected[0]},{expected[1]}], got {tuple(x.shape)}"
            )
        encoded = (
            self.input_projection(x.transpose(1, 2)).transpose(1, 2) + self.position
        )
        for layer in self.layers:
            encoded = layer(encoded)
        encoded = self.final_norm(encoded)
        pooled = self.pool(encoded.transpose(1, 2)).flatten(1)
        base = self.head(pooled)
        return self.future(base, future)

