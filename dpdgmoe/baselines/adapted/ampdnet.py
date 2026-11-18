"""AMPDNet (ConvPVNet) adapted to the benchmark protocol.

Adaptive multi-scale depthwise convolution (AMSConv) with variable- and
feature-wise gated linear units; the power channel is decoded directly to the
forecast horizon and corrected by a future-NWP head.
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from .common import FutureHead


class _GroupedGLU(nn.Module):
    def __init__(self, channels: int, groups: int, expansion: int = 2, dropout: float = 0.05):
        super().__init__()
        hidden = channels * expansion
        self.left = nn.Conv1d(channels, hidden, 1, groups=groups)
        self.right = nn.Conv1d(channels, hidden, 1, groups=groups)
        self.out = nn.Conv1d(hidden, channels, 1, groups=groups)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = self.left(x) * torch.sigmoid(self.right(x))
        return self.dropout(self.out(self.dropout(gated)))


class _AMSConv(nn.Module):
    """AMPDNet's parallel dilated depthwise convolution with learned mixing."""

    def __init__(self, channels: int, kernel_sizes=(5, 5, 5, 5), dilations=(1, 2, 3, 4)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel,
                    dilation=dilation,
                    padding="same",
                    groups=channels,
                )
                for kernel, dilation in zip(kernel_sizes, dilations)
            ]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(channels) for _ in self.convs])
        self.weights = nn.Parameter(torch.full((len(self.convs),), 1.0 / len(self.convs)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [weight * norm(conv(x)) for weight, conv, norm in zip(self.weights, self.convs, self.norms)]
        return torch.stack(branches, dim=0).sum(dim=0)


class AMPDNet(nn.Module):
    """AMPDNet/ConvPVNet adapted to the shared long-window protocol."""

    def __init__(self, input_dim: int, history_len: int, horizon: int, hidden_dim: int = 32,
                 patch_len: int = 4, stride: int = 2, layers: int = 2, dropout: float = 0.05,
                 target_idx: int = 0, future_dim: int = 16):
        super().__init__()
        self.input_dim = int(input_dim)
        self.history_len = int(history_len)
        self.horizon = int(horizon)
        self.target_idx = int(target_idx)
        self.embedding = nn.Conv1d(1, hidden_dim, kernel_size=patch_len, stride=stride)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "ams": _AMSConv(input_dim * hidden_dim),
                        "variable_glu": _GroupedGLU(input_dim * hidden_dim, input_dim, dropout=dropout),
                        "feature_glu": _GroupedGLU(input_dim * hidden_dim, hidden_dim, dropout=dropout),
                    }
                )
                for _ in range(layers)
            ]
        )
        patch_count = (history_len - patch_len) // stride + 1
        self.head = nn.Linear(hidden_dim * patch_count, horizon)
        self.future_head = FutureHead(future_dim)

    def forward(self, x: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.history_len, self.input_dim):
            raise ValueError(f"AMPDNet expected [B,{self.history_len},{self.input_dim}], got {tuple(x.shape)}")
        x = x.transpose(1, 2)
        embedded = self.embedding(x.unsqueeze(2).flatten(0, 1)).squeeze(1)
        # The public input has shape [B,M,L]; embed each variable independently.
        embedded = embedded.reshape(x.shape[0], self.input_dim, -1, embedded.shape[-1])
        for block in self.layers:
            residual = embedded
            z = rearrange(embedded, "b m d n -> b (m d) n")
            z = block["ams"](z)
            z = block["variable_glu"](z)
            z = rearrange(z, "b (m d) n -> b m d n", m=self.input_dim)
            z = rearrange(z, "b m d n -> b d m n")
            z = rearrange(z, "b d m n -> b (d m) n")
            z = block["feature_glu"](z)
            z = rearrange(z, "b (d m) n -> b m d n", m=self.input_dim)
            embedded = z + residual
        power = embedded[:, self.target_idx]
        return self.head(power.flatten(1)) + self.future_head(future)

