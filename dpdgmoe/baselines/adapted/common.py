"""Future-NWP heads shared by the structural adapters."""

from __future__ import annotations

import torch
from torch import nn


class FutureHead(nn.Module):
    """Additive step-wise correction from future NWP and calendar features."""

    def __init__(self, future_dim: int):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(future_dim, 1), nn.Tanh())

    def forward(self, future: torch.Tensor) -> torch.Tensor:
        return self.projection(future).squeeze(-1)


class FutureConditioner(nn.Module):
    """Step-wise multiplicative gate and additive bias from future NWP and calendar features."""

    def __init__(self, future_dim: int, hidden_dim: int):
        super().__init__()
        self.bias = nn.Sequential(
            nn.Linear(future_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(future_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, base: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        if base.shape[:2] != future.shape[:2]:
            raise ValueError(
                f"Future conditioner expected matching [B,H], got {tuple(base.shape)} and {tuple(future.shape)}"
            )
        return base * (1.0 + 0.1 * self.gate(future).squeeze(-1)) + self.bias(
            future
        ).squeeze(-1)


def patchify(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    return x.transpose(1, 2).unfold(-1, patch_len, stride)

