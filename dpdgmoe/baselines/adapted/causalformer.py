"""CausalFormer adapted to forecasting (the original task is temporal causal discovery)."""

from __future__ import annotations

import math

import torch
from torch import nn

from .common import FutureConditioner, patchify


class CausalFormer(nn.Module):
    """CausalFormer-inspired forecasting adapter.

    The source paper targets temporal causal discovery.  For the forecasting
    benchmark, variable-wise history patches are encoded, an asymmetric
    multi-head causal graph is learned across variables, and the target series
    tokens are decoded directly to the requested horizon.
    """

    def __init__(
        self,
        input_dim: int,
        history_len: int,
        horizon: int,
        d_model: int = 48,
        heads: int = 4,
        layers: int = 1,
        patch_len: int = 24,
        stride: int = 24,
        dropout: float = 0.05,
        target_idx: int = 0,
        future_dim: int = 16,
    ):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        if history_len < patch_len:
            raise ValueError("history_len must be at least patch_len")
        self.input_dim = int(input_dim)
        self.history_len = int(history_len)
        self.horizon = int(horizon)
        self.target_idx = int(target_idx)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.heads = int(heads)
        self.head_dim = d_model // heads
        self.patch_count = (history_len - patch_len) // stride + 1

        self.patch_projection = nn.Linear(patch_len, d_model)
        self.variable_embedding = nn.Parameter(
            torch.randn(1, input_dim, 1, d_model) * 0.02
        )
        self.patch_position = nn.Parameter(
            torch.randn(1, 1, self.patch_count, d_model) * 0.02
        )
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.causal_bias = nn.Parameter(torch.zeros(heads, input_dim, input_dim))
        self.causal_output = nn.Linear(d_model, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.head = nn.Linear(self.patch_count * d_model, horizon)
        self.future = FutureConditioner(future_dim, d_model)

    def forward(self, x: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        expected = (self.history_len, self.input_dim)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"CausalFormer expected [B,{expected[0]},{expected[1]}], got {tuple(x.shape)}"
            )

        patches = patchify(x, self.patch_len, self.stride)
        tokens = (
            self.patch_projection(patches)
            + self.variable_embedding
            + self.patch_position
        )
        summary = tokens.mean(dim=2)
        batch = x.shape[0]

        query = (
            self.query(summary)
            .view(batch, self.input_dim, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.key(summary)
            .view(batch, self.input_dim, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.value(summary)
            .view(batch, self.input_dim, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores + self.causal_bias.unsqueeze(0)
        adjacency = torch.softmax(scores, dim=-1)
        message = torch.matmul(adjacency, value)
        target_context = message[:, :, self.target_idx].reshape(batch, -1)
        target_context = self.causal_output(target_context).unsqueeze(1)

        target_tokens = tokens[:, self.target_idx] + target_context
        encoded = self.temporal(target_tokens)
        base = self.head(encoded.flatten(1))
        return self.future(base, future)

