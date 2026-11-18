"""SolarCAST adapted to single-station PV forecasting."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .common import FutureConditioner, patchify


class _SparseDirectedGraph(nn.Module):
    def __init__(self, d_model: int, nodes: int, topk: int):
        super().__init__()
        self.left = nn.Linear(d_model, d_model)
        self.right = nn.Linear(d_model, d_model)
        self.topk = min(max(1, int(topk)), int(nodes))

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        left = torch.tanh(self.left(nodes))
        right = torch.tanh(self.right(nodes))
        scores = torch.relu(
            torch.matmul(left, right.transpose(-1, -2))
            - torch.matmul(right, left.transpose(-1, -2))
        )
        if self.topk < scores.shape[-1]:
            values, indices = scores.topk(self.topk, dim=-1)
            scores = torch.zeros_like(scores).scatter(-1, indices, values)
        return torch.softmax(scores, dim=-1)


class SolarCAST(nn.Module):
    """SolarCAST-inspired synchronous-graph and lagged-transformer adapter.

    The source method expects target-site and neighboring-station GHI.  The
    S-series benchmark has one station per file, so normalized local variables
    are used as graph nodes and the substitution is recorded in result metadata.
    """

    def __init__(
        self,
        input_dim: int,
        history_len: int,
        horizon: int,
        d_model: int = 48,
        heads: int = 4,
        layers: int = 1,
        graph_depth: int = 2,
        topk: int = 5,
        patch_len: int = 24,
        stride: int = 24,
        dropout: float = 0.05,
        target_idx: int = 0,
        future_dim: int = 16,
    ):
        super().__init__()
        if history_len < patch_len:
            raise ValueError("history_len must be at least patch_len")
        self.input_dim = int(input_dim)
        self.history_len = int(history_len)
        self.horizon = int(horizon)
        self.target_idx = int(target_idx)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.patch_count = (history_len - patch_len) // stride + 1
        self.graph_depth = int(graph_depth)

        self.patch_projection = nn.Linear(patch_len, d_model)
        self.node_embedding = nn.Parameter(torch.randn(1, input_dim, 1, d_model) * 0.02)
        self.patch_position = nn.Parameter(
            torch.randn(1, 1, self.patch_count, d_model) * 0.02
        )
        self.graph = _SparseDirectedGraph(d_model, input_dim, topk)
        self.graph_projection = nn.Linear(d_model, d_model)
        self.graph_norm = nn.LayerNorm(d_model)

        self.lagged_conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, dilation=2, padding=2
        )
        self.lag_gate = nn.Linear(2 * d_model, d_model)
        lag_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.lagged_transformer = nn.TransformerEncoder(
            lag_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, horizon),
        )
        self.future = FutureConditioner(future_dim, d_model)

    def forward(self, x: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        expected = (self.history_len, self.input_dim)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"SolarCAST expected [B,{expected[0]},{expected[1]}], got {tuple(x.shape)}"
            )

        patches = patchify(x, self.patch_len, self.stride)
        tokens = (
            self.patch_projection(patches) + self.node_embedding + self.patch_position
        )
        nodes = tokens.mean(dim=2)
        for _ in range(self.graph_depth):
            adjacency = self.graph(nodes)
            message = torch.bmm(adjacency, nodes)
            nodes = self.graph_norm(nodes + self.graph_projection(message))
        graph_target = nodes[:, self.target_idx]

        batch, channels, patch_count, dim = tokens.shape
        lagged = tokens.reshape(batch * channels, patch_count, dim).transpose(1, 2)
        lagged = F.gelu(self.lagged_conv(lagged)).transpose(1, 2)
        lagged = lagged.reshape(batch, channels, patch_count, dim)
        target_tokens = lagged[:, self.target_idx]
        if channels > 1:
            auxiliary = torch.cat(
                [lagged[:, : self.target_idx], lagged[:, self.target_idx + 1 :]], dim=1
            ).mean(dim=1)
        else:
            auxiliary = target_tokens
        gate = torch.sigmoid(
            self.lag_gate(torch.cat([target_tokens, auxiliary], dim=-1))
        )
        lagged_target = self.lagged_transformer(target_tokens + gate * auxiliary).mean(
            dim=1
        )

        base = self.fusion(torch.cat([graph_target, lagged_target], dim=-1))
        return self.future(base, future)

