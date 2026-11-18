"""G-GraphTST adapted to the benchmark protocol.

Patch tokens with global tokens are encoded per variable by a Transformer, and a
dynamic top-k graph passes messages between variables.
"""

from __future__ import annotations

import torch
from torch import nn

from .common import FutureHead


class _DynamicGraph(nn.Module):
    def __init__(self, nodes: int, dim: int, topk: int):
        super().__init__()
        self.left = nn.Linear(dim, dim)
        self.right = nn.Linear(dim, dim)
        self.topk = min(int(topk), nodes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        left = torch.tanh(self.left(tokens))
        right = torch.tanh(self.right(tokens))
        scores = torch.relu(torch.matmul(left, right.transpose(-1, -2)) - torch.matmul(right, left.transpose(-1, -2)))
        if self.topk < scores.shape[-1]:
            values, indices = scores.topk(self.topk, dim=-1)
            sparse = torch.zeros_like(scores).scatter(-1, indices, values)
            scores = sparse
        return torch.softmax(scores, dim=-1)


class GGraphTST(nn.Module):
    """Global-token Transformer plus dynamic graph message passing."""

    def __init__(self, input_dim: int, history_len: int, horizon: int, d_model: int = 64,
                 patch_len: int = 16, stride: int = 8, heads: int = 4, layers: int = 2,
                 graph_depth: int = 2, topk: int = 4, dropout: float = 0.05,
                 target_idx: int = 0, future_dim: int = 16):
        super().__init__()
        self.input_dim = int(input_dim)
        self.history_len = int(history_len)
        self.horizon = int(horizon)
        self.target_idx = int(target_idx)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.patch_count = (history_len - patch_len) // stride + 1
        self.patch_projection = nn.Linear(patch_len, d_model)
        self.global_tokens = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=layers)
        self.graph = _DynamicGraph(input_dim, d_model, topk)
        self.graph_projection = nn.Linear(d_model, d_model)
        self.graph_depth = int(graph_depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model * self.patch_count, horizon)
        self.future_head = FutureHead(future_dim)

    def forward(self, x: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.history_len, self.input_dim):
            raise ValueError(f"G-GraphTST expected [B,{self.history_len},{self.input_dim}], got {tuple(x.shape)}")
        patches = x.transpose(1, 2).unfold(-1, self.patch_len, self.stride)
        tokens = self.patch_projection(patches)
        batch, nodes, count, dim = tokens.shape
        # Temporal attention is applied independently per node. Cross-node
        # interaction is handled by the dynamic graph below; flattening all
        # node patches into one attention sequence makes batch=128 infeasible
        # for a 1,440-step history and duplicates the graph branch's role.
        node_tokens = tokens.reshape(batch * nodes, count, dim)
        global_tokens = self.global_tokens.expand(batch * nodes, -1, -1)
        encoded = self.temporal(torch.cat([global_tokens, node_tokens], dim=1))
        tokens = encoded[:, 3:].reshape(batch, nodes, count, dim)
        for _ in range(self.graph_depth):
            summary = tokens.mean(dim=2)
            adjacency = self.graph(summary)
            message = torch.bmm(adjacency, summary)
            tokens = self.norm(tokens + self.graph_projection(message).unsqueeze(2))
        output = self.head(tokens.permute(0, 1, 3, 2).reshape(batch, nodes, dim * count)[:, self.target_idx])
        return output + self.future_head(future)

