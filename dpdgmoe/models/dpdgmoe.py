"""DPDG-MoE: Daily Prior and Dynamic-Graph Mixture of Experts.

The forecast for step ``l`` of the window is

    u_l = clip(b_l + lambda_r * tanh(r_l), 0, 1.2)

where ``b_l`` is the daily prior from Daily Pattern Encoding (DPE), ``r_l`` is the
residual predicted by the Routed Mixture of Experts (RMoE), and the
Time-block Dynamic Graph (TDG) supplies the graph feature of each step.
All quantities are normalized by the power scale ``s_y``.

Equation numbers in the comments refer to the paper.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PRIOR_MODES = ("learned", "fixed", "adaptive", "none")
GRAPH_MODES = ("dynamic", "static", "none")
RESIDUAL_MODES = ("moe", "single", "none")


# --------------------------------------------------------------------- blocks
class _Chomp1d(nn.Module):
    """Remove right padding so that a dilated convolution keeps the length."""

    def __init__(self, size: int):
        super().__init__()
        self.size = int(size)

    def forward(self, x):
        return x if self.size <= 0 else x[:, :, : -self.size].contiguous()


class TCNBlock(nn.Module):
    """Two dilated convolutions (kernel 3) with GELU, dropout, skip and GroupNorm."""

    def __init__(self, in_dim: int, out_dim: int, dilation: int, dropout: float):
        super().__init__()
        padding = 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=padding, dilation=dilation),
            _Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=padding, dilation=dilation),
            _Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()
        self.norm = nn.GroupNorm(1, out_dim)

    def forward(self, x):
        return self.norm(self.net(x) + self.skip(x))


class TCN(nn.Module):
    """Stack of :class:`TCNBlock` on ``[B, T, C]`` sequences."""

    def __init__(self, in_dim: int, hidden_dim: int, dilations: tuple[int, ...], dropout: float):
        super().__init__()
        blocks, current = [], in_dim
        for dilation in dilations:
            blocks.append(TCNBlock(current, hidden_dim, dilation, dropout))
            current = hidden_dim
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class FutureEncoder(nn.Module):
    """E^f = LN(GRU_f(TCN_f(X^f))), Eq. (3)."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.tcn = TCN(in_dim, hidden_dim, (1, 2), dropout)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        encoded, _ = self.gru(self.tcn(x))
        return self.norm(encoded)


class TimeBlockGraph(nn.Module):
    """Sparse graph message passing over time-block nodes, Eqs. (12)-(14).

    ``mode="dynamic"`` computes the edge scores from the node features of each
    sample. ``mode="static"`` (ablation) uses learned sample-independent logits.
    Each node keeps its ``topk`` highest-scoring neighbors per head.
    """

    def __init__(self, hidden_dim: int, num_nodes: int, num_heads: int, topk: int, dropout: float, mode: str):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.mode = mode
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_nodes = int(num_nodes)
        self.topk = min(int(topk), self.num_nodes)
        if mode == "dynamic":
            self.q = nn.Linear(hidden_dim, hidden_dim)
            self.k = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.static_logits = nn.Parameter(torch.zeros(num_heads, self.num_nodes, self.num_nodes))
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.o = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def _split_heads(self, x):
        batch, nodes, _ = x.shape
        return x.view(batch, nodes, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, nodes):
        batch, num_nodes, hidden = nodes.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} graph nodes, received {num_nodes}.")
        if self.mode == "dynamic":
            score = torch.matmul(self._split_heads(self.q(nodes)), self._split_heads(self.k(nodes)).transpose(-2, -1))
            score = score / math.sqrt(self.head_dim)
        else:
            score = self.static_logits.unsqueeze(0).expand(batch, -1, -1, -1)
        values = self._split_heads(self.v(nodes))
        if self.topk < num_nodes:
            top_scores, top_indices = torch.topk(score, k=self.topk, dim=-1)
            score = torch.full_like(score, float("-inf")).scatter(-1, top_indices, top_scores)
        adjacency = self.dropout(torch.softmax(score, dim=-1))
        messages = torch.matmul(adjacency, values).transpose(1, 2).contiguous().view(batch, num_nodes, hidden)
        z1 = self.norm1(nodes + self.o(messages))
        z2 = self.norm2(z1 + self.ffn(z1))
        return z2, adjacency


class Expert(nn.Module):
    """Two-layer MLP expert F_i(c_l), Eq. (16)."""

    def __init__(self, context_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context):
        return self.net(context).squeeze(-1)


# ---------------------------------------------------------------------- model
class DPDGMoE(nn.Module):
    """Daily Prior and Dynamic-Graph Mixture of Experts.

    Args:
        history_dim: number of input variables per history step
            (station variables, NWP and cyclic time features).
        future_dim: number of input variables per forecast step
            (NWP and cyclic time features).
        time_dim: number of cyclic time features; they are the last
            ``time_dim`` columns of both history and future inputs.
        target_index: column of the power target in the history input.
        history_len, horizon: input and output length ``L`` and ``H``.
        memory_days: number of same-time days ``D`` in the daily memory.
        hidden_dim: feature width ``d``.
        expert_hidden_dim: hidden width ``d_e`` of each expert.
        residual_limit: ``s_r / s_y``, the bound of the residual correction.
        prior_weights: initial coefficients ``w^(0)`` of the four prior candidates
            (Eq. 10). With ``prior_mode="learned"`` (the paper's model) the
            coefficients are ``softmax(theta)`` with ``theta`` initialized to
            ``log w^(0)`` and trained; ``"fixed"`` keeps them at ``w^(0)``.
        graph_block: time-block length ``P`` of the graph nodes.
        graph_heads, graph_topk: attention heads ``M_g`` and neighbors ``k_g``.
        n_specific_experts, specific_topk, n_shared_experts: RMoE layout.
        memory_size: capacity of the state memory.

    The remaining switches select the ablation variants of the paper; their
    defaults give the full model.
    """

    def __init__(
        self,
        *,
        history_dim: int,
        future_dim: int,
        time_dim: int,
        target_index: int,
        history_len: int,
        horizon: int,
        memory_days: int,
        hidden_dim: int = 96,
        expert_hidden_dim: int = 192,
        residual_limit: float = 1.0,
        prior_weights: tuple[float, ...] = (0.60, 0.25, 0.10, 0.05),
        graph_block: int = 16,
        graph_heads: int = 4,
        graph_topk: int = 4,
        n_specific_experts: int = 6,
        specific_topk: int = 2,
        n_shared_experts: int = 2,
        memory_size: int = 128,
        dropout: float = 0.25,
        prior_mode: str = "learned",
        graph_mode: str = "dynamic",
        residual_mode: str = "moe",
        use_specific: bool = True,
        use_shared: bool = True,
        sparse_routing: bool = True,
        use_future_nwp: bool = True,
        use_state_memory: bool = True,
        use_daily_feature: bool = True,
        use_power_summary: bool = True,
    ):
        super().__init__()
        if prior_mode not in PRIOR_MODES:
            raise ValueError(f"prior_mode must be one of {PRIOR_MODES}.")
        if graph_mode not in GRAPH_MODES:
            raise ValueError(f"graph_mode must be one of {GRAPH_MODES}.")
        if residual_mode not in RESIDUAL_MODES:
            raise ValueError(f"residual_mode must be one of {RESIDUAL_MODES}.")
        if residual_mode == "moe" and not (use_specific or use_shared):
            raise ValueError("The MoE residual needs the Specific or the Shared branch.")
        if not 1 <= specific_topk <= n_specific_experts:
            raise ValueError("specific_topk must be between 1 and n_specific_experts.")
        if len(prior_weights) != 4 or min(prior_weights) < 0 or sum(prior_weights) <= 0:
            raise ValueError("prior_weights must contain four non-negative values with a positive sum.")

        self.horizon = int(horizon)
        self.history_len = int(history_len)
        self.time_dim = int(time_dim)
        self.target_index = int(target_index)
        self.graph_block = int(graph_block)
        self.prior_mode = prior_mode
        self.graph_mode = graph_mode
        self.residual_mode = residual_mode
        self.use_specific = residual_mode == "moe" and bool(use_specific)
        self.use_shared = residual_mode == "moe" and bool(use_shared)
        self.sparse_routing = bool(sparse_routing)
        self.use_future_nwp = bool(use_future_nwp)
        self.use_state_memory = bool(use_state_memory)
        self.use_daily_feature = bool(use_daily_feature)
        self.use_power_summary = bool(use_power_summary)
        self.n_specific_experts = int(n_specific_experts)
        self.specific_topk = int(specific_topk)
        # Without a prior the residual is the whole forecast and must span [0, 1.2].
        self.residual_limit = 1.2 if prior_mode == "none" else float(residual_limit)

        # DPE: daily-memory feature e^d_l (Eq. 4).
        self.daily_memory_proj = nn.Sequential(
            nn.Linear(memory_days, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim)
        )
        # TDG: conditional encoders E^f and E^h (Eq. 3).
        self.future_encoder = FutureEncoder(future_dim, hidden_dim, dropout)
        self.history_encoder = TCN(history_dim, hidden_dim, (1, 2, 4, 8), dropout)
        # RMoE: routing state h_t (Eq. 5) and step embeddings v_l (Eq. 17).
        self.routing_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.step_embedding = nn.Parameter(torch.empty(self.horizon, hidden_dim))
        nn.init.normal_(self.step_embedding, mean=0.0, std=0.02)
        # TDG: time-block nodes and graph (Eqs. 11-14).
        self.history_nodes = math.ceil(self.history_len / self.graph_block)
        self.future_nodes = math.ceil(self.horizon / self.graph_block)
        if graph_mode != "none":
            self.graph = TimeBlockGraph(
                hidden_dim, self.history_nodes + self.future_nodes, graph_heads, graph_topk, dropout, graph_mode
            )
        # DPE: power summary z^p (Eq. 4).
        self.power_encoder = TCN(1 + time_dim, hidden_dim, (1, 2, 4, 8, 16), dropout)
        self.power_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        # Ablation: sample-conditioned prior weights instead of fixed ones.
        if prior_mode == "adaptive":
            self.prior_router = nn.Sequential(
                nn.Linear(hidden_dim + 5, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 4)
            )
        # RMoE experts (Eqs. 16-18). The single-head ablation keeps one unrouted expert.
        context_dim = 4 * hidden_dim
        if residual_mode == "single":
            self.specific_experts = nn.ModuleList([Expert(context_dim, expert_hidden_dim, dropout)])
        if self.use_specific:
            self.specific_experts = nn.ModuleList(
                [Expert(context_dim, expert_hidden_dim, dropout) for _ in range(n_specific_experts)]
            )
            self.specific_router = nn.Linear(hidden_dim, n_specific_experts, bias=False)
        if self.use_shared:
            self.shared_experts = nn.ModuleList(
                [Expert(context_dim, expert_hidden_dim, dropout) for _ in range(n_shared_experts)]
            )
            self.shared_router = nn.Linear(hidden_dim, n_shared_experts, bias=False)
        # RMoE: state memory and its fusion MLP_a (Eqs. 6-7).
        if self.use_state_memory:
            self.memory_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
            )
        self.register_buffer("state_memory", torch.zeros(int(memory_size), hidden_dim))
        self.register_buffer("state_memory_count", torch.zeros((), dtype=torch.long))
        weights = torch.tensor(prior_weights, dtype=torch.float32)
        self.register_buffer("prior_weights", weights / weights.sum())
        # Learnable prior coefficients (Eq. 10), created last so that the random
        # initialization of all other modules does not depend on the prior mode.
        if prior_mode == "learned":
            self.prior_logits = nn.Parameter(torch.log(self.prior_weights).clone())
        else:
            self.register_parameter("prior_logits", None)

    # ------------------------------------------------------------- helpers
    @torch.no_grad()
    def prior_coefficients(self) -> torch.Tensor:
        """Current coefficients of the four prior candidates (sample-independent modes)."""
        if self.prior_mode == "learned":
            return torch.softmax(self.prior_logits, dim=0).detach().cpu()
        if self.prior_mode == "none":
            return torch.zeros(4)
        if self.prior_mode == "fixed":
            return self.prior_weights.detach().cpu()
        return torch.full((4,), float("nan"))  # adaptive: the coefficients depend on the sample

    def _block_pool(self, sequence, num_nodes):
        """Mean of the valid positions in each block of ``graph_block`` steps (Eq. 11)."""
        batch, length, hidden = sequence.shape
        padding = num_nodes * self.graph_block - length
        valid = sequence.new_ones(batch, length, 1)
        if padding:
            sequence = torch.cat([sequence, sequence.new_zeros(batch, padding, hidden)], dim=1)
            valid = torch.cat([valid, valid.new_zeros(batch, padding, 1)], dim=1)
        sequence = sequence.view(batch, num_nodes, self.graph_block, hidden)
        valid = valid.view(batch, num_nodes, self.graph_block, 1)
        return (sequence * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)

    def _fuse_state_memory(self, state):
        """Blend the routing state with its nearest stored states (Eqs. 6-7)."""
        count = int(self.state_memory_count.item())
        if not self.use_state_memory or count == 0:
            return state
        stored = self.state_memory[:count].to(state)
        similarity = torch.matmul(F.normalize(state, dim=-1), F.normalize(stored, dim=-1).transpose(0, 1))
        reference = torch.matmul(torch.softmax(similarity, dim=-1), stored)
        alpha = torch.sigmoid(self.memory_fusion(torch.cat([state, reference], dim=-1)))
        return alpha * state + (1.0 - alpha) * reference

    @torch.no_grad()
    def update_state_memory(self, raw_state, mask):
        """Append the routing states of poorly predicted training samples (Eq. 24)."""
        if not self.use_state_memory:
            return
        candidates = raw_state.detach()[mask.to(dtype=torch.bool)].to(self.state_memory)
        if not candidates.numel():
            return
        existing = self.state_memory[: int(self.state_memory_count.item())]
        combined = torch.cat([existing, candidates], dim=0)[-self.state_memory.size(0) :]
        self.state_memory.zero_()
        self.state_memory[: combined.size(0)].copy_(combined)
        self.state_memory_count.fill_(combined.size(0))

    def _specific_branch(self, context, routing_state):
        """Step-wise Top-k routed Specific experts (Eq. 17) and the balance loss (Eq. 22)."""
        logits = self.specific_router(routing_state)  # [B, H, E_s]
        probabilities = torch.softmax(logits, dim=-1)
        top_logits, top_indices = torch.topk(logits, k=self.specific_topk, dim=-1)
        top_weights = torch.softmax(top_logits, dim=-1)
        # Numerically a no-op; kept so converted checkpoints reproduce the paper bit-for-bit.
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        if self.sparse_routing:
            gate = torch.zeros_like(probabilities).scatter(-1, top_indices, top_weights)
        else:  # dense-routing ablation
            gate = probabilities
        outputs = torch.stack([expert(context) for expert in self.specific_experts], dim=-1)
        selected = F.one_hot(top_indices, num_classes=self.n_specific_experts).sum(dim=-2).to(probabilities.dtype)
        balance_loss = self.n_specific_experts * torch.sum(selected.mean(dim=(0, 1)) * probabilities.mean(dim=(0, 1)))
        routing = {"specific_probabilities": probabilities, "specific_topk_indices": top_indices, "specific_gate": gate}
        return torch.sum(outputs * gate, dim=-1), balance_loss, routing

    # ------------------------------------------------------------- forward
    def forward(self, x_hist, x_fut, daily_memory, priors):
        """Forecast the normalized power of the whole window.

        Args:
            x_hist: history inputs ``[B, L, history_dim]``.
            x_fut: future inputs ``[B, H, future_dim]``.
            daily_memory: same-time power of the past ``D`` days ``[B, D, H]``.
            priors: the four prior candidates ``[B, 4, H]`` (Eq. 9).

        Returns:
            ``(prediction [B, H], aux)``; ``aux`` holds the prior, the balance
            loss, the routing states and the routing weights.
        """
        batch = x_hist.size(0)
        daily_feature = self.daily_memory_proj(daily_memory.transpose(1, 2))  # e^d, [B, H, d]
        if not self.use_future_nwp:  # ablation: keep only the calendar features
            x_fut = x_fut.clone()
            x_fut[:, :, : -self.time_dim] = 0.0
        future_feature = self.future_encoder(x_fut)  # E^f, [B, H, d]

        history_feature = self.history_encoder(x_hist)  # E^h, [B, L, d]
        _, raw_state = self.routing_gru(history_feature)
        raw_state = raw_state.squeeze(0)  # h_t
        state = self._fuse_state_memory(raw_state)  # h~_t

        nodes = torch.cat(
            [self._block_pool(history_feature, self.history_nodes), self._block_pool(future_feature, self.future_nodes)],
            dim=1,
        )
        if self.graph_mode == "none":
            graph_nodes, adjacency = nodes, None
        else:
            graph_nodes, adjacency = self.graph(nodes)
        graph_feature = F.interpolate(  # g_l, Eq. (15)
            graph_nodes[:, self.history_nodes :, :].transpose(1, 2), size=self.horizon, mode="linear", align_corners=False
        ).transpose(1, 2)

        power_input = torch.cat(
            [x_hist[:, :, self.target_index : self.target_index + 1], x_hist[:, :, -self.time_dim :]], dim=-1
        )
        power_summary = self.power_proj(self.power_encoder(power_input)[:, -1, :])  # z^p
        # Ablations: zero a DPE feature in the expert context, keeping its width.
        if not self.use_daily_feature:
            daily_feature = torch.zeros_like(daily_feature)
        if not self.use_power_summary:
            power_summary = torch.zeros_like(power_summary)
        context = torch.cat(  # c_l, Eq. (16)
            [daily_feature, future_feature, graph_feature, power_summary.unsqueeze(1).expand(-1, self.horizon, -1)],
            dim=-1,
        )

        # Daily prior b_l (Eq. 10).
        prior_values = priors.transpose(1, 2)  # [B, H, 4]
        if self.prior_mode == "learned":
            prior_weight = torch.softmax(self.prior_logits, dim=0).view(1, 1, -1).expand(batch, self.horizon, -1)
        elif self.prior_mode == "fixed":
            prior_weight = self.prior_weights.view(1, 1, -1).expand(batch, self.horizon, -1)
        elif self.prior_mode == "none":
            prior_weight = torch.zeros_like(prior_values)
        else:
            last_power = x_hist[:, -1, self.target_index].view(batch, 1, 1).expand(-1, self.horizon, -1)
            router_input = torch.cat([state.unsqueeze(1).expand(-1, self.horizon, -1), prior_values, last_power], dim=-1)
            prior_weight = torch.softmax(self.prior_router(router_input), dim=-1)
        prior = (prior_values * prior_weight).sum(dim=-1)

        # Residual r_l (Eqs. 17-18).
        balance_loss = prior.new_zeros(())
        routing = {}
        residual = torch.zeros_like(prior)
        if self.residual_mode == "single":
            residual = self.specific_experts[0](context)
        elif self.residual_mode == "moe":
            if self.use_specific:
                specific, balance_loss, routing = self._specific_branch(
                    context, state.unsqueeze(1) + self.step_embedding.unsqueeze(0)
                )
                residual = residual + specific
            if self.use_shared:
                shared_weights = torch.softmax(self.shared_router(state), dim=-1).unsqueeze(1)
                shared_weights = shared_weights.expand(-1, self.horizon, -1)
                shared_outputs = torch.stack([expert(context) for expert in self.shared_experts], dim=-1)
                residual = residual + torch.sum(shared_outputs * shared_weights, dim=-1)
                routing["shared_weights"] = shared_weights

        prediction = torch.clamp(prior + self.residual_limit * torch.tanh(residual), 0.0, 1.2)  # Eq. (19)
        aux = {
            "prior": prior,
            "prior_weight": prior_weight,
            "moe_balance_loss": balance_loss,
            "routing_state": state,
            "raw_routing_state": raw_state,
            "adjacency": adjacency,
            **routing,
        }
        return prediction, aux
