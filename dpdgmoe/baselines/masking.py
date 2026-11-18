"""Attention masks used by the vendored baseline attention layers."""

import torch


class TriangularCausalMask:
    """Boolean ``[B, 1, L, L]`` mask that hides future positions."""

    def __init__(self, B, L, device="cpu"):
        with torch.no_grad():
            self._mask = torch.ones(B, 1, L, L, dtype=torch.bool).triu(diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    """Causal mask restricted to the queries selected by ProbSparse attention."""

    def __init__(self, B, H, L, index, scores, device="cpu"):
        causal = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        causal = causal[None, None, :].expand(B, H, L, scores.shape[-1])
        selected = causal[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :]
        self._mask = selected.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
