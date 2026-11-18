"""Common input/output wrapper for the general-purpose baseline forecasters.

The original baselines expect a weather sequence as long as the history
window, whereas the benchmark supplies future NWP for exactly ``horizon``
steps. ``BaselineAdapter`` keeps that contract at the model boundary:

1. future NWP and calendar features are linearly resampled to the history
   length and passed to the original model;
2. a small step-aligned GRU encodes the unresampled future NWP, and a bounded
   residual head corrects the original model's forecast with it.

Attribute names are kept identical to the implementation used for the paper
so that its baseline checkpoints load without key conversion.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TemporalLengthAdapter(nn.Module):
    """Linearly resample a ``[B, T, C]`` sequence to ``target_len`` steps."""

    def __init__(self, target_len: int):
        super().__init__()
        if int(target_len) < 1:
            raise ValueError("target_len must be positive")
        self.target_len = int(target_len)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError(f"Expected [B, T, C], got {tuple(values.shape)}")
        if values.size(1) == self.target_len:
            return values
        resized = F.interpolate(values.transpose(1, 2), size=self.target_len, mode="linear", align_corners=True)
        return resized.transpose(1, 2).contiguous()


class FutureNWPEncoder(nn.Module):
    """Encode step-aligned future NWP and calendar features with a GRU."""

    def __init__(self, weather_dim: int, mark_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(int(weather_dim) + int(mark_dim), hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.GELU())

    def forward(self, future_weather: torch.Tensor, future_mark: torch.Tensor) -> torch.Tensor:
        encoded = self.input(torch.cat([future_weather, future_mark], dim=-1))
        encoded, _ = self.gru(encoded)
        return self.output(encoded)


class BaselineAdapter(nn.Module):
    """Wrap one baseline so that it maps the benchmark inputs to ``[B, H]``."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        model_name: str,
        history_len: int,
        horizon: int,
        history_dim: int,
        weather_dim: int,
        mark_dim: int = 5,
        hidden_dim: int = 64,
        residual_scale: float = 0.2,
    ):
        super().__init__()
        self.base_model = base_model
        self.model_name = model_name
        self.horizon = int(horizon)
        self.history_len = int(history_len)
        self.history_dim = int(history_dim)
        self.weather_dim = int(weather_dim)
        self.mark_dim = int(mark_dim)
        self.weather_to_history = TemporalLengthAdapter(self.history_len)
        self.future_encoder = FutureNWPEncoder(self.weather_dim, self.mark_dim, hidden_dim)
        self.history_encoder = nn.Sequential(
            nn.Linear(self.history_dim + self.weather_dim + self.mark_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.residual_scale = float(residual_scale)

    def _call_base_model(self, history, history_mark, history_weather, weather_resampled, mark_resampled):
        if self.model_name == "PatchTST":
            return self.base_model(history, weather_resampled, mark_resampled)
        if self.model_name == "TimeMixer":
            return self.base_model(history, history_mark, mark_resampled, weather_resampled)
        if self.model_name == "Cross-Unet":
            return self.base_model(history, history_mark, weather_resampled, mark_resampled, history_weather, history)
        # iTransformer and TimesNet share the signature (values, marks, weather).
        return self.base_model(history, mark_resampled, weather_resampled)

    def _as_prediction(self, output: torch.Tensor) -> torch.Tensor:
        if output.ndim == 3:
            return output[:, -self.horizon :, -1]
        if output.ndim == 2 and output.size(1) == self.horizon:
            return output
        raise ValueError(f"{self.model_name} returned an output of shape {tuple(output.shape)}")

    def forward(
        self,
        history: torch.Tensor,
        history_weather: torch.Tensor,
        history_mark: torch.Tensor,
        future_weather: torch.Tensor,
        future_mark: torch.Tensor,
    ) -> torch.Tensor:
        """Return the standardized power forecast of shape ``[B, H]``.

        Args:
            history: observed station variables ``[B, L, history_dim]`` (power last).
            history_weather: NWP over the history window ``[B, L, weather_dim]``.
            history_mark: calendar features over the history window ``[B, L, mark_dim]``.
            future_weather: NWP over the forecast window ``[B, H, weather_dim]``.
            future_mark: calendar features over the forecast window ``[B, H, mark_dim]``.
        """
        if history.shape[1:] != (self.history_len, self.history_dim):
            raise ValueError(f"history must be [B, {self.history_len}, {self.history_dim}]")
        if future_weather.shape[1:] != (self.horizon, self.weather_dim):
            raise ValueError(f"future_weather must be [B, {self.horizon}, {self.weather_dim}]")
        weather_resampled = self.weather_to_history(future_weather)
        mark_resampled = self.weather_to_history(future_mark)
        base_output = self._call_base_model(history, history_mark, history_weather, weather_resampled, mark_resampled)
        base_prediction = self._as_prediction(base_output)

        future_context = self.future_encoder(future_weather, future_mark)
        history_input = torch.cat([history, history_weather, history_mark], dim=-1)
        history_summary = self.history_encoder(history_input).mean(dim=1, keepdim=True).expand(-1, self.horizon, -1)
        fusion = torch.cat([future_context, history_summary, base_prediction.unsqueeze(-1)], dim=-1)
        residual = self.residual_head(fusion).squeeze(-1)
        return base_prediction + self.residual_scale * torch.tanh(residual)
