"""Structural adapters of six PV and time-series forecasting methods.

These are independent re-implementations written for this benchmark. Each keeps
the distinguishing operations of its source paper and exposes the interface
``model(history [B, L, 13], future [B, H, 16]) -> [B, H]``: the history holds the
standardized onsite and NWP variables, the future holds NWP, calendar and
cyclic time features. They are not reproductions of the authors' repositories;
``ADAPTER_METADATA`` records what was substituted for each method.
"""

from __future__ import annotations

from torch import nn

from .ampdnet import AMPDNet
from .causalformer import CausalFormer
from .g_graphtst import GGraphTST
from .solarcast import SolarCAST
from .tfcnet import TFCNet
from .winflashdilated_former import WinFlashDilatedFormer

ADAPTED_MODELS = ("AMPDNet", "G-GraphTST", "TFCNet", "CausalFormer", "SolarCAST", "WinFlashDilated-former")

ADAPTER_METADATA = {
    "AMPDNet": {
        "source_task": "PV power forecasting",
        "fidelity": "structural forecasting adapter",
        "input_substitution": "shared S-series protocol and direct multi-horizon head",
        "claim_boundary": "not an original-repository reproduction",
    },
    "G-GraphTST": {
        "source_task": "multivariate long-term PV power forecasting",
        "fidelity": "structural forecasting adapter",
        "input_substitution": "shared S-series protocol and direct multi-horizon head",
        "claim_boundary": "not an original-repository reproduction",
    },
    "TFCNet": {
        "source_task": "PV power forecasting",
        "fidelity": "structural forecasting adapter",
        "input_substitution": "shared S-series protocol; local implementation omits the original KAN head",
        "claim_boundary": "not an original-repository reproduction",
    },
    "CausalFormer": {
        "source_task": "temporal causal discovery",
        "fidelity": "structural forecasting adapter",
        "input_substitution": "S-series variables are treated as causal nodes",
        "claim_boundary": "not an original-author PV forecasting reproduction",
    },
    "SolarCAST": {
        "source_task": "target-site GHI forecasting from neighboring-station GHI",
        "fidelity": "structural forecasting adapter",
        "input_substitution": "single-site S-series variables replace unavailable neighboring-station GHI nodes",
        "claim_boundary": "not a faithful SolarCAST reproduction without aligned multi-site GHI",
    },
    "WinFlashDilated-former": {
        "source_task": "multi-horizon PV power forecasting",
        "fidelity": "paper-structure adapter",
        "input_substitution": "PyTorch exact window attention replaces the original FlashAttention-3 kernel",
        "claim_boundary": "architecture comparison, not original repository reproduction",
    },
}

# Training settings shared by the six adapters (MSE on standardized power).
TRAINING = {"lr": 2e-4, "weight_decay": 5e-4, "grad_clip": 1.0}


def adapted_config(
    model_name: str, *, history_len: int, horizon: int, history_dim: int, weather_dim: int, future_dim: int
) -> dict:
    """Return the constructor arguments used for ``model_name`` in the paper."""
    numeric_dim = history_dim + weather_dim
    target_idx = history_dim - 1  # power is the last onsite variable
    common = {"history_len": history_len, "horizon": horizon, "future_dim": future_dim}
    arguments = {
        "AMPDNet": dict(input_dim=numeric_dim, hidden_dim=32, layers=2, target_idx=target_idx),
        "G-GraphTST": dict(
            input_dim=numeric_dim, d_model=64, heads=4, layers=2, graph_depth=2,
            topk=min(4, numeric_dim), target_idx=target_idx,
        ),
        "TFCNet": dict(history_dim=history_dim, weather_dim=weather_dim, d_model=64, heads=4, layers=2),
        "CausalFormer": dict(input_dim=numeric_dim, d_model=48, heads=4, layers=1, target_idx=target_idx),
        "SolarCAST": dict(
            input_dim=numeric_dim, d_model=48, heads=4, layers=1, graph_depth=2,
            topk=min(5, numeric_dim), target_idx=target_idx,
        ),
        "WinFlashDilated-former": dict(input_dim=numeric_dim, d_model=64, heads=4, layers=2),
    }
    if model_name not in arguments:
        raise ValueError(f"Unknown adapted model {model_name!r}; choose from {ADAPTED_MODELS}")
    return {"model": model_name, "arguments": {**arguments[model_name], **common}, **TRAINING}


def build_adapted(config: dict) -> nn.Module:
    classes = {
        "AMPDNet": AMPDNet,
        "G-GraphTST": GGraphTST,
        "TFCNet": TFCNet,
        "CausalFormer": CausalFormer,
        "SolarCAST": SolarCAST,
        "WinFlashDilated-former": WinFlashDilatedFormer,
    }
    arguments = dict(config["arguments"])
    history_len, horizon = arguments.pop("history_len"), arguments.pop("horizon")
    if config["model"] == "TFCNet":
        history_dim, weather_dim = arguments.pop("history_dim"), arguments.pop("weather_dim")
        return TFCNet(history_dim, weather_dim, history_len, horizon, **arguments).float()
    input_dim = arguments.pop("input_dim")
    return classes[config["model"]](input_dim, history_len, horizon, **arguments).float()
