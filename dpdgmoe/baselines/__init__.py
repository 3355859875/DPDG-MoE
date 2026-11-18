"""General-purpose and PV-specific baselines used in the main comparison.

The network code under this package is vendored from the original
repositories (see ``NOTICE``) with import paths adapted. Every baseline is
wrapped by :class:`~dpdgmoe.baselines.adapter.BaselineAdapter` so that it
receives the same inputs as DPDG-MoE.
"""

from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

from torch import nn

from .adapter import BaselineAdapter

# Model name -> module that defines ``Model(configs)``.
BASELINES = {
    "Cross-Unet": "cross_unet",
    "PatchTST": "patchtst",
    "TimeMixer": "timemixer",
    "iTransformer": "itransformer",
    "TimesNet": "timesnet",
}


def baseline_config(
    model_name: str,
    *,
    history_len: int,
    horizon: int,
    history_dim: int,
    weather_dim: int,
    mark_dim: int = 5,
    overrides: dict | None = None,
) -> SimpleNamespace:
    """Return the hyperparameters used for ``model_name`` in the paper."""
    if model_name not in BASELINES:
        raise ValueError(f"Unknown baseline {model_name!r}; choose from {sorted(BASELINES)}")
    channels = history_dim + weather_dim
    values = {
        "model": model_name,
        "task_name": "long_term_forecast",
        "seq_len": int(history_len),
        "label_len": int(horizon),
        "pred_len": int(horizon),
        "enc_in": channels,
        "dec_in": channels,
        "c_out": channels,
        "d_model": 256,
        "n_heads": 4,
        "e_layers": 3,
        "d_layers": 4,
        "d_ff": 512,
        "dropout": 0.05,
        "factor": 10,
        "embed": "timeF",
        "freq": "15min",
        "activation": "gelu",
        "output_attention": False,
        "useweather": True,
        # Cross-Unet encoder and correlation-module options.
        "usenonlinearproject": False,
        "usebottle": True,
        "convmerge": False,
        "swichchannel": False,
        "twofilter": True,
        "seg_len": 12 if horizon <= 48 else 24,
        # PatchTST.
        "patch_len": 32,
        "stride": 8,
        "fc_dropout": 0.05,
        "head_dropout": 0.0,
        "padding_patch": "end",
        "revin": 1,
        "affine": 0,
        "subtract_last": 0,
        "decomposition": 0,
        "kernel_size": 25,
        "individual": 0,
        "add_itransformer": False,
        "add_cross_channel_attention": False,
        "addmasknight": False,
        "change_embed": False,
        "change_embed_dim": 0,
        "use_head_embed": False,
        "n_head_embed": 4,
        "mlp_embed_layer": 2,
        "dmodel_embed": 64,
        "patch_len_arryay": [48, 24, 12, 6],
        # iTransformer / TimeMixer / TimesNet.
        "use_norm": True,
        "cycle": 24,
        "model_type": "mlp",
        "use_revin": 1,
        "hidden_size": 128,
        "channel_independence": 0,
        "decomp_method": "moving_avg",
        "down_sampling_layers": 3,
        "down_sampling_window": 2,
        "down_sampling_method": "avg",
        "moving_avg": 25,
        "use_future_temporal_feature": 1,
        "top_k": 5,
        "num_kernels": 6,
        "alpha": 0.1,
        "top_p": 0.5,
        "pos": 1,
        "num_class": 1,
        "learning_rate": 1e-4,
        "history_dim": int(history_dim),
        "weather_dim": int(weather_dim),
        "mark_dim": int(mark_dim),
    }
    if model_name == "Cross-Unet":
        # The bottleneck fusion adds decoder tokens to the deepest encoder
        # level, which needs at least as many output segments as that level.
        input_segments = math.ceil(values["seq_len"] / values["seg_len"])
        bottleneck_segments = math.ceil(input_segments / (2 ** (values["e_layers"] - 1)))
        output_segments = math.ceil(values["pred_len"] / values["seg_len"])
        values["usebottle"] = output_segments >= bottleneck_segments
    if model_name in {"TimeMixer", "iTransformer", "TimesNet"}:
        # The timeF embeddings of these implementations use the minutely code.
        values["freq"] = "t"
    if model_name == "TimesNet":
        # Compact width from the original TimesNet scripts; the default width
        # gives a 200M+ parameter model on a 1,440-step history.
        values["d_model"] = 32
        values["d_ff"] = 32
    values.update(dict(overrides or {}))
    return SimpleNamespace(**values)


def build_baseline(config: SimpleNamespace) -> nn.Module:
    """Instantiate ``config.model`` wrapped in the benchmark adapter."""
    module = importlib.import_module(f"{__name__}.{BASELINES[config.model]}")
    base_model = module.Model(config).float()
    return BaselineAdapter(
        base_model,
        model_name=config.model,
        history_len=config.seq_len,
        horizon=config.pred_len,
        history_dim=config.history_dim,
        weather_dim=config.weather_dim,
        mark_dim=config.mark_dim,
    ).float()
