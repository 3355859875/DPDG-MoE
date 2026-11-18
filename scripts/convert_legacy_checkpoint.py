#!/usr/bin/env python3
"""Convert a checkpoint from the pre-release research code into this repository's format.

The research code named the model ``AdaptivePriorDyGMoE`` and wrapped it in an
adapter (``base_model.`` key prefix). This script renames the parameters to
the module names used here and drops modules that never contribute to the
forecast (an unused GRU summary, a disabled direct head and the static-graph
logits of the dynamic graph). Checkpoints of the comparison models keep their
parameter names and only need the new wrapper.

Example:
    python scripts/convert_legacy_checkpoint.py old/checkpoint.pth new/checkpoint.pth \
        --model DPDG-MoE --station S-2 --horizon 96 --paper-task
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dpdgmoe.data import PVForecastData  # noqa: E402
from dpdgmoe.trainer import (  # noqa: E402
    ALL_MODELS,
    CHECKPOINT_FORMAT,
    DEFAULT_CONFIG,
    MODEL_NAME,
    build_model,
    dpdgmoe_settings,
    model_settings,
)

# Legacy module prefix -> new module prefix (None: drop).
DPDGMOE_RENAMES = (
    ("daily_encoder.step_proj.", "daily_memory_proj."),
    ("daily_encoder.day_gru.", None),
    ("daily_encoder.summary_proj.", None),
    ("history_token_encoder.", "history_encoder."),
    ("history_router.", "routing_gru."),
    ("gate_step_embedding", "step_embedding"),
    ("graph.static_bias", "graph.static_logits"),
    ("recent_target_encoder.", "power_encoder."),
    ("recent_pool.", "power_proj."),
    ("direct_head.", None),
    ("anomaly_blend.", "memory_fusion."),
    ("anomaly_repository_count", "state_memory_count"),
    ("anomaly_repository", "state_memory"),
    ("static_prior_logits", "prior_logits"),
)


def convert_dpdgmoe_state_dict(legacy: dict, expected_keys) -> dict:
    """Rename legacy keys and keep only those the new model defines."""
    converted = {}
    for key, value in legacy.items():
        key = key.removeprefix("base_model.")
        for old, new in DPDGMOE_RENAMES:
            if key.startswith(old):
                key = None if new is None else new + key[len(old) :]
                break
        if key is not None and key in expected_keys:
            converted[key] = value
    return converted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("legacy_checkpoint")
    parser.add_argument("output")
    parser.add_argument("--model", required=True, choices=ALL_MODELS)
    parser.add_argument("--station", required=True)
    parser.add_argument("--horizon", required=True, type=int)
    parser.add_argument("--variant", default="full_model")
    parser.add_argument("--paper-task", action="store_true",
                        help="apply the task-specific overrides of this station and horizon from paper_tasks")
    parser.add_argument("--overrides", default=None, help="JSON object of extra DPDG-MoE settings")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    args = parser.parse_args()

    legacy = torch.load(args.legacy_checkpoint, map_location="cpu", weights_only=True)
    legacy = legacy.get("model", legacy)  # training-state files nest the weights
    if args.model == MODEL_NAME:
        overrides = json.loads(args.overrides) if args.overrides else {}
        if args.paper_task:
            config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
            task = next(t for t in config["paper_tasks"]["tasks"]
                        if t["station"] == args.station and t["horizon"] == args.horizon)
            overrides = {**task["overrides"], **overrides}
        settings = dpdgmoe_settings(args.horizon, args.variant, overrides=overrides)
        data = PVForecastData(
            Path(args.data_dir) / f"{args.station}.csv", args.horizon, prior_weights=tuple(settings["prior_weights"])
        )
        model = build_model(MODEL_NAME, data, settings)
        state = model.state_dict()
        converted = convert_dpdgmoe_state_dict(legacy, set(state))
        # The fixed prior weights were a non-persistent buffer in the legacy code.
        converted.setdefault("prior_weights", state["prior_weights"])
    else:  # baseline and adapted-model checkpoints keep their keys
        data = PVForecastData(Path(args.data_dir) / f"{args.station}.csv", args.horizon)
        settings = model_settings(args.model, data)
        model = build_model(args.model, data, settings)
        converted = dict(legacy)
    model.load_state_dict(converted, strict=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "model": args.model,
            "horizon": data.horizon,
            "station_csv": data.csv_path.name,
            "settings": settings,
            "state_dict": model.state_dict(),
        },
        args.output,
    )
    print(f"Wrote {args.output} ({len(converted)} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
