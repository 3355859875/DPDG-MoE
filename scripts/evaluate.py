#!/usr/bin/env python3
"""Evaluate a saved checkpoint on one split of its station.

Example:
    python scripts/evaluate.py runs/<id>/S-2/H96/DPDG-MoE/seed_2026/checkpoint.pth --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dpdgmoe.data import PVForecastData, WindowDataset  # noqa: E402
from dpdgmoe.metrics import forecast_metrics  # noqa: E402
from dpdgmoe.trainer import load_checkpoint, predict, resolve_device  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save", default=None, help="optional .npz path for predictions and targets")
    args = parser.parse_args()

    device = resolve_device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    prior_weights = tuple(payload["settings"].get("prior_weights", (0.60, 0.25, 0.10, 0.05)))
    data = PVForecastData(Path(args.data_dir) / payload["station_csv"], payload["horizon"], prior_weights=prior_weights)
    model, payload = load_checkpoint(args.checkpoint, data, device)
    loader = DataLoader(WindowDataset(data, args.split), batch_size=args.batch_size)
    prediction, target, origins, _ = predict(payload["model"], model, loader, data, device)
    metrics = forecast_metrics(prediction, target)["overall"]
    print(json.dumps({"model": payload["model"], "split": args.split, **metrics}, indent=2))
    if args.save:
        np.savez_compressed(args.save, prediction=prediction, target=target, origin=origins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
