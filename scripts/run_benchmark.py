#!/usr/bin/env python3
"""Train and evaluate models over stations x horizons x seeds and summarize them.

Examples:
    # The 20 DPDG-MoE runs reported in the paper (seed, batch size and settings per task)
    python scripts/run_benchmark.py --paper-tasks

    # Ablation variant on the paper's reference seeds (4 h, 12 h and 1 d)
    python scripts/run_benchmark.py --paper-tasks --variant no_prior --horizons 16 48 96

    # Hyperparameter sensitivity: change one setting of the reference runs
    python scripts/run_benchmark.py --paper-tasks --horizons 16 48 96 --set graph_topk=8

    # Comparators (three seeds, as in the paper)
    python scripts/run_benchmark.py --models Cross-Unet PatchTST TimeMixer iTransformer TimesNet

    # One quick task
    python scripts/run_benchmark.py --stations S-2 --horizons 96 --models DPDG-MoE --seeds 2026

Each run is written to ``<output-dir>/<station>/H<horizon>/<label>/seed_<seed>/``, where the label is the
model name plus the variant and any ``--set`` values. ``summary.csv`` lists the test mean and the sample
standard deviation over seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dpdgmoe.data import HORIZONS, STATIONS, PVForecastData  # noqa: E402
from dpdgmoe.trainer import ALL_MODELS, DEFAULT_CONFIG, MODEL_NAME, train_and_evaluate, write_json  # noqa: E402

PAPER_HORIZONS = (16, 48, 96, 384, 672)
COMPARATOR_SEEDS = (2026, 2027, 2028)
PRIOR_ONLY_BATCH = {16: 128, 48: 192, 96: 192}  # the larger batch of the full model does not fit in 24 GB
SUMMARY_METRICS = ("mae", "rmse", "day_mae", "r2")


def parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stations", nargs="+", default=list(STATIONS), choices=STATIONS)
    parser.add_argument("--horizons", nargs="+", type=int, default=list(PAPER_HORIZONS), choices=list(HORIZONS))
    parser.add_argument("--models", nargs="+", default=[MODEL_NAME], choices=ALL_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(COMPARATOR_SEEDS))
    parser.add_argument("--paper-tasks", action="store_true",
                        help="DPDG-MoE only: use the seed, batch size and task settings of the paper (configs: paper_tasks)")
    parser.add_argument("--variant", default="full_model", help="DPDG-MoE ablation variant (see configs/dpdgmoe.json)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="DPDG-MoE setting to change, e.g. graph_topk=8 or n_specific_experts=4 (repeatable)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=None, help="default: runs/<timestamp>")
    parser.add_argument("--resume", action="store_true", help="skip runs that already have metrics.json")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error("epochs, patience and batch-size must be positive")
    if (args.variant != "full_model" or args.overrides or args.paper_tasks) and args.models != [MODEL_NAME]:
        parser.error("--variant, --set and --paper-tasks apply to DPDG-MoE only")
    try:
        args.overrides = dict(item.split("=", 1) for item in args.overrides)
    except ValueError:
        parser.error("--set expects KEY=VALUE")
    args.overrides = {key: parse_value(value) for key, value in args.overrides.items()}
    return args


def run_plan(args, config: dict) -> list[dict]:
    """List the (station, horizon, model, seed, batch size, overrides) runs to execute."""
    plan = []
    if args.paper_tasks:
        studies = args.variant != "full_model" or bool(args.overrides)
        for task in config["paper_tasks"]["tasks"]:
            if task["station"] not in args.stations or task["horizon"] not in args.horizons:
                continue
            if studies and "ablation_seed" not in task:
                print(f"skip {task['station']} H={task['horizon']}: ablation and sensitivity runs cover 4 h-1 d only")
                continue
            batch = task["batch_size"]
            if args.variant == "prior_only":
                batch = PRIOR_ONLY_BATCH[task["horizon"]]
            plan.append({
                "station": task["station"], "horizon": task["horizon"], "model": MODEL_NAME,
                "seed": task["ablation_seed"] if studies else task["seed"], "batch_size": batch,
                # Ablation and sensitivity runs start from the reference model without task-specific settings.
                "overrides": dict(args.overrides) if studies else dict(task["overrides"]),
            })
        return plan
    for station in args.stations:
        for horizon in args.horizons:
            for model in args.models:
                for seed in args.seeds:
                    plan.append({"station": station, "horizon": horizon, "model": model, "seed": seed,
                                 "batch_size": args.batch_size,
                                 "overrides": dict(args.overrides) if model == MODEL_NAME else {}})
    return plan


def run_label(model: str, variant: str, cli_overrides: dict) -> str:
    label = model if variant == "full_model" else f"{model}_{variant}"
    for key, value in sorted(cli_overrides.items()):
        label += f"_{key}={value}"
    return label


def summarize(records: list[dict], output_dir: Path) -> None:
    groups: dict[tuple, list[dict]] = {}
    for record in records:
        groups.setdefault((record["station"], record["horizon"], record["model"]), []).append(record)
    rows = []
    for (station, horizon, model), items in sorted(groups.items()):
        row = {
            "station": station,
            "horizon": horizon,
            "task": HORIZONS[horizon],
            "model": model,
            "n_seeds": len(items),
            "seeds": " ".join(str(item["seed"]) for item in sorted(items, key=lambda item: item["seed"])),
        }
        for metric in SUMMARY_METRICS:
            values = [item["test"][metric] for item in items]
            row[f"test_{metric}_mean"] = statistics.fmean(values)
            row[f"test_{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None  # sample SD
        rows.append(row)
    if rows:
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir or ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    plan = run_plan(args, config)
    write_json(output_dir / "manifest.json",
               {"arguments": vars(args), "plan": plan, "started": datetime.now().isoformat()})

    records, data_cache = [], {}
    for item in plan:
        station, horizon, model, seed = item["station"], item["horizon"], item["model"], item["seed"]
        key = (station, horizon)
        if key not in data_cache:  # identical windows and scalers for all models and seeds of a task
            data_cache.clear()
            data_cache[key] = PVForecastData(
                Path(args.data_dir) / f"{station}.csv", horizon, prior_weights=tuple(config["prior_weights"])
            )
        label = run_label(model, args.variant if model == MODEL_NAME else "full_model",
                          args.overrides if model == MODEL_NAME else {})
        run_dir = output_dir / station / f"H{horizon}" / label / f"seed_{seed}"
        metrics_path = run_dir / "metrics.json"
        if args.resume and metrics_path.is_file():
            result = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            print(f"== {station} H={horizon} ({HORIZONS[horizon]}) {label} seed={seed} batch={item['batch_size']}",
                  flush=True)
            result = train_and_evaluate(
                model,
                data_cache[key],
                run_dir,
                seed=seed,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=item["batch_size"],
                num_workers=args.num_workers,
                device=args.device,
                variant=args.variant if model == MODEL_NAME else "full_model",
                config_path=args.config,
                overrides=item["overrides"] or None,
                log=lambda message: print(f"   {message}", flush=True),
            )
        records.append(
            {
                "station": station,
                "horizon": horizon,
                "model": label,
                "seed": seed,
                "validation": result["metrics"]["validation"]["overall"],
                "test": result["metrics"]["test"]["overall"],
                "run_dir": str(run_dir.relative_to(output_dir)),
            }
        )
        write_json(output_dir / "records.json", records)
        summarize(records, output_dir)
    print(f"Finished {len(records)} runs. Summary: {output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
