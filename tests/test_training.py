import json

import numpy as np
import pytest
from torch.utils.data import DataLoader

from dpdgmoe.data import PVForecastData, WindowDataset
from dpdgmoe.trainer import load_checkpoint, predict, train_and_evaluate


@pytest.mark.parametrize("model_name", ["DPDG-MoE", "PatchTST", "SolarCAST"])
def test_train_evaluate_and_reload(station_csv, tmp_path, model_name):
    data = PVForecastData(station_csv, 16)
    run_dir = tmp_path / model_name
    result = train_and_evaluate(model_name, data, run_dir, seed=2026, epochs=1, batch_size=256, device="cpu", log=lambda _: None)

    for name in ("checkpoint.pth", "metrics.json", "validation.npz", "test.npz", "training_history.json"):
        assert (run_dir / name).is_file()
    assert not (run_dir / "training_state.pth").exists()
    assert json.loads((run_dir / "metrics.json").read_text())["metrics"]["test"]["overall"]["mae"] >= 0
    assert result["data"]["samples"]["test"] == len(data.split_origins["test"])

    model, _ = load_checkpoint(run_dir / "checkpoint.pth", data)
    loader = DataLoader(WindowDataset(data, "test"), batch_size=256)
    prediction, _, _, _ = predict(model_name, model, loader, data, "cpu")
    np.testing.assert_array_equal(prediction, np.load(run_dir / "test.npz")["prediction"])


def test_learned_prior_has_own_optimizer_group(station_csv, tmp_path):
    from dpdgmoe.trainer import _dpdgmoe_optimizer, build_dpdgmoe, dpdgmoe_settings

    data = PVForecastData(station_csv, 16)
    settings = dpdgmoe_settings(16)
    optimizer = _dpdgmoe_optimizer(build_dpdgmoe(data, settings), settings)
    prior_group = optimizer.param_groups[1]
    assert prior_group["lr"] == settings["prior_lr"] == 1e-3 and prior_group["weight_decay"] == 0.0


def test_overrides_are_applied_and_checked():
    from dpdgmoe.trainer import dpdgmoe_settings

    assert dpdgmoe_settings(96, overrides={"graph_topk": 8})["graph_topk"] == 8
    with pytest.raises(ValueError):
        dpdgmoe_settings(96, overrides={"graph_top_k": 8})


def test_paper_tasks_cover_every_site_and_horizon():
    from dpdgmoe.trainer import DEFAULT_CONFIG, dpdgmoe_settings

    tasks = json.loads(DEFAULT_CONFIG.read_text())["paper_tasks"]["tasks"]
    assert sorted((t["station"], t["horizon"]) for t in tasks) == sorted(
        (s, h) for s in ("S-1", "S-2", "S-3", "S-4") for h in (16, 48, 96, 384, 672)
    )
    for task in tasks:
        dpdgmoe_settings(task["horizon"], overrides=task["overrides"])  # every override is a known setting
        assert ("ablation_seed" in task) == (task["horizon"] <= 96)
