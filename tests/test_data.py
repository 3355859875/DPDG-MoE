import numpy as np
import pytest

from dpdgmoe.data import HISTORY_STEPS, SLOTS_PER_DAY, PVForecastData, WindowDataset, chronological_split


def test_split_keeps_whole_windows_inside_each_segment():
    splits, bounds = chronological_split(list(range(10, 96)), total_steps=100, horizon=5, train_ratio=0.8, val_ratio=0.1)
    assert max(splits["train"]) + 5 <= bounds["train_end_step"]
    assert min(splits["validation"]) >= bounds["train_end_step"]
    assert max(splits["validation"]) + 5 <= bounds["validation_end_step"]
    assert min(splits["test"]) >= bounds["validation_end_step"]


@pytest.mark.parametrize("horizon", [16, 96])
def test_window_shapes_and_targets(station_csv, horizon):
    data = PVForecastData(station_csv, horizon)
    item = data.window(int(data.split_origins["test"][0]))
    assert item["history"].shape == (HISTORY_STEPS, 7)
    assert item["future_weather"].shape == (horizon, 6)
    assert item["daily_memory"].shape == (15, horizon)
    assert item["priors"].shape == (4, horizon)
    origin = int(item["origin"])
    expected = np.clip(data.history_raw[origin : origin + horizon, -1] / data.capacity, 0, 1.2)
    np.testing.assert_allclose(item["target"], expected, rtol=1e-6)
    assert len(WindowDataset(data, "train")) == len(data.split_origins["train"])


def test_prior_candidates_follow_eq9(station_csv):
    data = PVForecastData(station_csv, 16)
    item = data.window(int(data.split_origins["validation"][0]))
    memory, priors = item["daily_memory"], item["priors"]
    np.testing.assert_allclose(priors[0], memory[0])
    np.testing.assert_allclose(priors[1], memory[:3].mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(priors[2], memory[:7].mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(priors[3], np.median(memory, axis=0), rtol=1e-6)
    expected_base = 0.60 * priors[0] + 0.25 * priors[1] + 0.10 * priors[2] + 0.05 * priors[3]
    np.testing.assert_allclose(item["prior_base"], expected_base, rtol=1e-6)


def test_daily_memory_reads_only_past_values(station_csv):
    data = PVForecastData(station_csv, 96)
    origin = int(data.split_origins["test"][0])
    item = data.window(origin)
    for day in range(1, data.memory_days + 1):
        source = origin + np.arange(96) - day * SLOTS_PER_DAY
        assert (source < origin).all()
        expected = np.clip(data.history_raw[source, -1] / data.capacity, 0, 1.2)
        np.testing.assert_allclose(item["daily_memory"][day - 1], expected, rtol=1e-6)
    # Steps on later forecast days look one more day back, so they never read the forecast window.
    steps = np.arange(672)
    for day in range(1, 16):
        assert (steps - (steps // SLOTS_PER_DAY + day) * SLOTS_PER_DAY < 0).all()


def test_scalers_use_training_segment_only(station_csv):
    data = PVForecastData(station_csv, 16)
    train = data.history_raw[: data.train_end]
    np.testing.assert_allclose(data.history_mean[0], train.mean(axis=0), rtol=1e-5)
    assert data.capacity == pytest.approx(max(float(np.quantile(train[:, -1], 0.999)), 1.0))
