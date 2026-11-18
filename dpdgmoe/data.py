"""Data loading, chronological splitting and window construction.

One :class:`PVForecastData` object is built per station and forecast horizon.
It fixes the windows, splits and scaling statistics, so every model and seed
trained on that task sees exactly the same samples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SLOTS_PER_DAY = 96  # 15-min resolution
HISTORY_STEPS = 15 * SLOTS_PER_DAY  # every task uses a 15-day history
HORIZONS = {16: "4h", 48: "12h", 96: "1d", 384: "4d", 672: "7d", 1440: "15d"}
STATIONS = ("S-1", "S-2", "S-3", "S-4")

TIME_COLUMN = "Time"
TARGET_COLUMN = "power"
# Onsite measurements; the target must be the last column.
HISTORY_COLUMNS = (
    "lmd_totalirrad",
    "lmd_diffuseirrad",
    "lmd_temperature",
    "lmd_pressure",
    "lmd_winddirection",
    "lmd_windspeed",
    "power",
)
# NWP covariates aligned to valid time, available for history and forecast window.
WEATHER_COLUMNS = (
    "nwp_globalirrad",
    "nwp_directirrad",
    "nwp_temperature",
    "nwp_humidity",
    "nwp_windspeed",
    "nwp_winddirection",
)
DEFAULT_PRIOR_WEIGHTS = (0.60, 0.25, 0.10, 0.05)
OUTPUT_CLIP = 1.2  # upper bound of the capacity-normalized target


def calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
    """Return the five ``timeF`` calendar features in ``[-0.5, 0.5]`` (``[T, 5]``).

    Minute of hour, hour of day, day of week, day of month and day of year, as
    used by the general time-series baselines for 15-min data.
    """
    features = [
        index.minute / 59.0 - 0.5,
        index.hour / 23.0 - 0.5,
        index.day_of_week / 6.0 - 0.5,
        (index.day - 1) / 30.0 - 0.5,
        (index.day_of_year - 1) / 365.0 - 0.5,
    ]
    return np.vstack(features).transpose(1, 0).astype(np.float32)


def cyclic_features(index: pd.DatetimeIndex) -> np.ndarray:
    """Return sin/cos of time of day and of year plus a 06-18 h indicator (``[T, 5]``)."""
    hours = index.hour + index.minute / 60.0
    hour_angle = 2.0 * np.pi * hours / 24.0
    year_angle = 2.0 * np.pi * (index.day_of_year.to_numpy() - 1) / 366.0
    daylight = np.where((hours >= 6.0) & (hours <= 18.0), 1.0, 0.0)
    return np.stack(
        [np.sin(hour_angle), np.cos(hour_angle), np.sin(year_angle), np.cos(year_angle), daylight],
        axis=1,
    ).astype(np.float32)


def chronological_split(
    targets: list[int], total_steps: int, horizon: int, train_ratio: float, val_ratio: float
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Assign forecast origins to chronological train/validation/test segments.

    An origin belongs to a segment only if its whole forecast window
    ``[origin, origin + horizon)`` lies inside the segment. Windows that cross a
    boundary are discarded. The history window may reach into earlier segments.
    """
    if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0 and train_ratio + val_ratio < 1.0):
        raise ValueError("Ratios must be positive and train_ratio + val_ratio must be below 1.")
    train_end = int(total_steps * train_ratio)
    validation_end = int(total_steps * (train_ratio + val_ratio))
    splits: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for origin in targets:
        end = origin + horizon
        if end <= train_end:
            splits["train"].append(origin)
        elif train_end <= origin and end <= validation_end:
            splits["validation"].append(origin)
        elif validation_end <= origin and end <= total_steps:
            splits["test"].append(origin)
    if any(not values for values in splits.values()):
        raise RuntimeError(f"Split produced an empty segment: { {k: len(v) for k, v in splits.items()} }")
    return splits, {"train_end_step": train_end, "validation_end_step": validation_end}


def _safe_std(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    result[result == 0] = 1.0
    return result


class PVForecastData:
    """Windows, splits and scalers for one station and one forecast horizon.

    Args:
        csv_path: station CSV file (see ``data/README.md`` for the schema).
        horizon: number of 15-min forecast steps (one of :data:`HORIZONS`).
        train_ratio, val_ratio: chronological split ratios; the rest is test.
        prior_weights: weights of the four daily-prior candidates. They define
            the prior used in the loss weighting (``prior_base``).
    """

    def __init__(
        self,
        csv_path: str | Path,
        horizon: int,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        prior_weights: tuple[float, float, float, float] = DEFAULT_PRIOR_WEIGHTS,
    ):
        self.csv_path = Path(csv_path).resolve()
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Dataset does not exist: {self.csv_path}")
        if int(horizon) not in HORIZONS:
            raise ValueError(f"Unsupported horizon {horizon}; choose from {list(HORIZONS)}")
        if len(prior_weights) != 4:
            raise ValueError("prior_weights must contain four values.")
        self.horizon = int(horizon)
        self.history_steps = HISTORY_STEPS
        self.memory_days = HISTORY_STEPS // SLOTS_PER_DAY
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.prior_weights = tuple(float(weight) for weight in prior_weights)
        self.target_index = HISTORY_COLUMNS.index(TARGET_COLUMN)
        self._load()
        self._build_splits()
        self._fit_scalers()

    # ------------------------------------------------------------------ setup
    def _load(self) -> None:
        required = [TIME_COLUMN, *HISTORY_COLUMNS, *WEATHER_COLUMNS]
        source = pd.read_csv(self.csv_path)
        missing = [column for column in required if column not in source.columns]
        if missing:
            raise RuntimeError(f"{self.csv_path.name} is missing columns: {missing}")
        timestamps = pd.to_datetime(source[TIME_COLUMN], errors="coerce")
        if timestamps.isna().any() or timestamps.duplicated().any():
            raise RuntimeError(f"Column {TIME_COLUMN!r} must contain unique, valid timestamps.")

        source = source.loc[:, required].copy()
        source[TIME_COLUMN] = timestamps
        source = source.set_index(TIME_COLUMN).sort_index().resample("15min").mean()
        full_index = pd.date_range(source.index.min(), source.index.max(), freq="15min")
        source = source.reindex(full_index)

        self.timestamps = pd.DatetimeIndex(full_index)
        self.history_raw = source.loc[:, HISTORY_COLUMNS].to_numpy(dtype=np.float32)
        self.weather_raw = source.loc[:, WEATHER_COLUMNS].to_numpy(dtype=np.float32)
        self.calendar = calendar_features(self.timestamps)
        self._cyclic_raw = cyclic_features(self.timestamps)

    def _build_splits(self) -> None:
        # A window is usable only if history and forecast window are complete.
        valid_rows = np.isfinite(np.concatenate([self.history_raw, self.weather_raw], axis=1)).all(axis=1)
        invalid_prefix = np.concatenate([[0], np.cumsum(~valid_rows)])
        origins = [
            origin
            for origin in range(self.history_steps, len(self.timestamps) - self.horizon + 1)
            if invalid_prefix[origin + self.horizon] == invalid_prefix[origin - self.history_steps]
        ]
        splits, boundaries = chronological_split(
            origins, len(self.timestamps), self.horizon, self.train_ratio, self.val_ratio
        )
        self.split_origins = {name: np.asarray(values, dtype=np.int64) for name, values in splits.items()}
        self.split_boundaries = boundaries
        self.train_end = int(self.split_origins["train"][-1] + self.horizon)

    def _fit_scalers(self) -> None:
        """Fit every statistic on the training segment only."""
        history_train = self.history_raw[: self.train_end]
        weather_train = self.weather_raw[: self.train_end]
        if not (np.isfinite(history_train).all() and np.isfinite(weather_train).all()):
            raise RuntimeError("Training segment contains missing values.")
        self.history_mean = history_train.mean(axis=0, keepdims=True)
        self.history_std = _safe_std(history_train.std(axis=0, keepdims=True))
        self.weather_mean = weather_train.mean(axis=0, keepdims=True)
        self.weather_std = _safe_std(weather_train.std(axis=0, keepdims=True))
        self.history = (self.history_raw - self.history_mean) / self.history_std
        self.weather = (self.weather_raw - self.weather_mean) / self.weather_std

        cyclic_train = self._cyclic_raw[: self.train_end]
        self.cyclic = (self._cyclic_raw - cyclic_train.mean(axis=0, keepdims=True)) / _safe_std(
            cyclic_train.std(axis=0, keepdims=True)
        )

        target_train = history_train[:, self.target_index]
        # s_y: normalization scale of the power target (99.9th percentile).
        self.capacity = max(float(np.quantile(target_train, 0.999)), 1.0)
        # s_r: residual bound in power units (99th percentile), see Eq. (19).
        self.residual_limit = max(float(np.quantile(target_train, 0.99)), 1e-6)

    # ---------------------------------------------------------------- access
    @property
    def history_dim(self) -> int:
        return len(HISTORY_COLUMNS)

    @property
    def weather_dim(self) -> int:
        return len(WEATHER_COLUMNS)

    def standardized_to_power(self, values: np.ndarray) -> np.ndarray:
        """Map standardized power (baseline outputs) back to MW."""
        return values * self.history_std[0, self.target_index] + self.history_mean[0, self.target_index]

    def origin_times(self, origins: np.ndarray) -> np.ndarray:
        return self.timestamps[origins].asi8

    def window(self, origin: int) -> dict[str, np.ndarray]:
        """Return all model inputs and targets for the forecast origin ``origin``.

        ``origin`` is the index of the first forecast step. Shapes use
        ``L = history_steps``, ``H = horizon`` and ``D = memory_days``.
        """
        start, end = origin - self.history_steps, origin + self.horizon
        offsets = np.arange(self.horizon, dtype=np.int64)
        # Same-time power on each of the D days before the forecast day of a step.
        # Steps on later forecast days look further back, so no future value leaks in.
        completed_days = offsets // SLOTS_PER_DAY
        memory = [
            self.history_raw[origin + offsets - (completed_days + day) * SLOTS_PER_DAY, self.target_index]
            for day in range(1, self.memory_days + 1)
        ]
        memory = np.clip(np.stack(memory) / self.capacity, 0.0, OUTPUT_CLIP).astype(np.float32)
        priors = np.stack(
            [memory[0], memory[:3].mean(axis=0), memory[:7].mean(axis=0), np.median(memory, axis=0)]
        ).astype(np.float32)
        w1, w2, w3, w4 = self.prior_weights
        prior_base = (w1 * priors[0] + w2 * priors[1] + w3 * priors[2] + w4 * priors[3]).astype(np.float32)
        target = np.clip(
            self.history_raw[origin:end, self.target_index] / self.capacity, 0.0, OUTPUT_CLIP
        ).astype(np.float32)
        return {
            "history": self.history[start:origin],  # [L, 7] standardized, power last
            "history_weather": self.weather[start:origin],  # [L, 6] standardized NWP
            "history_calendar": self.calendar[start:origin],  # [L, 5] timeF
            "history_cyclic": self.cyclic[start:origin],  # [L, 5]
            "future_weather": self.weather[origin:end],  # [H, 6]
            "future_calendar": self.calendar[origin:end],  # [H, 5]
            "future_cyclic": self.cyclic[origin:end],  # [H, 5]
            "daily_memory": memory,  # [D, H] power / s_y
            "priors": priors,  # [4, H] prior candidates
            "prior_base": prior_base,  # [H] fixed-weight daily prior
            "target": target,  # [H] power / s_y, clipped to [0, 1.2]
            "target_standardized": self.history[origin:end, self.target_index],  # [H]
            "origin": np.int64(origin),
        }

    def summary(self) -> dict:
        """Describe the task, split sizes and target time ranges."""
        return {
            "csv": self.csv_path.name,
            "horizon_steps": self.horizon,
            "task": HORIZONS[self.horizon],
            "history_steps": self.history_steps,
            "split_ratios": [self.train_ratio, self.val_ratio, round(1.0 - self.train_ratio - self.val_ratio, 10)],
            "split_boundaries": self.split_boundaries,
            "samples": {name: int(len(values)) for name, values in self.split_origins.items()},
            "origin_time_bounds": {
                name: [str(self.timestamps[values[0]]), str(self.timestamps[values[-1]])]
                for name, values in self.split_origins.items()
            },
            "capacity": self.capacity,
            "residual_limit": self.residual_limit,
            "prior_weights": list(self.prior_weights),
        }


class WindowDataset(Dataset):
    """PyTorch view of one split; items are dictionaries of tensors."""

    def __init__(self, data: PVForecastData, split: str):
        self.data = data
        self.origins = data.split_origins[split]

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        return {key: torch.as_tensor(value) for key, value in self.data.window(int(self.origins[item])).items()}
