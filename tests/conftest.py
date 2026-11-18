import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpdgmoe.data import HISTORY_COLUMNS, WEATHER_COLUMNS  # noqa: E402



@pytest.fixture(scope="session")
def station_csv(tmp_path_factory) -> Path:
    """A 20-day synthetic station with a clear-sky-like daily power curve."""
    rng = np.random.default_rng(0)
    times = pd.date_range("2018-06-01", periods=20 * 96, freq="15min")
    hours = times.hour + times.minute / 60.0
    sun = np.clip(np.sin((hours - 6.0) / 12.0 * np.pi), 0.0, None)
    cloud = 0.7 + 0.3 * rng.random(len(times))
    frame = {"Time": times.strftime("%Y-%m-%d %H:%M:%S")}
    for column in WEATHER_COLUMNS:
        frame[column] = sun * 800 * cloud + rng.normal(0, 5, len(times))
    for column in HISTORY_COLUMNS:
        frame[column] = sun * 900 * cloud + rng.normal(0, 5, len(times))
    frame["power"] = 20.0 * sun * cloud
    path = tmp_path_factory.mktemp("data") / "S-T.csv"
    pd.DataFrame(frame).to_csv(path, index=False)
    return path
