"""Evaluation metrics, computed in MW on arrays of shape ``[N, H]``."""

from __future__ import annotations

import numpy as np

DAYLIGHT_MW = 0.5  # DayMAE uses targets above this power


def r2_score(prediction: np.ndarray, target: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    denominator = np.sum((target - target.mean()) ** 2)
    if denominator <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum((target - prediction) ** 2) / denominator)


def forecast_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    """Return overall MAE, RMSE, R2, DayMAE and MAPE plus per-step metrics."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = np.abs(prediction - target)
    nonzero = np.abs(target) > 1e-6
    daylight = target > DAYLIGHT_MW
    overall = {
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "r2": r2_score(prediction, target),
        "day_mae": float(error[daylight].mean()) if daylight.any() else 0.0,
        "mape": float((error[nonzero] / np.abs(target[nonzero])).mean() * 100) if nonzero.any() else float("nan"),
    }
    per_step = []
    for step in range(target.shape[1]):
        step_error = error[:, step]
        step_target = target[:, step]
        step_nonzero = np.abs(step_target) > 1e-6
        per_step.append(
            {
                "step": step + 1,
                "mae": float(step_error.mean()),
                "rmse": float(np.sqrt(np.mean(step_error**2))),
                "r2": r2_score(prediction[:, step], step_target),
                "mape": float((step_error[step_nonzero] / np.abs(step_target[step_nonzero])).mean() * 100)
                if step_nonzero.any()
                else float("nan"),
            }
        )
    return {"overall": overall, "per_step": per_step}
