"""Training, model selection and evaluation for DPDG-MoE and the baselines.

Model selection uses the validation split only. Test predictions are computed
once, from the restored best-validation checkpoint.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .baselines import BASELINES, baseline_config, build_baseline
from .baselines.adapted import ADAPTED_MODELS, ADAPTER_METADATA, adapted_config, build_adapted
from .data import PVForecastData, WindowDataset
from .metrics import forecast_metrics
from .models import DPDGMoE

MODEL_NAME = "DPDG-MoE"
# Order of the main comparison: proposed model, PV-specific, general time-series models.
ALL_MODELS = (MODEL_NAME, *BASELINES, *ADAPTED_MODELS)
CHECKPOINT_FORMAT = "dpdgmoe-checkpoint-v1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "dpdgmoe.json"
# Settings that are passed to the DPDG-MoE constructor; the rest are training settings.
_ARCHITECTURE_KEYS = (
    "hidden_dim",
    "expert_hidden_dim",
    "dropout",
    "graph_block",
    "graph_heads",
    "graph_topk",
    "n_specific_experts",
    "specific_topk",
    "n_shared_experts",
    "memory_size",
    "prior_mode",
    "graph_mode",
    "residual_mode",
    "use_specific",
    "use_shared",
    "sparse_routing",
    "use_future_nwp",
    "use_state_memory",
    "use_daily_feature",
    "use_power_summary",
)
_VARIANT_DEFAULTS = {
    "prior_mode": "learned",
    "graph_mode": "dynamic",
    "residual_mode": "moe",
    "use_specific": True,
    "use_shared": True,
    "sparse_routing": True,
    "use_future_nwp": True,
    "use_state_memory": True,
    "use_daily_feature": True,
    "use_power_summary": True,
}


# ---------------------------------------------------------------- utilities
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> str:
    if requested == "auto" or (requested.startswith("cuda") and not torch.cuda.is_available()):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def write_json(path: Path, payload) -> None:
    """Write JSON atomically; non-finite floats become ``null``."""

    def clean(value):
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, Path):
            return str(value)
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_save(payload, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _to_device(batch: dict, device: str) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


# ------------------------------------------------------------------ DPDG-MoE
def dpdgmoe_settings(horizon: int, variant: str = "full_model", config_path: str | Path = DEFAULT_CONFIG,
                     overrides: dict | None = None) -> dict:
    """Merge defaults, horizon-specific values, task-specific overrides and an ablation variant.

    ``overrides`` holds the per-task hyperparameters of the paper (see
    ``paper_tasks`` in the config) or the values of a sensitivity run; the
    ablation variant is applied last so that it always changes only its component.
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if variant not in config["variants"]:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(config['variants'])}")
    settings = {**_VARIANT_DEFAULTS, **config["defaults"]}
    settings.update(config["horizon_overrides"].get(str(int(horizon)), {}))
    overrides = dict(overrides or {})
    unknown = sorted(set(overrides) - set(settings))
    if unknown:
        raise ValueError(f"Unknown setting(s) in overrides: {unknown}")
    settings.update(overrides)
    settings.update(config["variants"][variant])
    settings["variant"] = variant
    settings["overrides"] = overrides
    settings["prior_weights"] = list(config["prior_weights"])
    return settings


def build_dpdgmoe(data: PVForecastData, settings: dict) -> DPDGMoE:
    time_dim = data.cyclic.shape[1]
    return DPDGMoE(
        history_dim=data.history_dim + data.weather_dim + time_dim,
        future_dim=data.weather_dim + time_dim,
        time_dim=time_dim,
        target_index=data.target_index,
        history_len=data.history_steps,
        horizon=data.horizon,
        memory_days=data.memory_days,
        residual_limit=data.residual_limit / max(data.capacity, 1e-6),
        prior_weights=tuple(settings["prior_weights"]),
        **{key: settings[key] for key in _ARCHITECTURE_KEYS},
    ).float()


def dpdgmoe_inputs(batch: dict) -> tuple[torch.Tensor, ...]:
    """Assemble the DPDG-MoE inputs from a batch dictionary."""
    x_hist = torch.cat([batch["history"], batch["history_weather"], batch["history_cyclic"]], dim=-1)
    x_fut = torch.cat([batch["future_weather"], batch["future_cyclic"]], dim=-1)
    return x_hist.float(), x_fut.float(), batch["daily_memory"].float(), batch["priors"].float()


def dpdgmoe_loss(prediction, batch, aux, settings: dict, capacity: float) -> torch.Tensor:
    """Composite objective of Eq. (23), evaluated in power units."""
    target = batch["target"]
    error = prediction * capacity - target * capacity
    absolute = error.abs()
    delta = settings["loss_delta"]
    smooth_l1 = torch.where(absolute <= delta, 0.5 * error.square() / delta, absolute - 0.5 * delta)  # Eq. (20)
    daylight = (target > settings["daylight_threshold"]).float()
    departure = (target * capacity - batch["prior_base"] * capacity).abs() / max(capacity, 1e-6)
    weight = 1.0 + settings["daylight_weight"] * daylight + settings["ramp_weight"] * departure  # Eq. (21)
    residual_penalty = (prediction * capacity - aux["prior"] * capacity).abs().mean()  # Eq. (22)
    return (
        (smooth_l1 * weight).mean()
        + settings["residual_reg_weight"] * residual_penalty
        + settings["moe_balance_loss_weight"] * aux["moe_balance_loss"]
    )


@torch.no_grad()
def predict_dpdgmoe(model, loader, data: PVForecastData, device: str, settings: dict | None = None):
    """Return predictions and targets in MW, origins, and optionally the mean loss."""
    model.eval()
    predictions, targets, origins = [], [], []
    total_loss, count = 0.0, 0
    for batch in loader:
        batch = _to_device(batch, device)
        output, aux = model(*dpdgmoe_inputs(batch))
        if settings is not None:
            total_loss += dpdgmoe_loss(output, batch, aux, settings, data.capacity).item() * output.size(0)
            count += output.size(0)
        predictions.append((output.cpu().numpy() * data.capacity).astype(np.float32))
        targets.append((batch["target"].cpu().numpy() * data.capacity).astype(np.float32))
        origins.append(batch["origin"].cpu().numpy())
    loss = total_loss / max(count, 1) if settings is not None else None
    return np.concatenate(predictions), np.concatenate(targets), np.concatenate(origins), loss


def _dpdgmoe_optimizer(model, settings: dict) -> AdamW:
    """AdamW; the learnable prior logits form their own group (rate ``prior_lr``, no weight decay)."""
    if model.prior_logits is None:
        return AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    others = [param for param in model.parameters() if param is not model.prior_logits]
    return AdamW(
        [{"params": others}, {"params": [model.prior_logits], "lr": settings["prior_lr"], "weight_decay": 0.0}],
        lr=settings["lr"],
        weight_decay=settings["weight_decay"],
    )


def _train_dpdgmoe(model, loaders, data, settings, run_dir: Path, device: str, epochs: int, patience: int, log):
    optimizer = _dpdgmoe_optimizer(model, settings)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=settings["min_lr"])
    state = _TrainingState(run_dir, model, optimizer, scheduler, device)
    for epoch in range(state.epoch, epochs):
        if state.wait >= patience:
            break
        model.train()
        for batch in loaders["train"]:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output, aux = model(*dpdgmoe_inputs(batch))
            loss = dpdgmoe_loss(output, batch, aux, settings, data.capacity)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=settings["grad_clip"])
            optimizer.step()
            # Store routing states of poorly predicted samples (Eq. 24).
            sample_error = (output.detach() - batch["target"]).abs().mean(dim=1)
            model.update_state_memory(aux["raw_routing_state"], sample_error > settings["memory_threshold"])
        scheduler.step()
        prediction, target, _, validation_loss = predict_dpdgmoe(model, loaders["validation"], data, device, settings)
        error = np.abs(prediction - target)
        record = {
            "epoch": epoch + 1,
            "validation_loss": validation_loss,
            "validation_mae": float(error.mean()),
            "validation_day_mae": float(error[target > 0.5].mean()) if np.any(target > 0.5) else 0.0,
        }
        state.end_epoch(record, improved=validation_loss < state.best)
        log(f"epoch {epoch + 1}/{epochs} val_loss={validation_loss:.6g} best={state.best:.6g} wait={state.wait}")
    return state.restore_best()


# ----------------------------------------------------------------- baselines
def baseline_inputs(batch: dict) -> tuple[torch.Tensor, ...]:
    """Inputs of the vendored baselines (wrapped by ``BaselineAdapter``)."""
    return tuple(
        batch[key].float()
        for key in ("history", "history_weather", "history_calendar", "future_weather", "future_calendar")
    )


def adapted_inputs(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs of the structural adapters: ``[B, L, 13]`` history and ``[B, H, 16]`` future."""
    history = torch.cat([batch["history"], batch["history_weather"]], dim=-1)
    future = torch.cat([batch["future_weather"], batch["future_calendar"], batch["future_cyclic"]], dim=-1)
    return history.float(), future.float()


@torch.no_grad()
def predict_standardized(model, loader, data: PVForecastData, device: str, inputs=baseline_inputs):
    """Predict with a model trained on standardized power; returns MW."""
    model.eval()
    predictions, targets, origins = [], [], []
    for batch in loader:
        batch = _to_device(batch, device)
        output = model(*inputs(batch))
        predictions.append(data.standardized_to_power(output.cpu().numpy()))
        targets.append(data.standardized_to_power(batch["target_standardized"].float().cpu().numpy()))
        origins.append(batch["origin"].cpu().numpy())
    return np.concatenate(predictions), np.concatenate(targets), np.concatenate(origins), None


def _train_mse(model, loaders, data, run_dir: Path, device: str, epochs: int, patience: int, log, *,
               optimizer, inputs, grad_clip: float | None, improve_on_tie: bool):
    """MSE training on standardized power with validation-MSE (MW^2) selection."""
    criterion = nn.MSELoss()
    state = _TrainingState(run_dir, model, optimizer, None, device)
    for epoch in range(state.epoch, epochs):
        if state.wait >= patience:
            break
        model.train()
        for batch in loaders["train"]:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(*inputs(batch)), batch["target_standardized"].float())
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        prediction, target, _, _ = predict_standardized(model, loaders["validation"], data, device, inputs)
        validation_loss = float(np.mean((prediction - target) ** 2))
        improved = validation_loss <= state.best if improve_on_tie else validation_loss < state.best
        state.end_epoch({"epoch": epoch + 1, "validation_loss": validation_loss}, improved=improved)
        log(f"epoch {epoch + 1}/{epochs} val_mse={validation_loss:.6g} best={state.best:.6g} wait={state.wait}")
    return state.restore_best()


# ---------------------------------------------------------- training state
class _TrainingState:
    """Best-checkpoint tracking, early stopping and resumable per-epoch state."""

    def __init__(self, run_dir: Path, model, optimizer, scheduler, device: str):
        self.run_dir = run_dir
        self.model, self.optimizer, self.scheduler, self.device = model, optimizer, scheduler, device
        self.best_path = run_dir / ".best_weights.pth"
        self.state_path = run_dir / "training_state.pth"
        self.epoch, self.wait, self.best, self.history = 0, 0, math.inf, []
        if self.state_path.is_file():
            state = torch.load(self.state_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(state["scheduler"])
            random.setstate(state["rng"]["python"])
            np.random.set_state(state["rng"]["numpy"])
            torch.set_rng_state(state["rng"]["torch"])
            if "cuda" in state["rng"] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["rng"]["cuda"])
            self.epoch, self.wait, self.best, self.history = state["epoch"], state["wait"], state["best"], state["history"]

    def end_epoch(self, record: dict, improved: bool) -> None:
        if improved:
            self.best, self.wait = record["validation_loss"], 0
            _atomic_save(self.model.state_dict(), self.best_path)
        else:
            self.wait += 1
        self.epoch = record["epoch"]
        self.history.append({**record, "best_validation_loss": self.best, "wait": self.wait})
        write_json(self.run_dir / "training_history.json", self.history)
        rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        _atomic_save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "rng": rng,
                "epoch": self.epoch,
                "wait": self.wait,
                "best": self.best,
                "history": self.history,
            },
            self.state_path,
        )

    def restore_best(self):
        if not self.best_path.is_file():
            raise RuntimeError(f"No checkpoint was saved in {self.run_dir}")
        self.model.load_state_dict(torch.load(self.best_path, map_location=self.device, weights_only=True))
        return self.model


# ---------------------------------------------------------------- entry point
def model_settings(model_name: str, data: PVForecastData, variant: str = "full_model",
                   config_path: str | Path = DEFAULT_CONFIG, overrides: dict | None = None) -> dict:
    """Return the settings used for ``model_name`` on the task described by ``data``."""
    if model_name == MODEL_NAME:
        settings = dpdgmoe_settings(data.horizon, variant, config_path, overrides)
        if tuple(settings["prior_weights"]) != data.prior_weights:
            raise ValueError("The data and the model must use the same prior_weights.")
        return settings
    shape = dict(history_len=data.history_steps, horizon=data.horizon,
                 history_dim=data.history_dim, weather_dim=data.weather_dim)
    if model_name in ADAPTED_MODELS:
        future_dim = data.weather_dim + data.calendar.shape[1] + data.cyclic.shape[1]
        return adapted_config(model_name, future_dim=future_dim, **shape)
    return vars(baseline_config(model_name, **shape))


def build_model(model_name: str, data: PVForecastData, settings: dict) -> nn.Module:
    """Build a model from the settings stored in its run record or checkpoint."""
    if model_name == MODEL_NAME:
        return build_dpdgmoe(data, settings)
    if model_name in ADAPTED_MODELS:
        return build_adapted(settings)
    return build_baseline(SimpleNamespace(**settings))


def predict(model_name: str, model, loader, data: PVForecastData, device: str, settings: dict | None = None):
    """Return ``(prediction, target, origins, loss)`` in MW for any supported model."""
    if model_name == MODEL_NAME:
        return predict_dpdgmoe(model, loader, data, device, settings)
    inputs = adapted_inputs if model_name in ADAPTED_MODELS else baseline_inputs
    return predict_standardized(model, loader, data, device, inputs)


def load_checkpoint(path: str | Path, data: PVForecastData, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Load a checkpoint written by :func:`train_and_evaluate`."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} file (convert legacy files first).")
    model = build_model(payload["model"], data, payload["settings"]).to(device)
    model.load_state_dict(payload["state_dict"])
    return model.eval(), payload


def train_and_evaluate(
    model_name: str,
    data: PVForecastData,
    run_dir: str | Path,
    *,
    seed: int,
    epochs: int = 200,
    patience: int = 8,
    batch_size: int = 128,
    num_workers: int = 0,
    device: str = "auto",
    variant: str = "full_model",
    config_path: str | Path = DEFAULT_CONFIG,
    overrides: dict | None = None,
    log=print,
) -> dict:
    """Train one model on one task and save checkpoint, predictions and metrics.

    Files written to ``run_dir``: ``checkpoint.pth`` (best validation
    weights and settings), ``validation.npz`` and ``test.npz`` (predictions and
    targets in MW), ``training_history.json`` and ``metrics.json``.
    """
    if model_name not in ALL_MODELS:
        raise ValueError(f"Unknown model {model_name!r}; choose from {ALL_MODELS}")
    if model_name != MODEL_NAME and (variant != "full_model" or overrides):
        raise ValueError("Ablation variants and overrides apply to DPDG-MoE only.")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device)

    set_seed(seed)
    settings = model_settings(model_name, data, variant, config_path, overrides)
    model = build_model(model_name, data, settings).to(device)

    loaders = {
        split: DataLoader(
            WindowDataset(data, split), batch_size=batch_size, shuffle=split == "train", num_workers=num_workers
        )
        for split in ("train", "validation", "test")
    }
    train_args = (model, loaders, data, run_dir, device, epochs, patience, log)
    if model_name == MODEL_NAME:
        model = _train_dpdgmoe(model, loaders, data, settings, run_dir, device, epochs, patience, log)
    elif model_name in ADAPTED_MODELS:
        optimizer = AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
        model = _train_mse(*train_args, optimizer=optimizer, inputs=adapted_inputs,
                           grad_clip=settings["grad_clip"], improve_on_tie=False)
    else:
        optimizer = Adam(model.parameters(), lr=settings["learning_rate"])
        model = _train_mse(*train_args, optimizer=optimizer, inputs=baseline_inputs,
                           grad_clip=None, improve_on_tie=True)

    metrics = {}
    for split in ("validation", "test"):
        prediction, target, origins, loss = predict(model_name, model, loaders[split], data, device, settings)
        np.savez_compressed(
            run_dir / f"{split}.npz",
            prediction=prediction,
            target=target,
            origin=origins,
            origin_time_ns=data.origin_times(origins),
        )
        metrics[split] = forecast_metrics(prediction, target)
        if loss is not None:
            metrics[split]["loss"] = loss

    _atomic_save(
        {
            "format": CHECKPOINT_FORMAT,
            "model": model_name,
            "horizon": data.horizon,
            "station_csv": data.csv_path.name,
            "settings": settings,
            "state_dict": model.state_dict(),
        },
        run_dir / "checkpoint.pth",
    )
    result = {
        "model": model_name,
        "variant": variant,
        "seed": seed,
        "parameters": count_parameters(model),
        "settings": settings,
        "training": {"max_epochs": epochs, "patience": patience, "batch_size": batch_size},
        "selection": {"split": "validation", "metric": "loss" if model_name == MODEL_NAME else "mse"},
        "data": data.summary(),
        "metrics": metrics,
    }
    if model_name in ADAPTED_MODELS:
        result["adapter"] = ADAPTER_METADATA[model_name]
    if model_name == MODEL_NAME:
        result["prior_coefficients"] = model.prior_coefficients().tolist()
    write_json(run_dir / "metrics.json", result)
    for temporary in (run_dir / "training_state.pth", run_dir / ".best_weights.pth"):
        temporary.unlink(missing_ok=True)
    return result
