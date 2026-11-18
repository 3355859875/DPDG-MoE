import pytest
import torch
from torch.utils.data import DataLoader

from dpdgmoe.data import PVForecastData, WindowDataset
from dpdgmoe.trainer import ALL_MODELS, MODEL_NAME, build_model, model_settings, predict


@pytest.mark.parametrize("model_name", [name for name in ALL_MODELS if name != MODEL_NAME])
def test_comparison_model_forecasts_the_whole_window(station_csv, model_name):
    data = PVForecastData(station_csv, 16)
    torch.manual_seed(0)
    model = build_model(model_name, data, model_settings(model_name, data))
    loader = DataLoader(WindowDataset(data, "test"), batch_size=4)
    batch = next(iter(loader))
    prediction, target, origins, _ = predict(model_name, model, [batch], data, "cpu")
    assert prediction.shape == target.shape == (4, 16)
    assert torch.isfinite(torch.as_tensor(prediction)).all()
    assert (origins == batch["origin"].numpy()).all()
