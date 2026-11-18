import pytest
import torch

from dpdgmoe.models import DPDGMoE
from dpdgmoe.trainer import dpdgmoe_settings

L, H, D = 64, 16, 7


def make_model(**overrides):
    kwargs = dict(
        history_dim=18,
        future_dim=11,
        time_dim=5,
        target_index=6,
        history_len=L,
        horizon=H,
        memory_days=D,
        hidden_dim=16,
        expert_hidden_dim=32,
        residual_limit=0.5,
        graph_block=8,
        dropout=0.0,
    )
    kwargs.update(overrides)
    return DPDGMoE(**kwargs).eval()


def inputs(batch=3):
    torch.manual_seed(0)
    return torch.randn(batch, L, 18), torch.randn(batch, H, 11), torch.rand(batch, D, H), torch.rand(batch, 4, H)


def test_output_shape_and_range():
    prediction, aux = make_model()(*inputs())
    assert prediction.shape == (3, H)
    assert prediction.min() >= 0 and prediction.max() <= 1.2
    assert aux["specific_gate"].shape == (3, H, 6)
    assert torch.all((aux["specific_gate"] > 0).sum(-1) == 2)  # Top-2 routing


@pytest.mark.parametrize("prior_mode", ["learned", "fixed"])
def test_prior_starts_as_weighted_sum_of_candidates(prior_mode):
    x_hist, x_fut, memory, priors = inputs()
    _, aux = make_model(prior_mode=prior_mode)(x_hist, x_fut, memory, priors)
    expected = 0.60 * priors[:, 0] + 0.25 * priors[:, 1] + 0.10 * priors[:, 2] + 0.05 * priors[:, 3]
    torch.testing.assert_close(aux["prior"], expected)


def test_no_prior_variant_ignores_prior_candidates():
    model = make_model(prior_mode="none")
    x_hist, x_fut, memory, priors = inputs()
    first, _ = model(x_hist, x_fut, memory, priors)
    second, _ = model(x_hist, x_fut, memory, torch.full_like(priors, 0.7))
    assert model.residual_limit == 1.2
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_prior_router_exists_only_in_adaptive_variant():
    assert not hasattr(make_model(), "prior_router")
    adaptive = make_model(prior_mode="adaptive")
    assert hasattr(adaptive, "prior_router")
    assert sum(p.numel() for p in adaptive.parameters()) > sum(p.numel() for p in make_model().parameters())


def test_every_unit_parameter_receives_gradient():
    model = make_model().train()
    model.update_state_memory(torch.randn(4, 16), torch.ones(4, dtype=torch.bool))
    prediction, aux = model(*inputs())
    (prediction.sum() + aux["moe_balance_loss"]).backward()
    unused = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    assert unused == []


def test_state_memory_keeps_latest_entries():
    model = make_model(memory_size=4)
    model.update_state_memory(torch.arange(6.0).repeat(16, 1).T, torch.ones(6, dtype=torch.bool))
    assert int(model.state_memory_count) == 4
    torch.testing.assert_close(model.state_memory[:, 0], torch.tensor([2.0, 3.0, 4.0, 5.0]))


def test_learned_prior_coefficients_are_trainable():
    model = make_model()
    assert model.prior_logits.requires_grad
    torch.testing.assert_close(model.prior_coefficients(), torch.tensor([0.60, 0.25, 0.10, 0.05]))
    assert make_model(prior_mode="fixed").prior_logits is None


@pytest.mark.parametrize("switch", ["use_daily_feature", "use_power_summary"])
def test_dpe_feature_ablation_changes_output(switch):
    full, _ = make_model()(*inputs())
    torch.manual_seed(0)
    ablated_model = make_model(**{switch: False})
    ablated_model.load_state_dict(make_model().state_dict())
    ablated, _ = ablated_model(*inputs())
    assert not torch.allclose(full, ablated)


ALL_VARIANTS = [
    "full_model", "fixed_prior", "no_prior", "adaptive_prior", "prior_only", "static_graph", "no_graph",
    "specific_only", "shared_only", "dense_routing", "single_residual", "no_state_memory", "no_future_nwp",
    "no_daily_feature", "no_power_summary",
]


def test_variant_list_matches_config():
    import json
    from dpdgmoe.trainer import DEFAULT_CONFIG

    assert sorted(json.loads(DEFAULT_CONFIG.read_text())["variants"]) == sorted(ALL_VARIANTS)


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_all_configured_variants_run(variant):
    settings = dpdgmoe_settings(H, variant)
    switches = {
        key: settings[key]
        for key in (
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
    }
    prediction, _ = make_model(**switches)(*inputs())
    assert torch.isfinite(prediction).all()
