from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import losses620  # noqa: E402
from config_schema import BridgeConfig, ExperimentConfig, ModelConfig  # noqa: E402
from losses620 import SpatialBridgeObjective620  # noqa: E402
from model620 import SpatialBridge620  # noqa: E402
from utils.dataset import AdaCUTLatentDataset  # noqa: E402


def _cfg() -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.model = ModelConfig(
        latent_channels=2,
        num_styles=2,
        base_dim=8,
        time_dim=8,
        num_res_blocks=1,
        style_attn_num_heads=2,
        tokenizer_dino_dim=6,
        contract_family="620_spatial_bridge",
    )
    cfg.bridge = BridgeConfig(
        w_flow=1.0,
        single_step_swd_weight=0.5,
        single_step_edge_weight=0.1,
        semantic_swd_num_projections=4,
        training_target_projection_kernel=3,
        training_target_projection_mode="source_low_target_high",
        bridge_sigma=0.02,
    )
    return cfg


def test_dataset_emits_selected_target_dino_tensors(tmp_path: Path) -> None:
    data_root = tmp_path / "latents"
    for style in ("photo", "monet"):
        (data_root / style).mkdir(parents=True)
    torch.save(torch.zeros(2, 4, 4), data_root / "photo" / "p0.pt")
    torch.save(torch.ones(2, 4, 4), data_root / "monet" / "m0.pt")
    torch.save(torch.full((2, 4, 4), 2.0), data_root / "monet" / "m1.pt")

    cls = torch.arange(18, dtype=torch.float32).view(3, 6)
    patches = torch.arange(3 * 4 * 6, dtype=torch.float32).view(3, 4, 6)
    dino_cache = tmp_path / "dino.pt"
    torch.save(
        {
            "rows": [
                {"style": "photo", "stem": "p0"},
                {"style": "monet", "stem": "m0"},
                {"style": "monet", "stem": "m1"},
            ],
            "cls_embeddings": cls,
            "patch_embeddings": patches,
        },
        dino_cache,
    )
    ds = AdaCUTLatentDataset(
        data_root=str(data_root),
        style_subdirs=["photo", "monet"],
        allow_hflip=False,
        identity_ratio=0.0,
        virtual_length_multiplier=1.0,
        dino_cache_path=str(dino_cache),
        dino_cache_required=True,
    )
    ds._cache_content_style_ids[0] = 0
    ds._cache_target_style_ids[0] = 1
    ds._cache_content_rands[0] = 0.0
    ds._cache_target_rands[0] = 0.75

    item = ds[0]
    assert torch.equal(item["target_style"], torch.full((2, 4, 4), 2.0))
    assert torch.equal(item["target_style_dino_cls"], cls[2])
    assert torch.equal(item["target_style_dino_patches"], patches[2])
    assert torch.equal(item["target_style_dino_hw"], torch.tensor([2, 2]))


def test_620_loss_uses_forward_not_integrate_transport() -> None:
    class DummyModel(torch.nn.Module):
        bridge_sigma = 0.02
        last_debug = {}

        def forward(self, x, **kwargs):  # noqa: ANN001
            return torch.zeros_like(x)

        def integrate_transport(self, *args, **kwargs):  # noqa: ANN001
            raise AssertionError("training loss must not integrate transport")

    objective = SpatialBridgeObjective620(_cfg())
    batch = torch.randn(2, 2, 4, 4)
    out = objective.compute(
        DummyModel(),
        content=batch,
        target_style=batch + 0.1,
        target_style_id=torch.tensor([0, 1]),
        conditioning={"target_style_dino_patches": torch.randn(2, 4, 6)},
    )
    assert torch.isfinite(out["loss"])


def test_vertical_fm_target_velocity_is_target_highpass_delta() -> None:
    objective = SpatialBridgeObjective620(_cfg())
    content = torch.randn(2, 2, 5, 5)
    target = torch.randn(2, 2, 5, 5)
    t = torch.tensor([0.25, 0.75])
    x_t, target_velocity = objective._vertical_state(content, target, t)
    projected_target, _ = objective._project_training_target(content, target)
    c_low = losses620._lowpass(content, 3)
    c_high = content - c_low
    t_high = projected_target - losses620._lowpass(projected_target, 3)
    expected_x_t = c_low + (1.0 - t).view(-1, 1, 1, 1) * c_high + t.view(-1, 1, 1, 1) * t_high
    expected = (projected_target - losses620._lowpass(projected_target, 3)) - (content - losses620._lowpass(content, 3))
    assert torch.allclose(x_t, expected_x_t, atol=1e-6)
    assert torch.allclose(target_velocity, expected, atol=1e-6)


def test_vertical_fm_optional_low_anchor_blends_target_lowpass() -> None:
    cfg = _cfg()
    cfg.bridge.training_target_projection_low_anchor = 0.5
    objective = SpatialBridgeObjective620(cfg)
    content = torch.randn(2, 2, 5, 5)
    target = torch.randn(2, 2, 5, 5)
    t = torch.tensor([0.25, 0.75])
    x_t, target_velocity = objective._vertical_state(content, target, t)
    projected_target, _ = objective._project_training_target(content, target)
    c_low = losses620._lowpass(content, 3)
    t_low = losses620._lowpass(projected_target, 3)
    c_high = content - c_low
    t_high = projected_target - t_low
    expected_x_t = 0.5 * c_low + 0.5 * t_low + (1.0 - t).view(-1, 1, 1, 1) * c_high + t.view(-1, 1, 1, 1) * t_high
    assert torch.allclose(x_t, expected_x_t, atol=1e-6)
    assert torch.allclose(target_velocity, t_high - c_high, atol=1e-6)


def test_vertical_fm_target_linear_low_mode_moves_low_frequency() -> None:
    cfg = _cfg()
    cfg.bridge.training_target_projection_low_mode = "target_linear"
    objective = SpatialBridgeObjective620(cfg)
    content = torch.randn(2, 2, 5, 5)
    target = torch.randn(2, 2, 5, 5)
    t = torch.tensor([0.25, 0.75])
    x_t, target_velocity = objective._vertical_state(content, target, t)
    projected_target, _ = objective._project_training_target(content, target)
    c_low = losses620._lowpass(content, 3)
    t_low = losses620._lowpass(projected_target, 3)
    c_high = content - c_low
    t_high = projected_target - t_low
    t4 = t.view(-1, 1, 1, 1)
    expected_x_t = (1.0 - t4) * c_low + t4 * t_low + (1.0 - t4) * c_high + t4 * t_high
    expected_velocity = (t_low - c_low) + (t_high - c_high)
    assert torch.allclose(x_t, expected_x_t, atol=1e-6)
    assert torch.allclose(target_velocity, expected_velocity, atol=1e-6)


def test_projected_target_defaults_to_source_low_target_high() -> None:
    cfg = _cfg()
    objective = SpatialBridgeObjective620(cfg)
    content = torch.randn(2, 2, 6, 6)
    target = torch.randn(2, 2, 6, 6)
    projected, metrics = objective._project_training_target(content, target)
    c_low = losses620._lowpass(content, 3)
    t_low = losses620._lowpass(target, 3)
    t_high = target - t_low
    expected = c_low + t_high
    assert torch.allclose(projected, expected, atol=1e-6)
    assert float(metrics["training_target_projection_active"].item()) == 1.0
    assert float(metrics["training_target_projection_mode_source_low_target_high"].item()) == 1.0


def test_projected_target_pure_vertical_flow_anchors_source_lowpass() -> None:
    cfg = _cfg()
    cfg.bridge.training_target_projection_mode = "pure_vertical_flow"
    objective = SpatialBridgeObjective620(cfg)
    content = torch.randn(2, 2, 6, 6)
    target = torch.randn(2, 2, 6, 6)
    projected, metrics = objective._project_training_target(content, target)
    c_low = losses620._lowpass(content, 3)
    t_low = losses620._lowpass(target, 3)
    t_high = target - t_low
    expected = c_low + t_high
    assert torch.allclose(projected, expected, atol=1e-6)
    assert float(metrics["training_target_projection_mode_pure_vertical_flow"].item()) == 1.0


def test_single_step_swd_receives_velocity_derived_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyModel(torch.nn.Module):
        bridge_sigma = 0.02
        last_debug = {}

        def forward(self, x, **kwargs):  # noqa: ANN001
            return torch.ones_like(x) * 0.25

    captured = {}

    def fake_swd(a, b, *, dirs):  # noqa: ANN001
        captured["endpoint"] = a.detach().clone()
        return a.new_tensor(0.0)

    objective = SpatialBridgeObjective620(_cfg())
    monkeypatch.setattr(objective, "_sample_t", lambda content: torch.full((content.shape[0],), 0.4, device=content.device))
    monkeypatch.setattr(losses620, "_sliced_wasserstein", fake_swd)
    content = torch.randn(2, 2, 4, 4)
    objective.compute(
        DummyModel(),
        content=content,
        target_style=torch.randn_like(content),
        target_style_id=torch.tensor([0, 1]),
        conditioning={},
    )
    x_t = objective.last_debug["x_t"]
    assert torch.allclose(captured["endpoint"], x_t + 0.6 * 0.25, atol=1e-6)


def test_620_endpoint_decomposition_metrics_track_low_and_high_bands(monkeypatch: pytest.MonkeyPatch) -> None:
    class PerfectVerticalModel(torch.nn.Module):
        bridge_sigma = 0.02
        last_debug = {}

        def __init__(self) -> None:
            super().__init__()
            self.velocity = None

        def forward(self, x, **kwargs):  # noqa: ANN001
            del x, kwargs
            return self.velocity

    objective = SpatialBridgeObjective620(_cfg())
    monkeypatch.setattr(objective, "_sample_t", lambda content: torch.full((content.shape[0],), 0.5, device=content.device))
    content = torch.randn(2, 2, 6, 6)
    target = torch.randn_like(content) + 0.5
    _x_t, target_velocity = objective._vertical_state(content, target, torch.full((2,), 0.5))
    model = PerfectVerticalModel()
    model.velocity = target_velocity
    out = objective.compute(
        model,
        content=content,
        target_style=target,
        target_style_id=torch.tensor([0, 1]),
        conditioning={},
    )

    z_hat1 = objective.last_debug["z_hat1"]
    projected_target = objective.last_debug["projected_target"]
    z_low = losses620._lowpass(z_hat1, objective.lowpass_kernel)
    c_low = losses620._lowpass(content, objective.lowpass_kernel)
    t_low = losses620._lowpass(projected_target, objective.lowpass_kernel)
    z_high = z_hat1 - z_low
    t_high = projected_target - t_low
    assert torch.allclose(out["endpoint_low_to_source"], (z_low - c_low).abs().mean(), atol=1e-6)
    assert torch.allclose(out["endpoint_low_to_target"], (z_low - t_low).abs().mean(), atol=1e-6)
    assert torch.allclose(out["endpoint_high_to_target"], (z_high - t_high).abs().mean(), atol=1e-6)
    assert torch.isfinite(out["endpoint_low_target_ratio"])


def test_620_endpoint_lowfreq_loss_uses_target_lowpass(monkeypatch: pytest.MonkeyPatch) -> None:
    class ZeroModel(torch.nn.Module):
        bridge_sigma = 0.02
        last_debug = {}

        def forward(self, x, **kwargs):  # noqa: ANN001
            del kwargs
            return torch.zeros_like(x)

    cfg = _cfg()
    cfg.bridge.w_content_lowpass_anchor = 0.75
    objective = SpatialBridgeObjective620(cfg)
    monkeypatch.setattr(objective, "_sample_t", lambda content: torch.full((content.shape[0],), 0.5, device=content.device))
    content = torch.randn(2, 2, 6, 6)
    target = torch.randn_like(content) + 0.25

    out = objective.compute(
        ZeroModel(),
        content=content,
        target_style=target,
        target_style_id=torch.tensor([0, 1]),
        conditioning={},
    )

    z_hat1 = objective.last_debug["z_hat1"]
    projected_target = objective.last_debug["projected_target"]
    expected = torch.nn.functional.l1_loss(
        losses620._lowpass(z_hat1, objective.lowpass_kernel).float(),
        losses620._lowpass(projected_target, objective.lowpass_kernel).float(),
    )
    assert torch.allclose(out["loss_endpoint_lowfreq"], expected, atol=1e-6)
    assert torch.allclose(out["endpoint_lowfreq"], expected * 0.75, atol=1e-6)


def test_source_endpoint_aux_uses_t0_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    class RecorderModel(torch.nn.Module):
        bridge_sigma = 0.02
        last_debug = {}

        def __init__(self) -> None:
            super().__init__()
            self.seen_t = []

        def forward(self, x, t=None, **kwargs):  # noqa: ANN001
            self.seen_t.append(("forward", None if t is None else t.detach().clone()))
            return torch.zeros_like(x)

        def predict_endpoint(self, x, t=None, **kwargs):  # noqa: ANN001
            self.seen_t.append(("endpoint", None if t is None else t.detach().clone()))
            return x + 0.1

    cfg = _cfg()
    cfg.bridge.source_endpoint_aux_weight = 0.5
    objective = SpatialBridgeObjective620(cfg)
    monkeypatch.setattr(objective, "_sample_t", lambda content: torch.full((content.shape[0],), 0.4, device=content.device))
    model = RecorderModel()
    content = torch.randn(2, 2, 6, 6)
    target = torch.randn_like(content)

    out = objective.compute(
        model,
        content=content,
        target_style=target,
        target_style_id=torch.tensor([0, 1]),
        conditioning={},
    )

    endpoint_ts = [t for kind, t in model.seen_t if kind == "endpoint"]
    assert endpoint_ts
    assert torch.allclose(endpoint_ts[0], torch.zeros_like(endpoint_ts[0]), atol=1e-6)
    assert torch.isfinite(out["loss_source_endpoint_aux"])


def test_t_sampling_power_biases_samples_toward_low_t() -> None:
    cfg = _cfg()
    cfg.bridge.t_min = 0.0
    cfg.bridge.t_max = 1.0
    cfg.bridge.t_sampling_power = 2.5
    objective = SpatialBridgeObjective620(cfg)
    torch.manual_seed(123)
    t = objective._sample_t(torch.zeros(4096, 2, 4, 4))
    assert float(t.mean().item()) < 0.4
    assert float(t.min().item()) >= 0.0
    assert float(t.max().item()) <= 1.0


def test_endpoint_energy_band_penalizes_only_out_of_band_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    class OffsetModel(torch.nn.Module):
        bridge_sigma = 0.02
        last_debug = {}

        def __init__(self, delta: float) -> None:
            super().__init__()
            self.delta = float(delta)

        def forward(self, x, **kwargs):  # noqa: ANN001
            del kwargs
            return torch.full_like(x, self.delta)

    cfg = _cfg()
    cfg.bridge.endpoint_energy_band_weight = 1.0
    objective = SpatialBridgeObjective620(cfg)
    monkeypatch.setattr(objective, "_sample_t", lambda content: torch.full((content.shape[0],), 0.5, device=content.device))
    content = torch.full((2, 2, 4, 4), 0.2)
    target = torch.full((2, 2, 4, 4), 0.4)

    in_band = objective.compute(
        OffsetModel(0.0),
        content=content,
        target_style=target,
        target_style_id=torch.tensor([0, 1]),
        conditioning={},
    )
    out_band = objective.compute(
        OffsetModel(1.0),
        content=content,
        target_style=target,
        target_style_id=torch.tensor([0, 1]),
        conditioning={},
    )

    assert torch.allclose(in_band["loss_endpoint_energy_band"], torch.tensor(0.0), atol=1e-6)
    assert float(out_band["loss_endpoint_energy_band"].item()) > 0.0


def test_i2sb_single_final_step_has_zero_variance(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    model = SpatialBridge620(cfg.model, cfg.bridge)
    model.eval()

    def fail_noise(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("final I2SB step must have zero variance")

    monkeypatch.setattr(torch, "randn_like", fail_noise)
    x = torch.randn(1, 2, 4, 4)
    out = model.integrate_transport(x, style_id=torch.tensor([0]), num_steps=1)
    assert out.shape == x.shape

def test_620_moe_cross_attention_reports_router_metrics() -> None:
    cfg = _cfg()
    cfg.model.style_moe_enabled = True
    cfg.model.style_moe_num_experts = 3
    cfg.model.style_moe_router_hidden_dim = 8
    model = SpatialBridge620(cfg.model, cfg.bridge)
    x = torch.randn(2, 2, 4, 4)
    patches = torch.randn(2, 4, 6)
    out = model(x, t=torch.tensor([0.25, 0.75]), style_dino_patches=patches)
    assert out.shape == x.shape
    assert "style_moe_router_entropy" in model.last_debug
    assert "style_moe_router_max_prob" in model.last_debug
    assert torch.isfinite(model.last_debug["style_moe_router_entropy"])


def test_endpoint_lowhigh_mode_converts_endpoint_delta_back_to_velocity() -> None:
    cfg = _cfg()
    cfg.model.endpoint_head_mode = "endpoint_lowhigh"
    cfg.model.endpoint_lowpass_kernel = 3
    cfg.model.endpoint_velocity_floor = 0.05
    model = SpatialBridge620(cfg.model, cfg.bridge)
    x = torch.randn(2, 2, 4, 4)
    patches = torch.randn(2, 4, 6)
    t = torch.tensor([0.25, 0.75])
    out = model(x, t=t, style_dino_patches=patches)
    assert out.shape == x.shape
    assert float(model.last_debug["endpoint_head_mode_lowhigh"].item()) == 1.0
    endpoint = model.predict_endpoint(x, t=t, style_dino_patches=patches)
    assert endpoint.shape == x.shape
    assert torch.isfinite(model.last_debug["endpoint_pred_abs"])


def test_endpoint_lowhigh_style_heads_report_style_actuation_debug() -> None:
    cfg = _cfg()
    cfg.model.endpoint_head_mode = "endpoint_lowhigh"
    cfg.model.endpoint_style_hidden_dim = 16
    model = SpatialBridge620(cfg.model, cfg.bridge)
    x = torch.randn(2, 2, 4, 4)
    patches = torch.randn(2, 4, 6)
    _ = model(x, t=torch.tensor([0.2, 0.8]), style_dino_patches=patches)
    assert "endpoint_style_low_abs" in model.last_debug
    assert "endpoint_style_high_abs" in model.last_debug
    assert torch.isfinite(model.last_debug["endpoint_style_low_abs"])
    assert torch.isfinite(model.last_debug["endpoint_style_high_abs"])
