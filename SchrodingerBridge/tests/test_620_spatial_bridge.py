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
    _, target_velocity = objective._vertical_state(content, target, t)
    expected = (target - losses620._lowpass(target, 3)) - (content - losses620._lowpass(content, 3))
    assert torch.allclose(target_velocity, expected, atol=1e-6)


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
