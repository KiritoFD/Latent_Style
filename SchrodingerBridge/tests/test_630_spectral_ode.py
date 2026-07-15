"""Tests for the active WEAVE contract (clean_base_v2_local).

Covers:
- M9: config style_attn_mode propagation to ResidualBlock
- Model build smoke for the active contract
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_schema import BridgeConfig, ModelConfig, load_experiment_config  # noqa: E402
from model import build_model_from_config  # noqa: E402
from model import WEAVE  # noqa: E402


ACTIVE_CONFIG = ROOT / "config.json"


def _load_active_config():
    return load_experiment_config(str(ACTIVE_CONFIG))


def test_active_config_builds_spectral_ode_bridge():
    """Active config must build a WEAVE."""
    cfg = _load_active_config()
    model = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge)
    assert isinstance(model, WEAVE), (
        f"Expected WEAVE, got {type(model).__name__}"
    )


def test_style_attn_mode_propagated_to_blocks():
    """M9 regression: config style_attn_mode must reach block.attn_mode.

    Previously spectral_bridge620.py did not pass attn_mode to
    ResidualBlock, so blocks silently defaulted to "softmax"
    even when config said "relu2". This test guards against regression.
    """
    cfg = _load_active_config()
    expected_mode = str(getattr(cfg.model, "style_attn_mode", "relu2")).strip().lower()
    model = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge)
    assert isinstance(model.blocks, torch.nn.ModuleList) and len(model.blocks) > 0
    for idx, block in enumerate(model.blocks):
        actual = str(getattr(block, "attn_mode", "<missing>")).strip().lower()
        assert actual == expected_mode, (
            f"block[{idx}].attn_mode = {actual!r}, expected {expected_mode!r} "
            f"(from config style_attn_mode)"
        )


def test_historical_target_dino_patches_uses_style_memory_contract():
    """Retired DINO label must keep the historical non-intrinsic checkpoint shape."""
    model_cfg = ModelConfig(
        latent_channels=4,
        num_styles=5,
        base_dim=64,
        time_dim=256,
        num_res_blocks=4,
        style_attn_num_heads=4,
        tokenizer_dino_dim=384,
        contract_family="weave",
        style_condition_source="target_dino_patches",
    )
    model = build_model_from_config(model_cfg, bridge_cfg=BridgeConfig())
    assert isinstance(model, WEAVE)
    assert model.use_intrinsic_style is False
    assert model.intrinsic_style_cnn is None


def test_forward_returns_three_velocity_subbands():
    """Active forward must return {'ll','lh','hl'} (HH removed - 628 L8 DEAD)."""
    cfg = _load_active_config()
    model = build_model_from_config(cfg.model, bridge_cfg=cfg.bridge).to("cuda")
    model.eval()
    B, C, H, W = 2, cfg.model.latent_channels, 32, 32
    x = torch.randn(B, C, H, W, device="cuda")
    t = torch.full((B,), 0.5, device="cuda")
    style_id = torch.zeros(B, dtype=torch.long, device="cuda")
    with torch.no_grad():
        out = model(x, t=t, style_id=style_id)
    assert set(out.keys()) == {"ll", "lh", "hl"}, f"unexpected keys: {set(out.keys())}"
    for k, v in out.items():
        assert v.shape == (B, C, H // 2, W // 2), f"{k} shape {v.shape}"
