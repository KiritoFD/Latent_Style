"""Phase 2A TDD: Style token masking (The Blindfolded Tokenizer).

Tests for random patch dropout + spatial shuffle on style tokens,
applied in StyleConditioner620 before returning img_tokens.

Theory (docs/630/mask.md):
- Content is globally topological -> destroyed by high-ratio dropout + shuffle
- Style is locally stationary -> survives masking
- Result: forces tokenizer to extract pure texture/color statistics,
  breaking Gate Collapse by eliminating content leakage risk.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config_schema import ModelConfig  # noqa: E402
from spectral_bridge620 import SpectralODEBridge620, build_spectral_ode_bridge_from_config  # noqa: E402
from style_encoder620 import StyleConditioner620  # noqa: E402


def _make_model_cfg(**overrides) -> ModelConfig:
    """Minimal config for SpectralODEBridge620."""
    cfg = ModelConfig(
        latent_channels=2,
        num_styles=2,
        base_dim=8,
        time_dim=8,
        num_res_blocks=1,
        style_attn_num_heads=2,
        tokenizer_dino_dim=6,
        contract_family="620_spectral_ode",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_conditioner(**overrides) -> StyleConditioner620:
    """Build a small StyleConditioner620 for testing."""
    cfg = _make_model_cfg()
    cond = StyleConditioner620(
        dino_dim=cfg.tokenizer_dino_dim,
        model_dim=cfg.base_dim,
        num_styles=cfg.num_styles,
        num_memory_tokens=16,
    )
    for k, v in overrides.items():
        setattr(cond, k, v)
    return cond


# ===== RED: Tests that should fail until masking is implemented =====


def test_style_mask_ratio_config_field_exists():
    """Config field style_mask_ratio must exist and default to 0.0 (disabled)."""
    cfg = _make_model_cfg()
    assert hasattr(cfg, "style_mask_ratio"), "style_mask_ratio config field missing"
    assert float(getattr(cfg, "style_mask_ratio", 0.0)) == 0.0


def test_style_mask_mode_config_field_exists():
    """Config field style_mask_mode must exist and default to 'none'."""
    cfg = _make_model_cfg()
    assert hasattr(cfg, "style_mask_mode"), "style_mask_mode config field missing"
    assert str(getattr(cfg, "style_mask_mode", "none")).strip().lower() == "none"


def test_conditioner_has_mask_attributes():
    """StyleConditioner620 must expose mask_ratio and mask_mode attributes."""
    cond = _make_conditioner()
    assert hasattr(cond, "mask_ratio"), "mask_ratio attribute missing"
    assert hasattr(cond, "mask_mode"), "mask_mode attribute missing"
    assert float(cond.mask_ratio) == 0.0
    assert str(cond.mask_mode) == "none"


def test_random_dropout_reduces_token_count():
    """Random dropout mode must reduce token count to keep_ratio * N."""
    cond = _make_conditioner()
    cond.mask_ratio = 0.75  # drop 75%, keep 25%
    cond.mask_mode = "random"
    batch, n_tokens, dino_dim = 2, 16, 6
    patches = torch.randn(batch, n_tokens, dino_dim)
    cls = torch.randn(batch, dino_dim)
    style_id = torch.tensor([0, 1])
    img_tokens, img_global = cond(
        style_dino_patches=patches,
        style_dino_cls=cls,
        style_id=style_id,
        batch=batch,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected_keep = int(n_tokens * 0.25)  # 4 tokens
    assert img_tokens.shape[1] == expected_keep, (
        f"random dropout: expected {expected_keep} tokens, got {img_tokens.shape[1]}"
    )


def test_shuffle_mode_preserves_token_count():
    """Shuffle mode must preserve token count but change order."""
    cond = _make_conditioner()
    cond.mask_ratio = 0.5  # unused in shuffle mode
    cond.mask_mode = "shuffle"
    batch, n_tokens, dino_dim = 2, 16, 6
    patches = torch.randn(batch, n_tokens, dino_dim)
    cls = torch.randn(batch, dino_dim)
    style_id = torch.tensor([0, 1])
    img_tokens, _ = cond(
        style_dino_patches=patches,
        style_dino_cls=cls,
        style_id=style_id,
        batch=batch,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert img_tokens.shape[1] == n_tokens, (
        f"shuffle mode: expected {n_tokens} tokens, got {img_tokens.shape[1]}"
    )


def test_none_mode_passes_through():
    """None mode must pass through all tokens unchanged."""
    cond = _make_conditioner()
    cond.mask_ratio = 0.75  # should be ignored
    cond.mask_mode = "none"
    batch, n_tokens, dino_dim = 2, 16, 6
    patches = torch.randn(batch, n_tokens, dino_dim)
    cls = torch.randn(batch, dino_dim)
    style_id = torch.tensor([0, 1])
    img_tokens, _ = cond(
        style_dino_patches=patches,
        style_dino_cls=cls,
        style_id=style_id,
        batch=batch,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert img_tokens.shape[1] == n_tokens


def test_masking_propagated_from_config():
    """build_spectral_ode_bridge_from_config must propagate mask config to conditioner."""
    cfg = _make_model_cfg(style_mask_ratio=0.6, style_mask_mode="random")
    model = build_spectral_ode_bridge_from_config(cfg)
    assert hasattr(model, "style_conditioner")
    assert float(model.style_conditioner.mask_ratio) == 0.6
    assert str(model.style_conditioner.mask_mode) == "random"


def test_random_dropout_is_stochastic():
    """Two calls with same input must produce different token selections (stochastic)."""
    cond = _make_conditioner()
    cond.mask_ratio = 0.5
    cond.mask_mode = "random"
    torch.manual_seed(42)
    batch, n_tokens, dino_dim = 1, 32, 6
    patches = torch.randn(batch, n_tokens, dino_dim)
    cls = torch.randn(batch, dino_dim)
    style_id = torch.tensor([0])
    out1, _ = cond(
        style_dino_patches=patches, style_dino_cls=cls, style_id=style_id,
        batch=batch, device=torch.device("cpu"), dtype=torch.float32,
    )
    out2, _ = cond(
        style_dino_patches=patches, style_dino_cls=cls, style_id=style_id,
        batch=batch, device=torch.device("cpu"), dtype=torch.float32,
    )
    # Different random subsets should (almost certainly) produce different outputs
    assert not torch.allclose(out1, out2), "random dropout must be stochastic"


def test_full_bridge_forward_with_masking():
    """Full SpectralODEBridge620 forward must work with masking enabled."""
    cfg = _make_model_cfg(style_mask_ratio=0.5, style_mask_mode="random")
    model = build_spectral_ode_bridge_from_config(cfg)
    model.eval()  # eval mode to avoid dropout noise in self-attn
    x = torch.randn(2, 2, 8, 8)
    t = torch.tensor([0.5, 0.5])
    style_id = torch.tensor([0, 1])
    patches = torch.randn(2, 16, 6)
    cls = torch.randn(2, 6)
    out = model(x, t=t, style_id=style_id, style_dino_patches=patches, style_dino_cls=cls)
    assert "ll" in out and "lh" in out and "hl" in out
    # DWT halves spatial dims: 8x8 -> 4x4
    assert out["ll"].shape == (2, 2, 4, 4)
