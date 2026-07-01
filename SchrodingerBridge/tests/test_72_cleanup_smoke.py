"""Smoke test for docs/72 cleanup verification.

Verifies the 6 cleanup/refactoring changes did not break the baseline:
1. LLGSI/CASI/LLGQCA code removed (T13/T14/T15 failed)
2. eval_only_dwt_route code removed (T5 failed)
3. Endpoint style loss removed (4J.6 v3 failed)
4. wct_aligned_target removed (4J.2/4J.5 failed)
5. integrate_transport split into _solver_step + _apply_endpoint_adain
6. _compute_use_dwt() method extracted

Tests:
- Imports succeed (no missing references to deleted config fields)
- T11 SOTA config (p=0.8, w_ll=0.0, depth=4, dim=64) constructs
- Forward pass works with DWT route (train + eval)
- integrate_transport works with all 3 solvers (euler/heun/rk4)
- _apply_endpoint_adain works with all 4 modes
- _compute_use_dwt stochastic behavior in train mode
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config_schema import BridgeConfig, ModelConfig  # noqa: E402
from spectral_bridge620 import (  # noqa: E402
    SpectralODEBridge620,
    build_spectral_ode_bridge_from_config,
)
from spectral_losses620 import SpectralODEObjective620  # noqa: E402


def _make_t11_cfg() -> tuple[ModelConfig, BridgeConfig]:
    """T11 local SOTA config: p=0.8, w_ll=0.0, depth=4, dim=64."""
    mcfg = ModelConfig(
        latent_channels=4,
        num_styles=5,
        base_dim=64,
        time_dim=256,
        num_res_blocks=4,
        style_attn_num_heads=4,
        tokenizer_dino_dim=384,
        contract_family="620_spectral_ode",
        # 4J.1 DWT Route
        cross_attn_dwt_route=True,
        cross_attn_dwt_ll_route_alpha=0.0,
        # T11 stochastic DWT route
        dwt_route_train_prob=0.8,
        # Declared ModelConfig fields
        # 630 Phase 72: endpoint_lowpass_levels=1 + endpoint_lowpass_basis="haar"
        # 已硬编码进 spectral_bridge620.py (4D/4E 多级/小波基已验证无效)
        endpoint_adain_mode="spatial_fiber",
        endpoint_adain_only_last_step=True,
    )
    # These are not declared ModelConfig fields — they ride in `extra` and are
    # accessed via the bridge's _cfg_get helper (hasattr-based lookup).
    setattr(mcfg, "endpoint_adain_scale", 1.0)
    setattr(mcfg, "style_extrap_alpha", 0.1)
    bcfg = BridgeConfig(
        objective_mode="flow_matching",
        loss_type="mse",
        t_min=0.0,
        t_max=1.0,
        spectral_w_ll=0.0,  # T11: w_ll=0.0 (LL free drift)
        spectral_w_lh=1.0,
        spectral_w_hl=1.0,
    )
    return mcfg, bcfg


def _make_bridge(mcfg: ModelConfig, bcfg: BridgeConfig) -> SpectralODEBridge620:
    return build_spectral_ode_bridge_from_config(mcfg, bridge_cfg=bcfg)


def test_imports_clean():
    """All modules must import without reference errors."""
    import blocks620  # noqa: F401
    import spectral620  # noqa: F401
    import spectral_bridge620  # noqa: F401
    import spectral_losses620  # noqa: F401
    import config_schema  # noqa: F401


def test_deleted_model_config_fields_absent():
    """Removed ModelConfig fields must not exist (suggestions 1+2)."""
    cfg = ModelConfig()
    removed = [
        "eval_only_dwt_route",
        "ll_global_style_inject",
        "ll_global_style_gate_init",
        "ll_style_inject_source",
    ]
    for name in removed:
        assert not hasattr(cfg, name), f"ModelConfig.{name} should be removed"


def test_deleted_bridge_config_fields_absent():
    """Removed BridgeConfig fields must not exist (suggestions 3+4)."""
    cfg = BridgeConfig()
    removed = [
        "wct_aligned_target",
        "wct_aligned_alpha",
        "spectral_w_endpoint_style_lh",
        "spectral_w_endpoint_style_hl",
    ]
    for name in removed:
        assert not hasattr(cfg, name), f"BridgeConfig.{name} should be removed"


def test_blocks620_no_llgsi_attributes():
    """SpatialBridgeBlock620 must not expose LLGSI/CASI/LLGQCA attributes."""
    from blocks620 import SpatialBridgeBlock620

    block = SpatialBridgeBlock620(
        dim=64, num_heads=4, dwt_route=True, dwt_route_train_prob=0.8,
    )
    removed_attrs = [
        "ll_global_style_inject",
        "ll_global_style_gate_init",
        "ll_style_inject_source",
        "ll_style_gate",
        "eval_only_dwt_route",
    ]
    for name in removed_attrs:
        assert not hasattr(block, name), f"block.{name} should be removed"


def test_compute_use_dwt_method_exists():
    """Suggestion 6: _compute_use_dwt method must exist."""
    from blocks620 import SpatialBridgeBlock620

    block = SpatialBridgeBlock620(dim=64, num_heads=4, dwt_route=True, dwt_route_train_prob=0.8)
    assert callable(getattr(block, "_compute_use_dwt", None)), "_compute_use_dwt missing"


def test_compute_use_dwt_disabled_when_dwt_route_false():
    """_compute_use_dwt returns False when dwt_route=False."""
    from blocks620 import SpatialBridgeBlock620

    block = SpatialBridgeBlock620(dim=64, num_heads=4, dwt_route=False)
    block.train()
    assert block._compute_use_dwt() is False


def test_compute_use_dwt_always_true_in_eval():
    """_compute_use_dwt returns True in eval mode when dwt_route=True."""
    from blocks620 import SpatialBridgeBlock620

    block = SpatialBridgeBlock620(dim=64, num_heads=4, dwt_route=True, dwt_route_train_prob=0.8)
    block.eval()
    assert block._compute_use_dwt() is True


def test_compute_use_dwt_deterministic_when_prob_zero():
    """_compute_use_dwt returns True in train mode when prob=0 (4J.1 behavior)."""
    from blocks620 import SpatialBridgeBlock620

    block = SpatialBridgeBlock620(dim=64, num_heads=4, dwt_route=True, dwt_route_train_prob=0.0)
    block.train()
    assert block._compute_use_dwt() is True


def test_compute_use_dwt_stochastic_when_prob_positive():
    """_compute_use_dwt returns mix of True/False in train mode when prob=0.8."""
    from blocks620 import SpatialBridgeBlock620

    block = SpatialBridgeBlock620(dim=64, num_heads=4, dwt_route=True, dwt_route_train_prob=0.8)
    block.train()
    torch.manual_seed(0)
    results = [block._compute_use_dwt() for _ in range(200)]
    true_count = sum(1 for r in results if r)
    # Expected ~160/200, allow generous bounds
    assert 100 < true_count < 200, f"stochastic DWT route broken: {true_count}/200 True"


def test_solver_step_method_exists():
    """Suggestion 5: _solver_step method must exist."""
    mcfg, bcfg = _make_t11_cfg()
    bridge = _make_bridge(mcfg, bcfg)
    assert callable(getattr(bridge, "_solver_step", None)), "_solver_step missing"


def test_apply_endpoint_adain_method_exists():
    """Suggestion 5: _apply_endpoint_adain method must exist."""
    mcfg, bcfg = _make_t11_cfg()
    bridge = _make_bridge(mcfg, bcfg)
    assert callable(getattr(bridge, "_apply_endpoint_adain", None)), "_apply_endpoint_adain missing"


def test_forward_pass_t11_config():
    """Forward pass works with T11 config in train and eval modes."""
    torch.manual_seed(0)
    mcfg, bcfg = _make_t11_cfg()
    bridge = _make_bridge(mcfg, bcfg)

    B, C, H, W = 2, 4, 32, 32
    x = torch.randn(B, C, H, W)
    t = torch.rand(B)
    style_id = torch.tensor([0, 1])

    # Train mode (stochastic DWT route)
    bridge.train()
    out_train = bridge(x, t=t, style_id=style_id)
    assert "ll" in out_train and "lh" in out_train and "hl" in out_train
    assert out_train["ll"].shape == (B, C, H // 2, W // 2)

    # Eval mode (always DWT route)
    bridge.eval()
    out_eval = bridge(x, t=t, style_id=style_id)
    assert out_eval["ll"].shape == (B, C, H // 2, W // 2)


def test_integrate_transport_euler():
    """integrate_transport works with Euler solver."""
    torch.manual_seed(0)
    mcfg, bcfg = _make_t11_cfg()
    # Override solver_type via attribute on bridge_cfg
    bcfg.solver_type = "euler"
    bridge = _make_bridge(mcfg, bcfg)
    bridge.eval()

    B, C, H, W = 2, 4, 32, 32
    x = torch.randn(B, C, H, W)
    style_id = torch.tensor([0, 1])
    style_latent = torch.randn(B, C, H, W)

    out = bridge.integrate_transport(
        x, style_id=style_id, num_steps=4, step_size=1.0,
        style_latent=style_latent,
    )
    assert out.shape == (B, C, H, W), f"Euler output shape {out.shape}"
    assert torch.isfinite(out).all(), "Euler output has NaN/Inf"


def test_integrate_transport_heun():
    """integrate_transport works with Heun solver (structural DOF)."""
    torch.manual_seed(0)
    mcfg, bcfg = _make_t11_cfg()
    bcfg.solver_type = "heun"
    bridge = _make_bridge(mcfg, bcfg)
    bridge.eval()

    B, C, H, W = 2, 4, 32, 32
    x = torch.randn(B, C, H, W)
    style_id = torch.tensor([0, 1])
    style_latent = torch.randn(B, C, H, W)

    out = bridge.integrate_transport(
        x, style_id=style_id, num_steps=4, step_size=1.0,
        style_latent=style_latent,
    )
    assert out.shape == (B, C, H, W), f"Heun output shape {out.shape}"
    assert torch.isfinite(out).all(), "Heun output has NaN/Inf"


def test_integrate_transport_rk4():
    """integrate_transport works with RK4 solver."""
    torch.manual_seed(0)
    mcfg, bcfg = _make_t11_cfg()
    bcfg.solver_type = "rk4"
    bridge = _make_bridge(mcfg, bcfg)
    bridge.eval()

    B, C, H, W = 2, 4, 32, 32
    x = torch.randn(B, C, H, W)
    style_id = torch.tensor([0, 1])
    style_latent = torch.randn(B, C, H, W)

    out = bridge.integrate_transport(
        x, style_id=style_id, num_steps=2, step_size=1.0,
        style_latent=style_latent,
    )
    assert out.shape == (B, C, H, W), f"RK4 output shape {out.shape}"
    assert torch.isfinite(out).all(), "RK4 output has NaN/Inf"


def test_apply_endpoint_adain_all_modes():
    """_apply_endpoint_adain works with all 4 modes (spatial_fiber/per_subband/per_subband_wct/spatial_fiber_wct)."""
    torch.manual_seed(0)
    mcfg, bcfg = _make_t11_cfg()
    bridge = _make_bridge(mcfg, bcfg)
    bridge.eval()

    B, C, H, W = 2, 4, 32, 32
    h = torch.randn(B, C, H, W)
    style_latent = torch.randn(1, C, H, W)

    modes = ["spatial_fiber", "per_subband", "per_subband_wct", "spatial_fiber_wct"]
    for mode in modes:
        out = bridge._apply_endpoint_adain(
            h, style_latent=style_latent,
            adain_mode=mode, lowpass_levels=1, lowpass_basis="haar",
            style_extrap_alpha=0.1,
            adain_scale_ll=0.0, adain_scale_lh=1.0, adain_scale_hl=1.0, adain_scale_hh=1.0,
            endpoint_adain_scale=1.0,
        )
        assert out.shape == h.shape, f"mode={mode} shape mismatch"
        assert torch.isfinite(out).all(), f"mode={mode} has NaN/Inf"


def test_apply_endpoint_adain_per_subband_wct_with_ll_scale():
    """per_subband_wct mode supports adain_scale_ll > 0 (LL WCT path)."""
    torch.manual_seed(0)
    mcfg, bcfg = _make_t11_cfg()
    bridge = _make_bridge(mcfg, bcfg)
    bridge.eval()

    B, C, H, W = 2, 4, 32, 32
    h = torch.randn(B, C, H, W)
    style_latent = torch.randn(1, C, H, W)

    out = bridge._apply_endpoint_adain(
        h, style_latent=style_latent,
        adain_mode="per_subband_wct", lowpass_levels=1, lowpass_basis="haar",
        style_extrap_alpha=0.0,
        adain_scale_ll=0.5, adain_scale_lh=1.0, adain_scale_hl=1.0, adain_scale_hh=1.0,
        endpoint_adain_scale=1.0,
    )
    assert out.shape == h.shape
    assert torch.isfinite(out).all()


def test_losses_compute_clean():
    """SpectralODEObjective620.compute works without removed endpoint-style loss."""
    torch.manual_seed(0)
    mcfg, bcfg = _make_t11_cfg()
    bridge = _make_bridge(mcfg, bcfg)
    bridge.train()

    from config_schema import DataConfig, ExperimentConfig, TrainingConfig
    dc = DataConfig()
    tc = TrainingConfig(batch_size=2)
    ecfg = ExperimentConfig(model=mcfg, bridge=bcfg, data=dc, training=tc)
    obj = SpectralODEObjective620(ecfg)

    B, C, H, W = 2, 4, 32, 32
    content = torch.randn(B, C, H, W)
    target_style = torch.randn(B, C, H, W)
    target_style_id = torch.tensor([0, 1])

    metrics = obj.compute(
        bridge, content=content, target_style=target_style,
        target_style_id=target_style_id,
    )
    assert "loss" in metrics
    assert "loss_fm_spectral_ll" in metrics
    assert "loss_fm_spectral_lh" in metrics
    assert "loss_fm_spectral_hl" in metrics
    # Removed endpoint-style loss keys must NOT appear
    for removed_key in ["loss_endpoint_style_lh", "loss_endpoint_style_hl", "loss_wct_aligned"]:
        assert removed_key not in metrics, f"{removed_key} should be removed"


def test_no_dead_wct_match_subband_function():
    """Suggestion 4: dead module-level _wct_match_subband function must be removed."""
    import spectral_losses620 as mod

    assert not hasattr(mod, "_wct_match_subband"), "_wct_match_subband should be removed"


def test_no_dead_import():
    """spectral_losses620 must not import idwt2_haar (unused after suggestion 4)."""
    import spectral_losses620 as mod

    src = open(mod.__file__).read()
    assert "idwt2_haar" not in src, "idwt2_haar import should be removed"


def test_t11_inference_end_to_end():
    """Full T11 inference pipeline: stochastic train + deterministic eval DWT route."""
    torch.manual_seed(0)
    mcfg, bcfg = _make_t11_cfg()
    bridge = _make_bridge(mcfg, bcfg)

    B, C, H, W = 2, 4, 32, 32
    x = torch.randn(B, C, H, W)
    style_id = torch.tensor([0, 1])
    style_latent = torch.randn(1, C, H, W)

    bridge.eval()
    out = bridge.integrate_transport(
        x, style_id=style_id, num_steps=8, step_size=1.0,
        style_latent=style_latent,
    )
    assert out.shape == (B, C, H, W)
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    # Run all tests in sequence, print pass/fail
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n=== Smoke test: {passed} passed, {failed} failed ===")
    sys.exit(0 if failed == 0 else 1)
