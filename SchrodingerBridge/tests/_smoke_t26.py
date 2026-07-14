"""Standalone smoke test for Plan H (T26) per_subband_wct_ll_ycbcr mode."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch

from config_schema import BridgeConfig, ModelConfig
from model import WEAVE


def _make_t11_cfg():
    mcfg = ModelConfig(
        latent_channels=4, num_styles=5, base_dim=64, time_dim=64,
        num_res_blocks=4, style_attn_num_heads=4,
        style_cross_attn_gate_init=0.05, style_attn_temperature=1.0,
        style_shortcut_alpha=1.0,
        cross_attn_dwt_route=True,
        dwt_route_train_prob=0.8,
        endpoint_adain_mode="per_subband_wct",
        endpoint_adain_scale_ll=0.0,
        endpoint_adain_scale_lh=0.3,
        endpoint_adain_scale_hl=0.3,
        endpoint_adain_scale_hh=0.5,
    )
    # style_extrap_alpha is read as a top-level config key, set on mcfg directly
    mcfg.style_extrap_alpha = 0.4
    bcfg = BridgeConfig()
    return mcfg, bcfg


def main():
    torch.manual_seed(0)
    mcfg, bcfg = _make_t11_cfg()
    bridge = WEAVE(mcfg, bcfg)
    bridge.eval()

    B, C, H, W = 2, 4, 32, 32
    h = torch.randn(B, C, H, W)
    style_latent = torch.randn(1, C, H, W)

    # Test 1: shape and finiteness
    out = bridge._apply_endpoint_adain(
        h, style_latent=style_latent,
        adain_mode="per_subband_wct_ll_ycbcr",
        lowpass_levels=1, lowpass_basis="haar",
        style_extrap_alpha=0.0,
        adain_scale_ll=0.5, adain_scale_lh=1.0, adain_scale_hl=1.0, adain_scale_hh=1.0,
        endpoint_adain_scale=1.0,
    )
    assert out.shape == h.shape, f"shape mismatch: {out.shape} vs {h.shape}"
    assert torch.isfinite(out).all(), "NaN/Inf in output"
    print("[PASS] shape + finiteness")

    # Test 2: luma preservation (the key theoretical property of Plan H)
    y_in = h.mean(dim=1, keepdim=True)
    y_out = out.mean(dim=1, keepdim=True)
    luma_drift = (y_out - y_in).abs().max().item()
    print(f"[INFO] luma drift (scale_ll=0.5): {luma_drift:.6f}")
    assert luma_drift < 0.3, f"luma drift too large: {luma_drift}"

    # Test 3: scale_ll=0 should be identity on LL (no luma drift at all)
    out0 = bridge._apply_endpoint_adain(
        h, style_latent=style_latent,
        adain_mode="per_subband_wct_ll_ycbcr",
        lowpass_levels=1, lowpass_basis="haar",
        style_extrap_alpha=0.0,
        adain_scale_ll=0.0, adain_scale_lh=0.0, adain_scale_hl=0.0, adain_scale_hh=0.0,
        endpoint_adain_scale=1.0,
    )
    # When all scales=0, the linear interpolation returns original
    drift0 = (out0 - h).abs().max().item()
    print(f"[INFO] drift at scale=0.0: {drift0:.6e}")
    assert drift0 < 1e-5, f"scale=0 should be identity, drift={drift0}"

    # Test 4: scale_ll=1.0 should preserve luma exactly (full chroma match, luma unchanged)
    out1 = bridge._apply_endpoint_adain(
        h, style_latent=style_latent,
        adain_mode="per_subband_wct_ll_ycbcr",
        lowpass_levels=1, lowpass_basis="haar",
        style_extrap_alpha=0.0,
        adain_scale_ll=1.0, adain_scale_lh=0.0, adain_scale_hl=0.0, adain_scale_hh=0.0,
        endpoint_adain_scale=1.0,
    )
    y_out1 = out1.mean(dim=1, keepdim=True)
    luma_drift1 = (y_out1 - y_in).abs().max().item()
    print(f"[INFO] luma drift at scale_ll=1.0 (HF=0): {luma_drift1:.6e}")
    # Key invariant: when only LL is touched, luma should be exactly preserved
    assert luma_drift1 < 1e-5, f"luma must be preserved at scale_ll=1.0: {luma_drift1}"

    # Test 5: forward pass with new mode in T11 config
    mcfg2 = ModelConfig(
        latent_channels=4, num_styles=5, base_dim=64, time_dim=64,
        num_res_blocks=4, style_attn_num_heads=4,
        style_cross_attn_gate_init=0.05, style_attn_temperature=1.0,
        style_shortcut_alpha=1.0,
        cross_attn_dwt_route=True,
        dwt_route_train_prob=0.8,
        endpoint_adain_mode="per_subband_wct_ll_ycbcr",
        endpoint_adain_scale_ll=0.5,
        endpoint_adain_scale_lh=0.3,
        endpoint_adain_scale_hl=0.3,
        endpoint_adain_scale_hh=0.5,
    )
    mcfg2.style_extrap_alpha = 0.4
    bridge2 = WEAVE(mcfg2, bcfg)
    bridge2.eval()
    x = torch.randn(2, 4, 32, 32)
    style_id = torch.tensor([0, 1])
    t = torch.tensor([0.5, 0.5])
    with torch.no_grad():
        v = bridge2(x, t=t, style_id=style_id)
    assert "ll" in v and "lh" in v and "hl" in v, "missing velocity keys"
    for k in ("ll", "lh", "hl"):
        assert torch.isfinite(v[k]).all(), f"NaN in v_{k}"
    print("[PASS] forward pass with T26 config")

    print("\nALL SMOKE TESTS PASSED (5/5)")


if __name__ == "__main__":
    main()
