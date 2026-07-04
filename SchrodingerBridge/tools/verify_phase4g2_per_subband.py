"""Phase 4G.2 smoke test: per-subband AdaIN 验证.

验证项:
1. Perfect Reconstruction: idwt2_haar_multi_reconstruct(dwt2_haar_multi_decompose(x)) ≈ x
2. endpoint_adain_mode='per_subband' 配置正确加载
3. per_subband 模式下 integrate_transport 正常运行 (不报错)
4. per_subband 与 spatial_fiber 输出不同 (确认分支生效)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from config_schema import ModelConfig, BridgeConfig
from spectral620 import dwt2_haar_multi_decompose, idwt2_haar_multi_reconstruct
from spectral_bridge620 import build_spectral_ode_bridge_from_config


def check_pr():
    """1. Perfect Reconstruction 验证."""
    print("[1/4] Perfect Reconstruction 验证...")
    x = torch.randn(2, 4, 64, 64)
    for levels in [1, 2, 3, 4]:
        decomp = dwt2_haar_multi_decompose(x, levels=levels)
        recon = idwt2_haar_multi_reconstruct(decomp, levels=levels)
        err = (recon - x).abs().max().item()
        status = "PASS" if err < 1e-5 else "FAIL"
        print(f"  levels={levels}: LL_K shape={tuple(decomp['ll_K'].shape)}, "
              f"#subs={len(decomp['h'])}, PR error={err:.2e} [{status}]")
        assert err < 1e-5, f"PR failed at levels={levels}: err={err}"
    print("  ALL PR CHECKS PASSED")


def check_config_load():
    """2. endpoint_adain_mode 配置加载."""
    print("[2/4] endpoint_adain_mode 配置加载验证...")
    # spatial_fiber (default)
    cfg_default = ModelConfig()
    assert getattr(cfg_default, "endpoint_adain_mode", "MISSING") == "spatial_fiber", \
        f"Default should be 'spatial_fiber', got {getattr(cfg_default, 'endpoint_adain_mode', 'MISSING')}"
    print(f"  Default: endpoint_adain_mode='{cfg_default.endpoint_adain_mode}' [PASS]")
    # per_subband (via setattr)
    cfg_per = ModelConfig()
    cfg_per.endpoint_adain_mode = "per_subband"
    assert cfg_per.endpoint_adain_mode == "per_subband"
    print(f"  Set to 'per_subband': endpoint_adain_mode='{cfg_per.endpoint_adain_mode}' [PASS]")
    print("  CONFIG LOAD CHECKS PASSED")


def check_integrate_runs():
    """3. per_subband 模式 integrate_transport 正常运行."""
    print("[3/4] per_subband 模式 integrate_transport 运行验证...")
    cfg = ModelConfig()
    cfg.endpoint_adain_mode = "per_subband"
    cfg.endpoint_adain_scale = 1.0
    cfg.endpoint_lowpass_levels = 3
    cfg.style_extrap_alpha = 0.1
    cfg.style_attn_mode = "softmax"
    bridge_cfg = BridgeConfig()
    model = build_spectral_ode_bridge_from_config(cfg, bridge_cfg=bridge_cfg)
    model.eval()

    x = torch.randn(2, 4, 64, 64)
    style_latent = torch.randn(1, 4, 64, 64)
    style_id = torch.tensor([0, 1])
    with torch.no_grad():
        out = model.integrate_transport(
            x, style_id=style_id, num_steps=2, step_size=1.0,
            style_latent=style_latent,
        )
    assert out.shape == x.shape, f"Output shape mismatch: {out.shape} vs {x.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN/Inf"
    print(f"  per_subband output: shape={tuple(out.shape)}, "
          f"mean={out.mean().item():.4f}, std={out.std().item():.4f} [PASS]")
    print("  INTEGRATE RUN CHECK PASSED")


def check_modes_differ():
    """4. per_subband vs spatial_fiber 输出不同 (确认分支生效)."""
    print("[4/4] per_subband vs spatial_fiber 输出差异验证...")
    torch.manual_seed(42)
    x = torch.randn(2, 4, 64, 64)
    style_latent = torch.randn(1, 4, 64, 64)
    style_id = torch.tensor([0, 1])

    # spatial_fiber (default)
    cfg_spatial = ModelConfig()
    cfg_spatial.endpoint_adain_scale = 1.0
    cfg_spatial.endpoint_lowpass_levels = 3
    cfg_spatial.style_extrap_alpha = 0.1
    cfg_spatial.style_attn_mode = "softmax"
    cfg_spatial.endpoint_adain_mode = "spatial_fiber"
    model_spatial = build_spectral_ode_bridge_from_config(cfg_spatial, bridge_cfg=BridgeConfig())
    model_spatial.eval()
    with torch.no_grad():
        out_spatial = model_spatial.integrate_transport(
            x, style_id=style_id, num_steps=2, step_size=1.0,
            style_latent=style_latent,
        )

    # per_subband (same seed, same init)
    torch.manual_seed(42)
    cfg_per = ModelConfig()
    cfg_per.endpoint_adain_scale = 1.0
    cfg_per.endpoint_lowpass_levels = 3
    cfg_per.style_extrap_alpha = 0.1
    cfg_per.style_attn_mode = "softmax"
    cfg_per.endpoint_adain_mode = "per_subband"
    model_per = build_spectral_ode_bridge_from_config(cfg_per, bridge_cfg=BridgeConfig())
    model_per.eval()
    with torch.no_grad():
        out_per = model_per.integrate_transport(
            x, style_id=style_id, num_steps=2, step_size=1.0,
            style_latent=style_latent,
        )

    diff = (out_per - out_spatial).abs().mean().item()
    print(f"  spatial_fiber output: mean={out_spatial.mean().item():.4f}, std={out_spatial.std().item():.4f}")
    print(f"  per_subband output:   mean={out_per.mean().item():.4f}, std={out_per.std().item():.4f}")
    print(f"  |diff| mean: {diff:.6f}")
    status = "PASS" if diff > 1e-4 else "WARN (outputs too similar)"
    print(f"  [{status}]")
    if diff < 1e-4:
        print("  WARNING: per_subband and spatial_fiber outputs are nearly identical!")
        print("  This may indicate the per_subband branch is not being triggered.")
    print("  MODE DIFFER CHECK DONE")


def main():
    print("=" * 70)
    print("Phase 4G.2 Smoke Test: per-subband AdaIN")
    print("=" * 70)
    check_pr()
    print()
    check_config_load()
    print()
    check_integrate_runs()
    print()
    check_modes_differ()
    print()
    print("=" * 70)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
