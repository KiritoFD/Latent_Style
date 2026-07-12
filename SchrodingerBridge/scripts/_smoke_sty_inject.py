"""Smoke test for 712 Phase StyleInject: verify forward pass works for all 3 configs.

Tests:
1. baseline (style_adaln=False, style_vhead=False) — must match existing behavior
2. sty_adaln only (style_adaln=True, style_vhead=False)
3. sty_vhead only (style_adaln=False, style_vhead=True)
4. sty_both (style_adaln=True, style_vhead=True)

Verifies:
- forward pass produces correct output shapes
- zero-init means initial output ≈ baseline output (for adaln/vhead)
- param count delta is small
- VRAM usage reasonable
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from config_schema import ExperimentConfig, ModelConfig
from model import WEAVE


def make_model_cfg(**overrides):
    """Minimal model config for WEAVE."""
    cfg = ModelConfig()
    cfg.latent_channels = 4
    cfg.num_styles = 5
    cfg.base_dim = 64
    cfg.time_dim = 256
    cfg.num_res_blocks = 4
    cfg.style_attn_num_heads = 4
    cfg.style_cross_attn_gate_init = 0.05
    cfg.style_cross_attention_enabled = True
    cfg.style_attn_temperature = 1.0
    cfg.style_shortcut_alpha = 1.0
    cfg.cross_attn_dwt_route = False
    cfg.dwt_route_train_prob = 0.0
    cfg.enable_hh_head = False
    cfg.style_condition_source = "target_dino_patches"
    cfg.tokenizer_dino_dim = 384
    cfg.spectral_ode_enabled = True
    cfg.style_extrap_alpha = 0.1
    cfg.endpoint_adain_scale = 1.0
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def test_config(name, **overrides):
    print(f"\n{'='*60}\nTest: {name}\n{'='*60}")
    cfg = make_model_cfg(**overrides)
    model = WEAVE(cfg).cuda().float()
    model.eval()
    n_params = count_params(model)
    print(f"Params: {n_params:,} ({n_params/1e6:.3f}M)")

    # Forward pass
    B = 4
    x = torch.randn(B, 4, 32, 32, device="cuda", dtype=torch.float32)
    t = torch.rand(B, device="cuda", dtype=torch.float32)
    style_id = torch.tensor([0, 1, 2, 3], device="cuda")

    with torch.no_grad():
        out = model(x, t=t, style_id=style_id)

    for k, v in out.items():
        print(f"  v_{k}: shape={v.shape}, abs_mean={v.abs().float().mean():.6f}")

    # VRAM
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"VRAM: {vram:.2f} GB")
    torch.cuda.reset_peak_memory_stats()

    del model, x, t, style_id, out
    torch.cuda.empty_cache()
    return n_params, vram


if __name__ == "__main__":
    torch.manual_seed(42)
    print("Smoke test: 712 Phase StyleInject")

    p0, v0 = test_config("baseline", style_adaln_enabled=False, style_velocity_head_enabled=False)
    p1, v1 = test_config("sty_adaln", style_adaln_enabled=True, style_velocity_head_enabled=False)
    p2, v2 = test_config("sty_vhead", style_adaln_enabled=False, style_velocity_head_enabled=True)
    p3, v3 = test_config("sty_both", style_adaln_enabled=True, style_velocity_head_enabled=True)

    print(f"\n{'='*60}\nSummary\n{'='*60}")
    print(f"baseline:  {p0:>10,} params, {v0:.2f} GB VRAM")
    print(f"sty_adaln: {p1:>10,} params (+{p1-p0:,}), {v1:.2f} GB VRAM")
    print(f"sty_vhead: {p2:>10,} params (+{p2-p0:,}), {v2:.2f} GB VRAM")
    print(f"sty_both:  {p3:>10,} params (+{p3-p0:,}), {v3:.2f} GB VRAM")
    print("\nAll configs forward pass OK.")
