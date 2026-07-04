#!/usr/bin/env python
"""
R4-A Extended: Test larger velocity scales to find optimal value
"""

from __future__ import annotations

import sys
import torch
import numpy as np
from pathlib import Path
import json

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
sys.path.insert(0, str(_SRC_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

from config_schema import ExperimentConfig
from model import build_model_from_config
from style_families import prune_state_dict_for_tokenizer_family
from utils.training import strip_compile_prefix


def main():
    checkpoint_path = Path(r"g:\GitHub\Latent_Style\SchrodingerBridge\exp\task4_iter\r2b_with_antiwhiten\epoch_0002.pt")
    output_dir = Path(r"g:\GitHub\Latent_Style\SchrodingerBridge\exp\task4_iter\r4a_velocity_scaling")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ExperimentConfig.from_mapping(ckpt["config"])
    model = build_model_from_config(config.model, bridge_cfg=config.bridge).to(device)
    state_dict = strip_compile_prefix(ckpt["model_state_dict"])
    state_dict, _ = prune_state_dict_for_tokenizer_family(
        state_dict,
        tokenizer_family=str(getattr(config.model, "tokenizer_family", "legacy_factorized")),
        contract_family=str(getattr(config.model, "contract_family", "legacy")),
        style_injection_mode=str(getattr(config.model, "style_injection_mode", "none")),
        proximal_mode=str(getattr(config.model, "proximal_mode", "off")),
        style_delta_mode=str(getattr(config.model, "style_delta_mode", "none")),
        output_appearance_alignment_mode=str(getattr(config.model, "output_appearance_alignment_mode", "none")),
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Test extended scales
    scales = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]

    B, C, H, W = 2, 4, 32, 32

    print("="*80)
    print("R4-A EXTENDED: Finding Optimal Velocity Scale")
    print("="*80)
    print(f"{'Scale':<8} {'Vel Norm':>10} {'z1 Norm':>10} {'Alpha':>8} {'Status':>15}")
    print("-"*65)

    results = {}

    for scale in scales:
        with torch.no_grad():
            x = torch.randn(B, C, H, W, device=device)
            y = torch.randn(B, C, H, W, device=device) * 0.5
            t = torch.zeros(B, device=device)
            style_id = torch.tensor([0, 1], device=device)

            v_pred = model(x, t=t, style_id=style_id, style_latent=y, target_latent=y, velocity_scale=scale)
            z_1_hat = x + v_pred  # t=0

            v_norm = v_pred.pow(2).mean().sqrt().item()
            z1_norm = z_1_hat.pow(2).mean().sqrt().item()
            y_norm = y.pow(2).mean().sqrt().item()
            alpha = ((z_1_hat - x).pow(2).mean().sqrt() / ((y - x).pow(2).mean().sqrt() + 1e-6)).item()

            results[f"scale_{scale}"] = {
                "scale": scale,
                "velocity_norm": v_norm,
                "z1_hat_norm": z1_norm,
                "target_norm": y_norm,
                "endpoint_alpha": alpha,
            }

            # Determine status
            if alpha >= 0.95 and alpha <= 1.05:
                status = "✅ OPTIMAL"
            elif alpha > 1.05:
                status = "⚠️ OVERSHOOT"
            elif alpha >= 0.7:
                status = "🟡 GOOD"
            elif alpha >= 0.4:
                status = "🟠 PARTIAL"
            else:
                status = "❌ LOW"

            print(f"{scale:<8.1f} {v_norm:>10.4f} {z1_norm:>10.4f} {alpha:>8.4f} {status:>15}")

    # Save extended results
    ext_results_path = output_dir / "r4a_extended_scales.json"
    with open(ext_results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("ANALYSIS:")
    print("="*80)

    # Find optimal range
    optimal_scales = [s for s in results if 0.9 <= results[s]["endpoint_alpha"] <= 1.1]
    if optimal_scales:
        print(f"✅ OPTIMAL RANGE FOUND: {[results[s]['scale'] for s in optimal_scales]}")
    else:
        low = max(results.keys(), key=lambda k: results[k]["endpoint_alpha"])
        print(f"⚠️ No optimal scale in tested range")
        print(f"   Best: scale={results[low]['scale']} (alpha={results[low]['endpoint_alpha']:.4f})")
        print(f"   Recommendation: test scales {results[low]['scale']*2:.0f}-{results[low]['scale']*3:.0f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
