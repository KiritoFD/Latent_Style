#!/usr/bin/env python
"""
R4-A: Velocity Scaling Quick Test (Simplified)

直接加载模型 + 手动推理，不依赖复杂的外部eval流程
"""

from __future__ import annotations

import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import json

# Setup paths
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
    print(f"Device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ExperimentConfig.from_mapping(ckpt["config"])

    # Build model
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
    print("Model loaded successfully!")

    # Test different velocity scales
    scales = [1.0, 1.5, 2.0, 3.0]

    # Create dummy input for testing (batch of latents)
    B, C, H, W = 2, 4, 32, 32  # latent dimensions

    print("\n" + "="*70)
    print("R4-A: VELOCITY MAGNITUDE SCALING TEST")
    print("="*70)

    results = {}

    for scale in scales:
        print(f"\n--- Testing velocity_scale={scale} ---")

        with torch.no_grad():
            # Create random test input (simulating source latent z_0)
            x = torch.randn(B, C, H, W, device=device)
            y = torch.randn(B, C, H, W, device=device) * 0.5  # target latent (smaller magnitude like real data)
            t = torch.zeros(B, device=device)  # t=0 for endpoint prediction
            style_id = torch.tensor([0, 1], device=device)

            # Forward pass with velocity scaling
            v_pred = model(
                x,
                t=t,
                style_id=style_id,
                style_latent=y,
                target_latent=y,
                velocity_scale=scale,
            )

            # Compute endpoint
            z_1_hat = x + (1.0 - t.view(-1, 1, 1, 1)) * v_pred

            # Collect statistics
            v_stats = {
                "velocity_mean": v_pred.mean().item(),
                "velocity_std": v_pred.std().item(),
                "velocity_norm": v_pred.pow(2).mean().sqrt().item(),
                "z1_hat_mean": z_1_hat.mean().item(),
                "z1_hat_std": z_1_hat.std().item(),
                "z1_hat_norm": z_1_hat.pow(2).mean().sqrt().item(),
                "y_norm": y.pow(2).mean().sqrt().item(),
                "displacement_norm": (z_1_hat - x).pow(2).mean().sqrt().item(),
                "target_distance": (y - x).pow(2).mean().sqrt().item(),
                "endpoint_alpha": ((z_1_hat - x).pow(2).mean().sqrt() / ((y - x).pow(2).mean().sqrt() + 1e-6)).item(),
            }

            results[f"scale_{scale}"] = v_stats

            print(f"  Velocity: mean={v_stats['velocity_mean']:.4f}, std={v_stats['velocity_std']:.4f}, norm={v_stats['velocity_norm']:.4f}")
            print(f"  z_1_hat:  mean={v_stats['z1_hat_mean']:.4f}, std={v_stats['z1_hat_std']:.4f}, norm={v_stats['z1_hat_norm']:.4f}")
            print(f"  Target(y): norm={v_stats['y_norm']:.4f}")
            print(f"  Endpoint alpha: {v_stats['endpoint_alpha']:.4f} (1.0=reached target)")
            print(f"  Displacement / Target distance: {v_stats['displacement_norm']:.4f} / {v_stats['target_distance']:.4f}")

    # Save results
    results_path = output_dir / "r4a_velocity_scaling_test.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {results_path}")

    # Print summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"{'Scale':<8} {'Vel Norm':>10} {'z1 Norm':>10} {'Alpha':>8} {'Reached?'}")
    print("-"*50)
    for scale in scales:
        key = f"scale_{scale}"
        r = results[key]
        alpha = r["endpoint_alpha"]
        reached = "✓ YES" if alpha >= 0.95 else ("~ PARTIAL" if alpha >= 0.7 else "✗ NO")
        print(f"{scale:<8.1f} {r['velocity_norm']:>10.4f} {r['z1_hat_norm']:>10.4f} {alpha:>8.4f} {reached:>10}")

    print("\n" + "="*70)
    print("INTERPRETATION:")
    print("- If alpha increases with scale → velocity magnitude hypothesis CONFIRMED")
    print("- If alpha stays low regardless of scale → root cause is elsewhere")
    print("- Optimal scale: highest alpha without overshooting (alpha > 1.1)")
    print("="*70)


if __name__ == "__main__":
    main()
