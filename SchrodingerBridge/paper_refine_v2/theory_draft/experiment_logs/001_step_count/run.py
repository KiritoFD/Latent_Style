"""
Experiment 1: Step-count endpoint error analysis.

Goal: Verify Proposition 3 (Euler discretization error bound O(dt)).
Method: For K in {1, 2, 4, 8, 12, 16, 32, 64, 128, 256} steps:
  1. Compute endpoint z_K via Euler integration with K steps.
  2. Use z_256 as reference ground truth.
  3. Compute ||z_K - z_256||_2 and ||z_K - z_256||_2 / ||z_256 - z_0||_2.
  4. Verify O(1/K) scaling.

Also measures:
  - Velocity norms ||v(z_k, t_k)|| at each step for K=256 (for Experiment 2)
  - Per-step velocity and path energy accumulation
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Add probe_base to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from probe_base import load_model_and_config, load_dataset, get_batch


@torch.no_grad()
def integrate_steps(
    model: torch.nn.Module,
    x: torch.Tensor,
    style_id: torch.Tensor,
    num_steps: int,
    horizon: float = 1.0,
    record_velocities: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Euler integration with K steps.
    Returns (endpoint, velocities tensor if record_velocities else None).
    """
    steps = max(1, int(num_steps))
    dt = horizon / float(steps)
    h = x.clone()
    velocities = []

    for idx in range(steps):
        t_val = horizon * ((idx + 0.5) / float(steps))
        t_tensor = torch.full((x.shape[0],), t_val, device=x.device, dtype=x.dtype)
        v = model.forward(h, t=t_tensor, style_id=style_id)
        if record_velocities:
            velocities.append(v.clone())
        h = h + v * dt

    if record_velocities and velocities:
        return h, torch.stack(velocities, dim=1)  # [B, K, C, H, W]
    return h, None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("Experiment 001: Step-count endpoint error analysis")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load model
    ckpt_path = str(Path(__file__).resolve().parents[4] /
                     "ablation_destructive_7epoch/D0_full_correct_7ep/epoch_0007.pt")
    model, config = load_model_and_config(ckpt_path, device)

    # Load dataset
    dataset = load_dataset(config)

    # Parameters
    batch_size = 32
    step_counts = [1, 2, 4, 8, 12, 16, 32, 64, 128, 256]
    ref_steps = 256
    num_batches = 5  # 5 batches x 32 = 160 samples

    results = {
        "config": "D0_full_correct_7ep",
        "num_batches": num_batches,
        "batch_size": batch_size,
        "step_counts": step_counts,
        "ref_steps": ref_steps,
        "batches": [],
    }

    for batch_idx in range(num_batches):
        print(f"\n--- Batch {batch_idx + 1}/{num_batches} ---")
        batch = get_batch(dataset, batch_size)
        content = batch["content"].to(device)
        target_style = batch["target_style"].to(device)
        target_style_id = batch["target_style_id"].to(device)

        # Reference: 256-step integration
        t0 = time.time()
        z_ref, vel_ref = integrate_steps(model, content, target_style_id, ref_steps, record_velocities=True)
        ref_time = time.time() - t0
        print(f"  Reference ({ref_steps} steps): {ref_time:.2f}s")
        print(f"  z_ref shape: {z_ref.shape}")

        # Compute reference displacement norm for relative error
        ref_disp_norm = torch.linalg.vector_norm(z_ref - content, dim=(1, 2, 3)).mean().item()

        batch_record = {
            "content_mean": content.mean().item(),
            "content_std": content.std().item(),
            "ref_disp_norm": ref_disp_norm,
            "steps": {},
        }

        # Test each step count
        for K in step_counts:
            if K == ref_steps:
                error = torch.linalg.vector_norm(z_ref - z_ref, dim=(1, 2, 3)).mean().item()
                rel_error = 0.0
                time_taken = 0.0
            else:
                t0 = time.time()
                z_K, _ = integrate_steps(model, content, target_style_id, K, record_velocities=False)
                time_taken = time.time() - t0
                error = torch.linalg.vector_norm(z_K - z_ref, dim=(1, 2, 3)).mean().item()
                rel_error = error / max(ref_disp_norm, 1e-8)

            print(f"  K={K:3d}: error={error:.6f}, rel_error={rel_error:.6f}, time={time_taken:.3f}s")
            batch_record["steps"][str(K)] = {
                "abs_error": error,
                "rel_error": rel_error,
                "time_sec": time_taken,
            }

        # Also record per-step velocities for t-dependence analysis (from ref)
        vel_norms = torch.linalg.vector_norm(vel_ref, dim=(2, 3, 4)).mean(dim=0).cpu().numpy().tolist()  # [K]
        batch_record["velocity_norms_per_step"] = vel_norms
        batch_record["t_values"] = [(0.5 + idx) / ref_steps for idx in range(ref_steps)]

        results["batches"].append(batch_record)

    # Save results
    output_path = Path(__file__).resolve().parent / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Summary: Average across batches")
    print("=" * 70)
    # Collect errors across batches
    from collections import defaultdict
    avg_errors = defaultdict(list)
    for b in results["batches"]:
        for K_str, data in b["steps"].items():
            avg_errors[K_str].append(data["abs_error"])

    print(f"{'Steps':>6} | {'Abs Error':>12} | {'Rel to Disp':>12} | {'Scaling':>10}")
    print("-" * 46)
    ref_err = np.mean(avg_errors[str(max(step_counts))])
    print(f"{'K':>6} | {'mean±std':>12} | {'% of disp':>12} | {'Err(K)/Err(2K)':>10}")
    print("-" * 46)
    for K in step_counts:
        vals = avg_errors[str(K)]
        mean_err = np.mean(vals)
        std_err = np.std(vals)
        # Relative to displacement
        disp_vals = [b["ref_disp_norm"] for b in results["batches"]]
        mean_disp = np.mean(disp_vals)
        rel_disp = mean_err / max(mean_disp, 1e-10)
        # Scaling: compare to next (coarser) step count
        ratio_str = f"{mean_err:.4f}±{std_err:.4f}"
        rel_str = f"{rel_disp*100:.2f}%"
        # Ratio vs previous
        prev_K = max(sk for sk in step_counts if sk < K and sk > 0) if K > 1 else None
        if prev_K is not None:
            prev_err = np.mean(avg_errors[str(prev_K)])
            scaling = prev_err / max(mean_err, 1e-10)
            scaling_str = f"1/{scaling:.1f}"
        else:
            scaling_str = "-"
        print(f"{K:6d} | {ratio_str:>14} | {rel_str:>12} | {scaling_str:>10}")
    print("=" * 70)


if __name__ == "__main__":
    main()
