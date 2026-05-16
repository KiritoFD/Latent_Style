"""
Experiment 2: Velocity field t-dependence analysis.

Goal: Test the t-independence assumption of Proposition 2.
Specifically, test whether E[||v_θ(z, t, s)||^2] is approximately constant in t.

We evaluate two scenarios:
  A. Same input z_0, vary t: v_θ(z_0, t, s) for t ∈ [0, 1]
     → Tests whether the learned velocity field depends on t for a fixed input.
  B. Bridge states z_t at each t: v_θ(z_t, t, s) where z_t = (1-t)z_0 + t*z̃_1 + noise
     → Tests the actual training-relevant quantity (velocity at the bridge state).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from probe_base import load_model_and_config, load_dataset, get_batch


@torch.no_grad()
def evaluate_velocity_fixed_input(
    model: torch.nn.Module,
    x: torch.Tensor,
    style_id: torch.Tensor,
    t_values: list[float],
) -> dict:
    """
    Evaluate ||v_θ(x, t, s)||^2 for the SAME input x at multiple t values.
    Returns dict of statistics per t.
    """
    results = {}
    for t in t_values:
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        v = model.forward(x, t=t_tensor, style_id=style_id)
        sq_norm = (v ** 2).sum(dim=(1, 2, 3))  # [B]
        results[f"t_{t:.3f}"] = {
            "mean": sq_norm.mean().item(),
            "std": sq_norm.std().item(),
            "min": sq_norm.min().item(),
            "max": sq_norm.max().item(),
            "median": sq_norm.median().item(),
        }
    return results


@torch.no_grad()
def evaluate_velocity_bridge_states(
    model: torch.nn.Module,
    content: torch.Tensor,
    matched_target: torch.Tensor,
    style_id: torch.Tensor,
    t_values: list[float],
    bridge_sigma: float = 0.0,
) -> dict:
    """
    Evaluate ||v_θ(z_t, t, s)||^2 where z_t follows the bridge process.
    z_t = (1-t) * content + t * matched_target + σ * √(t(1-t)) * ε
    This is the actual training scenario.
    """
    results = {}
    for t in t_values:
        t4 = t
        base = (1.0 - t4) * content + t4 * matched_target
        if bridge_sigma > 0 and t > 0 and t < 1:
            bridge_var = t * (1.0 - t)
            bridge_std = np.sqrt(max(bridge_var, 1e-8))
            noise = torch.randn_like(content)
            z_t = base + bridge_sigma * bridge_std * noise
        else:
            z_t = base

        t_tensor = torch.full((content.shape[0],), t, device=content.device, dtype=content.dtype)
        v = model.forward(z_t, t=t_tensor, style_id=style_id)
        sq_norm = (v ** 2).sum(dim=(1, 2, 3))
        results[f"t_{t:.3f}"] = {
            "mean": sq_norm.mean().item(),
            "std": sq_norm.std().item(),
            "min": sq_norm.min().item(),
            "max": sq_norm.max().item(),
            "median": sq_norm.median().item(),
        }
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("Experiment 002: Velocity field t-dependence analysis")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load model - use D0 (full control)
    ckpt_path = str(Path(__file__).resolve().parents[4] /
                     "ablation_destructive_7epoch/D0_full_correct_7ep/epoch_0007.pt")
    model, config = load_model_and_config(ckpt_path, device)

    # Load dataset
    dataset = load_dataset(config)

    # Parameters
    batch_size = 64
    num_batches = 5
    t_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bridge_sigma = 0.05  # typical value

    all_results_fixed = []
    all_results_bridge = []

    for batch_idx in range(num_batches):
        print(f"\n--- Batch {batch_idx + 1}/{num_batches} ---")
        batch = get_batch(dataset, batch_size)
        content = batch["content"].to(device)
        target_style = batch["target_style"].to(device)
        target_style_id = batch["target_style_id"].to(device)

        # Scenario A: Fixed input, vary t
        res_fixed = evaluate_velocity_fixed_input(model, content, target_style_id, t_values)
        all_results_fixed.append(res_fixed)

        # Scenario B: Bridge states z_t
        res_bridge = evaluate_velocity_bridge_states(
            model, content, target_style, target_style_id, t_values, bridge_sigma
        )
        all_results_bridge.append(res_bridge)

        t_norms_fixed = [res_fixed[f"t_{t:.3f}"]["mean"] for t in t_values]
        t_norms_bridge = [res_bridge[f"t_{t:.3f}"]["mean"] for t in t_values]
        print(f"  Fixed input: t=0.0: {t_norms_fixed[0]:.4f}  t=0.5: {t_norms_fixed[5]:.4f}  t=1.0: {t_norms_fixed[10]:.4f}")
        print(f"  Bridge:      t=0.0: {t_norms_bridge[0]:.4f}  t=0.5: {t_norms_bridge[5]:.4f}  t=1.0: {t_norms_bridge[10]:.4f}")

    # Aggregate results
    print("\n" + "=" * 70)
    print("Aggregated Results (mean ± std across batches)")
    print("=" * 70)
    print(f"\nScenario A: Fixed input z_0, v_theta(z_0, t, s)")
    print(f"{'t':>6} | {'Mean|v|^2':>14} | {'Std':>10} | {'CV':>8}")
    print("-" * 42)
    for t in t_values:
        vals = [r[f"t_{t:.3f}"]["mean"] for r in all_results_fixed]
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        cv = std_v / max(mean_v, 1e-10)
        print(f"{t:6.1f} | {mean_v:14.6f} | {std_v:10.6f} | {cv:8.4f}")

    print(f"\nScenario B: Bridge states z_t, v_theta(z_t, t, s)")
    print(f"{'t':>6} | {'Mean|v|^2':>14} | {'Std':>10} | {'CV':>8}")
    print("-" * 42)
    for t in t_values:
        vals = [r[f"t_{t:.3f}"]["mean"] for r in all_results_bridge]
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        cv = std_v / max(mean_v, 1e-10)
        print(f"{t:6.1f} | {mean_v:14.6f} | {std_v:10.6f} | {cv:8.4f}")

    # T-independence test: variance of mean across t
    fixed_means = [np.mean([r[f"t_{t:.3f}"]["mean"] for r in all_results_fixed]) for t in t_values]
    bridge_means = [np.mean([r[f"t_{t:.3f}"]["mean"] for r in all_results_bridge]) for t in t_values]
    fixed_t_var = np.var(fixed_means)
    bridge_t_var = np.var(bridge_means)
    print(f"\n--- t-independence test ---")
    print(f"  Var[E[|v|^2 | t]] (fixed input): {fixed_t_var:.6f}")
    print(f"  Var[E[|v|^2 | t]] (bridge):      {bridge_t_var:.6f}")
    print(f"  Mean |v|^2 (fixed input):        {np.mean(fixed_means):.6f}")
    print(f"  Mean |v|^2 (bridge):             {np.mean(bridge_means):.6f}")
    print(f"  Ratio (var/mean) fixed:          {fixed_t_var / max(np.mean(fixed_means), 1e-10):.6f}")
    print(f"  Ratio (var/mean) bridge:         {bridge_t_var / max(np.mean(bridge_means), 1e-10):.6f}")

    # Save results
    output = {
        "config": "D0_full_correct_7ep",
        "bridge_sigma": bridge_sigma,
        "t_values": t_values,
        "fixed_input": all_results_fixed,
        "bridge_states": all_results_bridge,
        "summary": {
            "fixed_means_per_t": fixed_means,
            "bridge_means_per_t": bridge_means,
            "fixed_t_variance": fixed_t_var,
            "bridge_t_variance": bridge_t_var,
            "fixed_overall_mean": np.mean(fixed_means),
            "bridge_overall_mean": np.mean(bridge_means),
        }
    }
    output_path = Path(__file__).resolve().parent / "results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
