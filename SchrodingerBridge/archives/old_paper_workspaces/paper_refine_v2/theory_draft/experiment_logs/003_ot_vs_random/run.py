"""
Experiment 3: OT vs Random Coupling — Velocity Variance Analysis.

Goal: Test Proposition 5 (OT coupling stabilizes endpoint supervision).
Method:
  1. For each content latent, compute:
     a) OT-matched target via Sinkhorn (current training)
     b) Random-matched target (uniform assignment)
  2. Compute target velocity statistics for both:
     - ||target - content||^2 (displacement magnitude)
     - Variance of velocity direction (angular spread)
  3. Compare: does OT coupling reduce variance?
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
from probe_base import load_model_and_config, load_dataset


@torch.no_grad()
def compute_ot_vs_random_stats(
    model: torch.nn.Module,
    content: torch.Tensor,
    target_style: torch.Tensor,
    style_id_tensor: torch.Tensor,
    device: torch.device,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iters: int = 60,
) -> dict:
    """Compare OT-matched vs randomly-matched velocity statistics."""
    B = content.shape[0]

    # ---- 1. OT matching (simplified Sinkhorn) ----
    from ot_cost import SWDTransportCost
    # We need to build a minimal config for SWDTransportCost
    config = {
        "bridge": {
            "ot_cost_mode": "swd",
            "swd_distance_mode": "cdf",
            "swd_num_projections": 64,
            "swd_patch_sizes": [3, 5, 7, 15],
            "swd_cdf_num_bins": 32,
            "swd_cdf_tau": 0.01,
            "swd_cdf_sample_size": 256,
            "swd_deterministic_subsample": True,
        }
    }
    transport_cost = SWDTransportCost(config)

    # Compute pairwise cost matrix
    cost_matrix = transport_cost.pairwise_cost(content.float(), target_style.float())
    cost_matrix = cost_matrix.float()

    # Sinkhorn
    n_src, n_tgt = cost_matrix.shape
    mu = torch.full((n_src,), 1.0 / max(n_src, 1), device=device)
    nu = torch.full((n_tgt,), 1.0 / max(n_tgt, 1), device=device)
    kernel = torch.exp((-cost_matrix / sinkhorn_epsilon).clamp(min=-80, max=80))
    u = torch.ones_like(mu)
    v = torch.ones_like(nu)
    for _ in range(sinkhorn_iters):
        u = mu / (kernel @ v).clamp_min(1e-12)
        v = nu / (kernel.t() @ u).clamp_min(1e-12)
    plan = u.unsqueeze(1) * kernel * v.unsqueeze(0)
    plan = plan / plan.sum().clamp_min(1e-12)

    # Sample from OT plan
    row_probs = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-12)
    sampled_cols = torch.multinomial(row_probs, num_samples=1, replacement=True).squeeze(1)
    ot_matched = target_style.index_select(0, sampled_cols)

    # Expected cost
    ot_cost = (plan * cost_matrix).sum()

    # ---- 2. Random matching ----
    rand_indices = torch.randperm(B, device=device)
    rand_matched = target_style[rand_indices]
    rand_cost_matrix = cost_matrix[torch.arange(B), rand_indices]
    rand_cost = rand_cost_matrix.mean()

    # ---- 3. Compute target velocity statistics ----
    ot_velocity = ot_matched - content           # [B, C, H, W]
    rand_velocity = rand_matched - content

    # Velocity magnitude
    ot_vel_norm_sq = (ot_velocity ** 2).sum(dim=(1, 2, 3))
    rand_vel_norm_sq = (rand_velocity ** 2).sum(dim=(1, 2, 3))

    # Directional variance: cosine similarity to mean direction
    ot_flat = ot_velocity.view(B, -1)
    rand_flat = rand_velocity.view(B, -1)
    ot_mean_dir = ot_flat.mean(dim=0)
    rand_mean_dir = rand_flat.mean(dim=0)
    ot_mean_dir_norm = max(ot_mean_dir.norm().item(), 1e-8)
    rand_mean_dir_norm = max(rand_mean_dir.norm().item(), 1e-8)

    ot_cos_sim = (ot_flat @ ot_mean_dir) / (ot_flat.norm(dim=1) * ot_mean_dir_norm + 1e-8)
    rand_cos_sim = (rand_flat @ rand_mean_dir) / (rand_flat.norm(dim=1) * rand_mean_dir_norm + 1e-8)

    return {
        "ot": {
            "mean_displacement_sq": ot_vel_norm_sq.mean().item(),
            "std_displacement_sq": ot_vel_norm_sq.std().item(),
            "cv_displacement": (ot_vel_norm_sq.std() / ot_vel_norm_sq.mean()).item(),
            "mean_cos_sim": ot_cos_sim.mean().item(),
            "std_cos_sim": ot_cos_sim.std().item(),
            "cost": ot_cost.item(),
        },
        "random": {
            "mean_displacement_sq": rand_vel_norm_sq.mean().item(),
            "std_displacement_sq": rand_vel_norm_sq.std().item(),
            "cv_displacement": (rand_vel_norm_sq.std() / rand_vel_norm_sq.mean()).item(),
            "mean_cos_sim": rand_cos_sim.mean().item(),
            "std_cos_sim": rand_cos_sim.std().item(),
            "cost": rand_cost.item(),
        },
        "ratio_ot_to_random": {
            "displacement_var": (ot_vel_norm_sq.var() / rand_vel_norm_sq.var()).item(),
            "cost": ot_cost.item() / max(rand_cost.item(), 1e-10),
        }
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("Experiment 003: OT vs Random Coupling — Velocity Variance")
    print("=" * 70)
    print(f"Device: {device} | Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    ckpt_path = str(Path(__file__).resolve().parents[4] /
                     "ablation_destructive_7epoch/D0_full_correct_7ep/epoch_0007.pt")
    model, config = load_model_and_config(ckpt_path, device)
    dataset = load_dataset(config)

    batch_size = 64
    num_batches = 10

    all_results = []
    for batch_idx in range(num_batches):
        print(f"\n--- Batch {batch_idx + 1}/{num_batches} ---")

        # Get a balanced batch: 1 style per batch
        indices = torch.randint(0, len(dataset), (batch_size,))
        batch_list = [dataset[int(idx)] for idx in indices]
        content = torch.stack([b["content"] for b in batch_list], dim=0)
        target_style = torch.stack([b["target_style"] for b in batch_list], dim=0)
        style_ids = torch.tensor([b["target_style_id"] for b in batch_list], dtype=torch.long)

        content = content.to(device)
        target_style = target_style.to(device)
        style_ids = style_ids.to(device)

        stats = compute_ot_vs_random_stats(model, content, target_style, style_ids, device)
        all_results.append(stats)

        print(f"  OT cost:      {stats['ot']['cost']:.4f}")
        print(f"  Random cost:  {stats['random']['cost']:.4f}")
        print(f"  Cost ratio:   {stats['ratio_ot_to_random']['cost']:.4f}")
        print(f"  OT disp CV:   {stats['ot']['cv_displacement']:.4f}")
        print(f"  Random disp CV: {stats['random']['cv_displacement']:.4f}")
        print(f"  OT cos sim:   {stats['ot']['mean_cos_sim']:.4f}")
        print(f"  Random cos sim: {stats['random']['mean_cos_sim']:.4f}")

    # Aggregate
    print("\n" + "=" * 70)
    print("Summary (mean ± std across batches)")
    print("=" * 70)

    for key in ["ot", "random"]:
        print(f"\n--- {key.upper()} coupling ---")
        for metric in ["cost", "mean_displacement_sq", "std_displacement_sq",
                       "cv_displacement", "mean_cos_sim", "std_cos_sim"]:
            vals = [r[key][metric] for r in all_results]
            print(f"  {metric:25s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"\n--- Ratios ---")
    for metric in ["cost", "displacement_var"]:
        vals = [r["ratio_ot_to_random"][metric] for r in all_results]
        print(f"  {metric:25s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Save
    output_path = Path(__file__).resolve().parent / "results.json"
    with open(output_path, "w") as f:
        json.dump({
            "config": "D0_full_correct_7ep",
            "num_batches": num_batches,
            "batch_size": batch_size,
            "results": all_results,
            "summary": {
                "ot_cost_mean": float(np.mean([r["ot"]["cost"] for r in all_results])),
                "random_cost_mean": float(np.mean([r["random"]["cost"] for r in all_results])),
                "cost_ratio_mean": float(np.mean([r["ratio_ot_to_random"]["cost"] for r in all_results])),
                "ot_cv_mean": float(np.mean([r["ot"]["cv_displacement"] for r in all_results])),
                "random_cv_mean": float(np.mean([r["random"]["cv_displacement"] for r in all_results])),
            }
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
