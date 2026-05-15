"""
Experiment 4: Trajectory Straightness Analysis.

Goal: Characterize the straightness of learned transport trajectories.
This directly validates the "learned transport path" interpretation.

Metrics:
  1. Path length ratio (PLR):
     PLR = ||z_K - z_0|| / Σ_k ||z_{k+1} - z_k||
     PLR = 1 → perfectly straight line
     PLR < 1 → curved/spiraling trajectory

  2. Directional consistency (DC):
     DC = (1/(K-1)) Σ_k cos(v_k, v_{k+1})
     DC = 1 → all steps in same direction
     DC = 0 → random walk

  3. Total curvature proxy (TCP):
     TCP = (1/(K-1)) Σ_k ||v_k/||v_k|| - v_{k+1}/||v_{k+1}||||

  4. Energy concentration:
     What fraction of the total displacement comes from
     the first half vs second half of the trajectory?

Also compare across checkpoints:
  - D0 (full control) = mainline OT-trained
  - D2 (no kinetic) = what happens without kinetic regularization
  - D1 (no terminal SWD) = what happens without endpoint distribution control
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
def compute_trajectory_metrics(
    model: torch.nn.Module,
    x: torch.Tensor,
    style_id: torch.Tensor,
    num_steps: int = 256,
    horizon: float = 1.0,
) -> dict:
    """
    Integrate and compute straightness metrics.
    Returns dict of per-sample metrics.
    """
    B = x.shape[0]
    steps = max(2, int(num_steps))
    dt = horizon / float(steps)

    # Store all trajectory points and velocities
    h = x.clone()
    points = [h.clone()]
    velocities = []

    for idx in range(steps):
        t_val = horizon * ((idx + 0.5) / float(steps))
        t_tensor = torch.full((B,), t_val, device=x.device, dtype=x.dtype)
        v = model.forward(h, t=t_tensor, style_id=style_id)
        velocities.append(v.clone())
        h = h + v * dt
        points.append(h.clone())

    # Stack: [B, K+1, C, H, W] and [B, K, C, H, W]
    points = torch.stack(points, dim=1)   # B x (K+1) x C x H x W
    velocities = torch.stack(velocities, dim=1)  # B x K x C x H x W

    # Displacement vectors at each step
    step_disps = velocities * dt  # B x K x C x H x W

    # 1. Path length ratio
    total_disp = torch.linalg.vector_norm(points[:, -1] - x, dim=(1, 2, 3))  # [B]
    path_length = torch.linalg.vector_norm(step_disps, dim=(2, 3, 4)).sum(dim=1)  # [B]
    plr = total_disp / path_length.clamp(min=1e-8)

    # 2. Directional consistency (cosine similarity between consecutive steps)
    v_normed = velocities / (torch.linalg.vector_norm(velocities, dim=(2, 3, 4), keepdim=True) + 1e-8)
    v_flat = v_normed.view(B, steps, -1)  # B x K x (C*H*W)
    cos_sim = (v_flat[:, :-1] * v_flat[:, 1:]).sum(dim=2)  # B x (K-1)
    dc = cos_sim.mean(dim=1)  # [B]

    # 3. Curvature proxy: angular change between consecutive normalized velocities
    angular_diff = torch.linalg.vector_norm(
        v_flat[:, :-1] - v_flat[:, 1:], dim=2
    )  # B x (K-1)
    tcp = angular_diff.mean(dim=1)  # [B]

    # 4. Energy concentration: fraction of displacement in first half vs second half
    half_k = steps // 2
    first_half_disp = torch.linalg.vector_norm(
        points[:, half_k] - points[:, 0], dim=(1, 2, 3)
    )
    second_half_disp = torch.linalg.vector_norm(
        points[:, -1] - points[:, half_k], dim=(1, 2, 3)
    )
    total = (first_half_disp + second_half_disp).clamp(min=1e-8)
    energy_ratio = first_half_disp / total  # > 0.5 means more change in first half

    # 5. Maximum deviation from straight line (normalized)
    # For each sample, find the point furthest from the chord
    chord_dir = points[:, -1] - points[:, 0]  # B x C x H x W
    chord_norm = torch.linalg.vector_norm(chord_dir, dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
    chord_unit = chord_dir / chord_norm

    # For each intermediate point, compute distance from chord
    # z_proj = z_0 + ((z_t - z_0) · chord_unit) * chord_unit
    z_0 = points[:, 0:1]  # B x 1 x C x H x W
    chord_unit_exp = chord_unit.unsqueeze(1)  # B x 1 x C x H x W
    offsets = points[:, 1:-1] - z_0  # B x (K-1) x C x H x W
    proj_lengths = (offsets * chord_unit_exp).sum(dim=(2, 3, 4), keepdim=True)  # B x (K-1) x 1

    proj = z_0 + proj_lengths * chord_unit_exp
    dev = torch.linalg.vector_norm(points[:, 1:-1] - proj, dim=(2, 3, 4))  # B x (K-1)
    max_dev = dev.max(dim=1).values  # [B]
    norm_max_dev = max_dev / chord_norm.squeeze(1).clamp(min=1e-8)

    return {
        "path_length_ratio": plr.cpu().numpy().tolist(),
        "directional_consistency": dc.cpu().numpy().tolist(),
        "curvature_proxy": tcp.cpu().numpy().tolist(),
        "energy_ratio_first_half": energy_ratio.cpu().numpy().tolist(),
        "max_normalized_deviation": norm_max_dev.cpu().numpy().tolist(),
        "total_disp_norm": total_disp.cpu().numpy().tolist(),
        "path_length": path_length.cpu().numpy().tolist(),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("Experiment 004: Trajectory Straightness Analysis")
    print("=" * 70)
    print(f"Device: {device} | Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Models to compare
    experiments = [
        ("D0 (full control)", str(Path(__file__).resolve().parents[4] /
            "ablation_destructive_7epoch/D0_full_correct_7ep/epoch_0007.pt")),
        ("D1 (no terminal SWD)", str(Path(__file__).resolve().parents[4] /
            "ablation_destructive_7epoch/D1_no_terminal_swd/epoch_0007.pt")),
        ("D2 (no kinetic)", str(Path(__file__).resolve().parents[4] /
            "ablation_destructive_7epoch/D2_no_kinetic/epoch_0007.pt")),
    ]

    num_steps = 128  # good balance of accuracy and speed
    batch_size = 32
    num_batches = 5

    all_results = {}

    for label, ckpt_path in experiments:
        print(f"\n{'='*70}")
        print(f"Model: {label}")
        print(f"{'='*70}")
        model, config = load_model_and_config(ckpt_path, device)
        dataset = load_dataset(config)

        batch_metrics = defaultdict(list)

        for batch_idx in range(num_batches):
            batch = get_batch(dataset, batch_size)
            content = batch["content"].to(device)
            target_style_id = batch["target_style_id"].to(device)

            metrics = compute_trajectory_metrics(
                model, content, target_style_id, num_steps=num_steps
            )

            for key in metrics:
                batch_metrics[key].extend(metrics[key])

            print(f"  Batch {batch_idx+1}/{num_batches}: "
                  f"PLR={np.mean(metrics['path_length_ratio']):.4f} "
                  f"DC={np.mean(metrics['directional_consistency']):.4f} "
                  f"Dev={np.mean(metrics['max_normalized_deviation']):.4f}")

        # Aggregate
        summary = {}
        print(f"\n  === Summary ({label}) ===")
        print(f"  {'Metric':<35s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
        print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        for key in ["path_length_ratio", "directional_consistency",
                     "curvature_proxy", "energy_ratio_first_half",
                     "max_normalized_deviation"]:
            vals = np.array(batch_metrics[key])
            summary[key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "p5": float(np.percentile(vals, 5)),
                "p25": float(np.percentile(vals, 25)),
                "p50": float(np.percentile(vals, 50)),
                "p75": float(np.percentile(vals, 75)),
                "p95": float(np.percentile(vals, 95)),
            }
            print(f"  {key:<35s} {summary[key]['mean']:>10.4f} {summary[key]['std']:>10.4f} "
                  f"{summary[key]['min']:>10.4f} {summary[key]['max']:>10.4f}")

        all_results[label] = {
            "summary": summary,
            "raw_metrics": {k: v for k, v in batch_metrics.items()},
        }

    # Side-by-side comparison
    print("\n" + "=" * 70)
    print("Cross-Model Comparison")
    print("=" * 70)
    print(f"\n{'Metric':<35s}", end="")
    for label, _, in experiments:
        print(f" {label:>20s}", end="")
    print()

    for metric in ["path_length_ratio", "directional_consistency",
                   "max_normalized_deviation", "energy_ratio_first_half"]:
        print(f"{metric:<35s}", end="")
        for label, _, in experiments:
            val = all_results[label]["summary"][metric]["mean"]
            print(f" {val:>20.4f}", end="")
        print()

    # Save
    output = {
        "config": "D0/D1/D2 comparison",
        "num_steps": num_steps,
        "batch_size": batch_size,
        "num_batches": num_batches,
        # Only save summaries (raw data can be large)
        "results": {
            label: data["summary"]
            for label, data in all_results.items()
        }
    }
    output_path = Path(__file__).resolve().parent / "results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
