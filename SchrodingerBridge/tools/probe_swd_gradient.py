#!/usr/bin/env python3
"""Probe the SWD gradient, sorting stability, and loss landscape of a 620 bridge model.

Usage:
    python tools/probe_swd_gradient.py \
        --checkpoint exp/620_spatial_bridge/.../epoch_0008.pt \
        --config exp/620_spatial_bridge/.../config.json \
        --output-dir 620/fog/gradient_probe/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

# Ensure the src/ directory is on sys.path so we can import modules directly.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config_schema import ExperimentConfig, load_experiment_config  # noqa: E402
from model620 import SpatialBridge620, build_spatial_bridge620_from_config  # noqa: E402
from losses620 import SpatialBridgeObjective620, _sliced_wasserstein  # noqa: E402
from utils.dataset import AdaCUTLatentDataset  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _load_checkpoint_model(checkpoint_path: str, config: ExperimentConfig, device: torch.device) -> SpatialBridge620:
    """Load the model from a checkpoint file."""
    model = build_spatial_bridge620_from_config(config.model, bridge_cfg=config.bridge)
    model.to(device)
    model.eval()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    logger.info("Loaded checkpoint %s", checkpoint_path)
    return model


def _load_batch(dataset: AdaCUTLatentDataset, device: torch.device) -> dict[str, Any]:
    """Load a single batch from the dataset."""
    dataset.set_epoch(0)

    # Collect a small batch manually.
    items = [dataset[i] for i in range(min(8, len(dataset)))]
    batch: dict[str, Any] = {}
    for key in items[0].keys():
        values = [item[key] for item in items]
        if torch.is_tensor(values[0]):
            batch[key] = torch.stack(values).to(device)
        elif isinstance(values[0], int):
            batch[key] = torch.tensor(values, dtype=torch.long, device=device)
        else:
            batch[key] = values
    return batch


def compute_swd_gradient(
    model: SpatialBridge620,
    loss_obj: SpatialBridgeObjective620,
    content: torch.Tensor,
    target_style: torch.Tensor,
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Compute the gradient of SWD w.r.t. velocity at v=0 (the trivial solution).

    Returns a dict with gradient magnitude, direction stats, and per-sample norms.
    """
    # Work with a copy that requires grad.
    x = content.detach().clone().requires_grad_(True)
    bsz = x.shape[0]

    # Compute the projected target (same as training).
    projected_target, _ = loss_obj._project_training_target(x, target_style)

    # At v=0, z_1 = x + (1-t)*v = x  (with t=0).
    z_1 = x  # v=0 -> z_1 = x

    # Compute SWD between z_1 and projected target.
    projection_dirs = loss_obj._projection_dirs(z_1)
    swd_value = _sliced_wasserstein(z_1, projected_target, dirs=projection_dirs)

    # Backward to get ∇_x SWD.
    grad_x = torch.autograd.grad(swd_value, x, create_graph=False, retain_graph=False)[0]

    if grad_x is None:
        raise RuntimeError("SWD gradient is None — check requires_grad.")

    # The gradient of SWD w.r.t. velocity v is (1-t) * ∇_x SWD.
    # At t=0, (1-t) = 1, so grad_v = grad_x.
    grad_v = grad_x  # (1-t) * grad_x with t=0

    grad_norm = grad_v.norm(p=2).item()
    grad_norm_per_sample = grad_v.view(bsz, -1).norm(p=2, dim=1)

    grad_flat = grad_v.view(bsz, -1)
    grad_mean = grad_flat.mean().item()
    grad_std = grad_flat.std().item()

    # Direction: cosine similarity between gradient and v_target direction.
    v_target = (projected_target - x).detach()
    v_target_flat = v_target.view(bsz, -1)
    cos_sim = (grad_flat * v_target_flat).sum(dim=1) / (
        grad_flat.norm(p=2, dim=1) * v_target_flat.norm(p=2, dim=1) + 1e-8
    )

    return {
        "swd_at_v0": float(swd_value.item()),
        "grad_v_norm": float(grad_norm),
        "grad_v_norm_per_sample": [float(v) for v in grad_norm_per_sample.tolist()],
        "grad_v_mean": float(grad_mean),
        "grad_v_std": float(grad_std),
        "grad_v_cos_sim_with_vtarget": [float(v) for v in cos_sim.tolist()],
        "grad_v_cos_sim_mean": float(cos_sim.mean().item()),
    }


def measure_sorting_stability(
    model: SpatialBridge620,
    loss_obj: SpatialBridgeObjective620,
    content: torch.Tensor,
    target_style: torch.Tensor,
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Measure how stable SWD projection sorting is under velocity perturbations.

    For each perturbation level ε, we perturb the velocity and measure what fraction
    of elements change position in the sorted order of SWD projections.
    """
    bsz = content.shape[0]
    projected_target, _ = loss_obj._project_training_target(content, target_style)

    # Get one projection direction.
    projection_dirs = loss_obj._projection_dirs(content)
    one_dir = projection_dirs[0:1]  # [1, dim]

    # Base: v=0, z_1 = content.
    flat_x = content.detach().float().reshape(bsz, -1)
    base_proj = flat_x @ one_dir.t()  # [bsz, 1]
    base_sorted, base_indices = torch.sort(base_proj.squeeze(), dim=0)

    epsilons = [1e-5, 1e-4, 1e-3, 1e-2]
    results: dict[str, Any] = {"epsilons": epsilons, "fraction_changed": []}

    v_target = (projected_target - content).detach()
    # Normalize v_target to unit norm per sample.
    v_target_norm = v_target.view(bsz, -1)
    v_target_norm = v_target_norm / (v_target_norm.norm(p=2, dim=1, keepdim=True) + 1e-8)
    v_target_unit = v_target_norm.view_as(v_target)

    for eps in epsilons:
        eps_tensor = torch.tensor(eps, device=device, dtype=content.dtype)
        v_perturbed = eps_tensor * v_target_unit
        z_1_perturbed = content + v_perturbed
        flat_perturbed = z_1_perturbed.detach().float().reshape(bsz, -1)
        proj_perturbed = flat_perturbed @ one_dir.t()
        _, perturbed_indices = torch.sort(proj_perturbed.squeeze(), dim=0)

        changed = (base_indices != perturbed_indices).float().mean().item()
        results["fraction_changed"].append(float(changed))

    return results


def scan_loss_landscape(
    model: SpatialBridge620,
    loss_obj: SpatialBridgeObjective620,
    content: torch.Tensor,
    target_style: torch.Tensor,
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """1D scan of the loss landscape along the v_target direction.

    For s in [0, 0.05, 0.1, ..., 2.0]:
        v = s * v_target
        z_1 = content + v  (t=0)
        FM_loss = ||v - v_target||²
        SWD_loss = SWD(z_1, target)
        total = α * FM + β * SWD
    """
    bsz = content.shape[0]
    projected_target, _ = loss_obj._project_training_target(content, target_style)

    v_target = (projected_target - content).detach()
    projection_dirs = loss_obj._projection_dirs(content)

    fm_weight = loss_obj.fm_weight
    swd_weight = loss_obj.single_step_swd_weight

    s_values = [round(s, 3) for s in torch.arange(0.0, 2.05, 0.05).tolist()]

    scan_data: list[dict[str, float]] = []

    for s in s_values:
        s_tensor = torch.tensor(s, device=device, dtype=content.dtype)
        v = s_tensor * v_target
        z_1 = content + v  # t=0 -> z_1 = content + v

        fm_loss = (v - v_target).pow(2).mean().item()
        swd_loss = _sliced_wasserstein(z_1, projected_target, dirs=projection_dirs).item()
        total_loss = fm_weight * fm_loss + swd_weight * swd_loss

        scan_data.append({
            "s": float(s),
            "fm_loss": float(fm_loss),
            "swd_loss": float(swd_loss),
            "total_loss": float(total_loss),
        })

    return {
        "fm_weight": float(fm_weight),
        "swd_weight": float(swd_weight),
        "scan": scan_data,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe SWD gradient, sorting stability, and loss landscape."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="620/fog/gradient_probe",
        help="Output directory for results (default: 620/fog/gradient_probe/)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of samples to load (default: 8)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # 1. Load config and model.
    config = load_experiment_config(args.config)
    logger.info("Loaded config from %s", args.config)

    model = _load_checkpoint_model(args.checkpoint, config, device)
    logger.info("Model loaded: %s", type(model).__name__)

    # 2. Create loss object.
    loss_obj = SpatialBridgeObjective620(config)

    # 3. Load dataset and a small batch.
    data_cfg = config.data
    dataset = AdaCUTLatentDataset(
        data_root=data_cfg.data_root,
        style_subdirs=data_cfg.style_subdirs,
        allow_hflip=bool(data_cfg.allow_hflip),
        identity_ratio=data_cfg.identity_ratio,
        batch_size_hint=args.batch_size,
        balance_target_styles_per_batch=bool(data_cfg.balance_target_styles_per_batch),
        preload_to_gpu=False,
        virtual_length_multiplier=float(data_cfg.virtual_length_multiplier),
        content_style_sampling_weights=data_cfg.content_style_sampling_weights,
        target_style_sampling_weights=data_cfg.target_style_sampling_weights,
        pairing_cache_path=data_cfg.pairing_cache_path,
        pairing_cache_topk=int(data_cfg.pairing_cache_topk),
        pairing_cache_active_topk=int(data_cfg.pairing_cache_active_topk),
        pairing_cache_sample_mode=str(data_cfg.pairing_cache_sample_mode),
        pairing_cache_rank_schedule=str(data_cfg.pairing_cache_rank_schedule),
        pairing_cache_min_topk=int(data_cfg.pairing_cache_min_topk),
        pairing_cache_curriculum_epochs=int(data_cfg.pairing_cache_curriculum_epochs),
        pairing_cache_rank_power=float(data_cfg.pairing_cache_rank_power),
        pairing_cache_explore_prob=float(data_cfg.pairing_cache_explore_prob),
        pairing_cache_explore_topk=int(data_cfg.pairing_cache_explore_topk),
        pairing_cache_dual_target_mix=float(data_cfg.pairing_cache_dual_target_mix),
        pairing_cache_dual_target_topk=int(data_cfg.pairing_cache_dual_target_topk),
        pairing_cache_aux_target_topk=int(data_cfg.pairing_cache_aux_target_topk),
        pairing_cache_cross_only=bool(data_cfg.pairing_cache_cross_only),
        latent_cache_mode=str(data_cfg.latent_cache_mode),
        latent_cache_dir=str(data_cfg.latent_cache_dir),
        dino_cache_path=data_cfg.dino_cache_path,
        dino_cache_required=bool(data_cfg.dino_cache_required),
        dino_bank_limit_per_style=int(data_cfg.dino_bank_limit_per_style),
        style_caption_path=str(getattr(data_cfg, "style_caption_path", "")),
        device=str(device),
    )

    batch = _load_batch(dataset, device)
    content = batch["content"]
    target_style = batch["target_style"]
    logger.info(
        "Loaded batch: %d samples, content shape %s, target_style shape %s",
        content.shape[0],
        tuple(content.shape),
        tuple(target_style.shape),
    )

    # 4. SWD gradient at v=0.
    logger.info("Computing SWD gradient at v=0 ...")
    grad_results = compute_swd_gradient(model, loss_obj, content, target_style, batch, device)
    logger.info("  SWD@v=0: %.6f", grad_results["swd_at_v0"])
    logger.info("  ||∇_v SWD||: %.6f", grad_results["grad_v_norm"])
    logger.info("  Per-sample norms: %s", grad_results["grad_v_norm_per_sample"])
    logger.info("  cos(∇_v SWD, v_target) mean: %.6f", grad_results["grad_v_cos_sim_mean"])

    # 5. Sorting stability.
    logger.info("Measuring sorting stability ...")
    stability_results = measure_sorting_stability(model, loss_obj, content, target_style, batch, device)
    for eps, frac in zip(stability_results["epsilons"], stability_results["fraction_changed"]):
        logger.info("  ε=%.0e: fraction changed = %.6f", eps, frac)

    # 6. Loss landscape scan.
    logger.info("Scanning loss landscape ...")
    landscape_results = scan_loss_landscape(model, loss_obj, content, target_style, batch, device)
    logger.info("  Loss landscape (FM weight=%.3f, SWD weight=%.3f):", landscape_results["fm_weight"], landscape_results["swd_weight"])
    for entry in landscape_results["scan"]:
        logger.info(
            "    s=%.2f  FM=%.6f  SWD=%.6f  Total=%.6f",
            entry["s"],
            entry["fm_loss"],
            entry["swd_loss"],
            entry["total_loss"],
        )

    # 7. Output.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "batch_size": int(content.shape[0]),
        "content_shape": list(content.shape),
        "swd_gradient_at_v0": grad_results,
        "sorting_stability": stability_results,
        "loss_landscape_scan": landscape_results,
    }

    report_path = output_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Report saved to %s", report_path)


if __name__ == "__main__":
    main()