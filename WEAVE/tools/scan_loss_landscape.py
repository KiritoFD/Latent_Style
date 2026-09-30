#!/usr/bin/env python3
"""Loss landscape scanner for the 620 spatial bridge model.

Usage:
    python tools/scan_loss_landscape.py \
        --checkpoint path/to/checkpoint.pt \
        --config config.json \
        --output-dir docs/620/fog/landscape/ \
        --num-samples 5 \
        --num-points 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# Allow imports from the src/ package at the repository root.
_repo_root = Path(__file__).resolve().parent.parent
_src_root = _repo_root / "src"
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from src.config_schema import load_experiment_config
from src.model620 import build_spatial_bridge620_from_config, SpatialBridge620
from src.losses620 import SpatialBridgeObjective620, _sliced_wasserstein, _lowpass
from src.utils.dataset import AdaCUTLatentDataset
from src.utils.training import strip_compile_prefix


_MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan 1D / 2D loss landscape around a model checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a .pt checkpoint file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a config.json file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/620/fog/landscape/",
        help="Directory to write PNG plots and JSON data (default: docs/620/fog/landscape/).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of data samples to evaluate (default: 5).",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=40,
        help="Number of points along each scan axis (default: 40).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for dataset loading (default: 1).",
    )
    return parser


def _load_model_and_loss(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[SpatialBridge620, SpatialBridgeObjective620, dict[str, Any]]:
    experiment_cfg = load_experiment_config(config_path)
    model = build_spatial_bridge620_from_config(
        experiment_cfg.model,
        bridge_cfg=experiment_cfg.bridge,
        use_checkpointing=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = strip_compile_prefix(checkpoint["model_state_dict"])
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    loss_fn = SpatialBridgeObjective620(experiment_cfg)
    return model, loss_fn, checkpoint


def _load_data_loader(
    config_path: str,
    device: torch.device,
    batch_size: int,
    num_samples: int,
) -> list[dict[str, Any]]:
    experiment_cfg = load_experiment_config(config_path)
    data_cfg = experiment_cfg.data
    train_cfg = experiment_cfg.training

    dataset = AdaCUTLatentDataset(
        data_root=str(data_cfg.data_root),
        style_subdirs=list(data_cfg.style_subdirs),
        allow_hflip=bool(data_cfg.allow_hflip),
        identity_ratio=float(data_cfg.identity_ratio) if data_cfg.identity_ratio is not None else None,
        batch_size_hint=batch_size,
        balance_target_styles_per_batch=bool(data_cfg.balance_target_styles_per_batch),
        preload_to_gpu=False,
        virtual_length_multiplier=float(data_cfg.virtual_length_multiplier),
        content_style_sampling_weights=(
            list(data_cfg.content_style_sampling_weights)
            if data_cfg.content_style_sampling_weights is not None
            else None
        ),
        target_style_sampling_weights=(
            list(data_cfg.target_style_sampling_weights)
            if data_cfg.target_style_sampling_weights is not None
            else None
        ),
        pairing_cache_path=str(data_cfg.pairing_cache_path or ""),
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
        dino_cache_path=str(data_cfg.dino_cache_path),
        dino_cache_required=bool(data_cfg.dino_cache_required),
        dino_bank_limit_per_style=int(data_cfg.dino_bank_limit_per_style),
        style_caption_path=str(data_cfg.style_caption_path),
        device="cpu",
    )
    dataset.set_epoch(0)
    samples: list[dict[str, Any]] = []
    for idx in range(min(num_samples, len(dataset))):
        item = dataset[idx]
        for key, value in item.items():
            if torch.is_tensor(value):
                item[key] = value.unsqueeze(0).to(device)
            elif isinstance(value, (int, float)):
                item[key] = torch.tensor([value], device=device)
        samples.append(item)
    return samples


def _compute_single_sample(
    model: SpatialBridge620,
    loss_fn: SpatialBridgeObjective620,
    sample: dict[str, Any],
    num_points: int,
    device: torch.device,
) -> dict[str, Any]:
    content = sample["content"]
    target_style = sample["target_style"]
    target_style_id = sample["target_style_id"]
    source_style_id = sample.get("source_style_id")

    conditioning = _build_conditioning(sample)

    loss_dict = loss_fn.compute(
        model,
        content=content,
        target_style=target_style,
        target_style_id=target_style_id,
        source_style_id=source_style_id,
        conditioning=conditioning,
    )

    debug_state = loss_fn.last_debug
    x_t = debug_state["x_t"]
    target_velocity = debug_state["target_velocity"]
    projected_target = debug_state["projected_target"]
    pred_velocity = debug_state["pred_velocity"]

    t_tensor = torch.tensor([loss_dict["t_mean"].item()], device=device, dtype=content.dtype)
    t = float(t_tensor.item())

    with torch.no_grad():
        v_curr = model(
            x_t,
            t=t_tensor,
            style_id=target_style_id,
            style_dino_patches=conditioning.get("target_style_dino_patches"),
            style_dino_cls=conditioning.get("target_style_dino_cls"),
            content_dino_patches=conditioning.get("content_dino_patches"),
            style_latent=target_style,
            style_text_tokens=conditioning.get("target_style_text_tokens"),
        )

    v_target = target_velocity
    alpha = float(loss_fn.fm_weight)
    beta = float(loss_fn.single_step_swd_weight) if loss_fn.single_step_swd_weight > 0 else 1.0

    # ---- 1D scan along v_target ----
    s_vals = np.linspace(0.0, 2.0, num_points).astype(np.float32)
    fm_1d = np.zeros(num_points, dtype=np.float32)
    swd_1d = np.zeros(num_points, dtype=np.float32)
    total_1d = np.zeros(num_points, dtype=np.float32)

    with torch.no_grad():
        for i, s in enumerate(s_vals):
            v = torch.tensor(s, device=device, dtype=content.dtype) * v_target
            z_1 = x_t + (1.0 - t) * v
            fm_1d[i] = float(F.mse_loss(v.float(), v_target.float()).item())
            dirs = loss_fn._projection_dirs(z_1)
            swd_1d[i] = float(_sliced_wasserstein(z_1, projected_target, dirs=dirs).item())
            total_1d[i] = alpha * fm_1d[i] + beta * swd_1d[i]

    v_curr_norm = float(v_curr.float().norm().item())
    v_target_norm = float(v_target.float().norm().item())
    s_curr = v_curr_norm / max(v_target_norm, 1e-8)
    s_curr = min(s_curr, float(s_vals[-1]))
    s_curr = max(s_curr, float(s_vals[0]))

    # ---- 2D scan along v_target + random orthogonal direction ----
    v_target_flat = v_target.float().reshape(1, -1)
    random_dir = torch.randn_like(v_target_flat)
    dot = (v_target_flat * random_dir).sum(dim=1, keepdim=True)
    random_dir = random_dir - dot * v_target_flat / (v_target_flat.square().sum(dim=1, keepdim=True) + 1e-8)
    random_dir = random_dir / (random_dir.norm(dim=1, keepdim=True) + 1e-8)
    random_dir = random_dir.to(dtype=content.dtype)

    total_2d = np.zeros((num_points, num_points), dtype=np.float32)
    s1_vals = np.linspace(0.0, 2.0, num_points, dtype=np.float32)
    s2_vals = np.linspace(-1.0, 1.0, num_points, dtype=np.float32)

    with torch.no_grad():
        for i, s1 in enumerate(s1_vals):
            v_comp = torch.tensor(s1, device=device, dtype=content.dtype) * v_target
            for j, s2 in enumerate(s2_vals):
                v = v_comp + torch.tensor(s2, device=device, dtype=content.dtype) * random_dir.reshape_as(v_target)
                z_1 = x_t + (1.0 - t) * v
                fm_v = float(F.mse_loss(v.float(), v_target.float()).item())
                dirs = loss_fn._projection_dirs(z_1)
                swd_v = float(_sliced_wasserstein(z_1, projected_target, dirs=dirs).item())
                total_2d[i, j] = alpha * fm_v + beta * swd_v

    return {
        "t": t,
        "s_vals": s_vals.tolist(),
        "s1_vals": s1_vals.tolist(),
        "s2_vals": s2_vals.tolist(),
        "fm_1d": fm_1d.tolist(),
        "swd_1d": swd_1d.tolist(),
        "total_1d": total_1d.tolist(),
        "s_curr": float(s_curr),
        "total_2d": total_2d.tolist(),
        "alpha": alpha,
        "beta": beta,
        "v_curr_norm": v_curr_norm,
        "v_target_norm": v_target_norm,
    }


def _build_conditioning(sample: dict[str, Any]) -> dict[str, Any]:
    conditioning: dict[str, Any] = {}
    for key in (
        "target_style_dino_patches",
        "target_style_dino_cls",
        "content_dino_patches",
        "target_style_text_tokens",
    ):
        if key in sample and torch.is_tensor(sample[key]):
            conditioning[key] = sample[key]
    return conditioning


def _make_1d_plot(
    result: dict[str, Any],
    sample_idx: int,
    output_dir: Path,
) -> None:
    if not _MATPLOTLIB_AVAILABLE:
        return
    s_vals = np.array(result["s_vals"])
    s_curr = float(result["s_curr"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s_vals, result["fm_1d"], label="FM (||v - v_target||²)", color="tab:blue", linewidth=1.5)
    ax.plot(s_vals, result["swd_1d"], label="SWD(z₁, target)", color="tab:orange", linewidth=1.5)
    ax.plot(s_vals, result["total_1d"], label="Total (α·FM + β·SWD)", color="tab:red", linewidth=2.0)
    ax.axvline(x=s_curr, color="tab:green", linestyle="--", linewidth=1.5, label=f"v_curr (s={s_curr:.3f})")
    ax.axvline(x=1.0, color="gray", linestyle=":", linewidth=1.0, label="v_target (s=1)")
    ax.set_xlabel("s (velocity scale along v_target)")
    ax.set_ylabel("Loss value")
    ax.set_title(f"1D Loss Landscape — Sample {sample_idx} (t={result['t']:.3f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"landscape_1d_sample_{sample_idx:02d}.png", dpi=150)
    plt.close(fig)


def _make_2d_plot(
    result: dict[str, Any],
    sample_idx: int,
    output_dir: Path,
) -> None:
    if not _MATPLOTLIB_AVAILABLE:
        return
    s1_vals = np.array(result["s1_vals"])
    s2_vals = np.array(result["s2_vals"])
    total_2d = np.array(result["total_2d"])
    s_curr = float(result["s_curr"])

    fig, ax = plt.subplots(figsize=(8, 6))
    s1_grid, s2_grid = np.meshgrid(s1_vals, s2_vals, indexing="ij")
    levels = np.linspace(total_2d.min(), total_2d.max(), 20)
    contour = ax.contourf(s1_grid, s2_grid, total_2d, levels=levels, cmap="viridis", alpha=0.9)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Total Loss (α·FM + β·SWD)")
    ax.plot(s_curr, 0.0, "go", markersize=8, markeredgecolor="white", label=f"v_curr (s1={s_curr:.3f}, s2=0)")
    ax.plot(1.0, 0.0, "r*", markersize=10, markeredgecolor="white", label="v_target (s1=1, s2=0)")
    ax.set_xlabel("s1 (velocity scale along v_target)")
    ax.set_ylabel("s2 (velocity scale along random orthogonal direction)")
    ax.set_title(f"2D Loss Landscape — Sample {sample_idx} (t={result['t']:.3f})")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / f"landscape_2d_sample_{sample_idx:02d}.png", dpi=150)
    plt.close(fig)


def _make_aggregate_plot(
    all_results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    if not _MATPLOTLIB_AVAILABLE:
        return
    num_points = len(all_results[0]["s_vals"])
    fm_agg = np.zeros(num_points, dtype=np.float64)
    swd_agg = np.zeros(num_points, dtype=np.float64)
    total_agg = np.zeros(num_points, dtype=np.float64)
    for r in all_results:
        fm_agg += np.array(r["fm_1d"], dtype=np.float64)
        swd_agg += np.array(r["swd_1d"], dtype=np.float64)
        total_agg += np.array(r["total_1d"], dtype=np.float64)
    n = max(len(all_results), 1)
    fm_agg /= n
    swd_agg /= n
    total_agg /= n

    s_vals = np.array(all_results[0]["s_vals"])
    s_curr_vals = [float(r["s_curr"]) for r in all_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s_vals, fm_agg, label="FM (avg)", color="tab:blue", linewidth=1.5)
    ax.plot(s_vals, swd_agg, label="SWD (avg)", color="tab:orange", linewidth=1.5)
    ax.plot(s_vals, total_agg, label="Total (avg)", color="tab:red", linewidth=2.0)
    for idx, sc in enumerate(s_curr_vals):
        ax.axvline(x=sc, color="tab:green", linestyle="--", alpha=0.5, linewidth=1.0)
    ax.axvline(x=1.0, color="gray", linestyle=":", linewidth=1.0, label="v_target (s=1)")
    ax.set_xlabel("s (velocity scale along v_target)")
    ax.set_ylabel("Loss value")
    ax.set_title("1D Loss Landscape — Aggregate over Samples")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "landscape_1d_aggregate.png", dpi=150)
    plt.close(fig)

    # 2D aggregate
    fp = all_results[0]
    num_s1 = len(fp["s1_vals"])
    num_s2 = len(fp["s2_vals"])
    total_2d_agg = np.zeros((num_s1, num_s2), dtype=np.float64)
    for r in all_results:
        total_2d_agg += np.array(r["total_2d"], dtype=np.float64)
    total_2d_agg /= n

    s1_vals = np.array(fp["s1_vals"])
    s2_vals = np.array(fp["s2_vals"])

    fig, ax = plt.subplots(figsize=(8, 6))
    s1_grid, s2_grid = np.meshgrid(s1_vals, s2_vals, indexing="ij")
    levels = np.linspace(total_2d_agg.min(), total_2d_agg.max(), 20)
    contour = ax.contourf(s1_grid, s2_grid, total_2d_agg, levels=levels, cmap="viridis", alpha=0.9)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label("Total Loss (avg)")
    for idx, sc in enumerate(s_curr_vals):
        ax.plot(sc, 0.0, "go", markersize=5, alpha=0.7, markeredgecolor="white")
    ax.plot(1.0, 0.0, "r*", markersize=10, markeredgecolor="white", label="v_target")
    ax.set_xlabel("s1 (along v_target)")
    ax.set_ylabel("s2 (along random orthogonal direction)")
    ax.set_title("2D Loss Landscape — Aggregate over Samples")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "landscape_2d_aggregate.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = _build_parser().parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.checkpoint} ...")
    model, loss_fn, checkpoint = _load_model_and_loss(args.config, args.checkpoint, device)
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")

    print(f"Loading {args.num_samples} data samples ...")
    samples = _load_data_loader(args.config, device, args.batch_size, args.num_samples)
    print(f"  Loaded {len(samples)} samples.")

    all_results: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        print(f"  Computing landscape for sample {idx + 1}/{len(samples)} ...")
        result = _compute_single_sample(model, loss_fn, sample, args.num_points, device)
        result["sample_idx"] = idx
        all_results.append(result)

        if _MATPLOTLIB_AVAILABLE:
            _make_1d_plot(result, idx, output_dir)
            _make_2d_plot(result, idx, output_dir)

    if _MATPLOTLIB_AVAILABLE and len(all_results) > 1:
        print("  Generating aggregate plots ...")
        _make_aggregate_plot(all_results, output_dir)

    # Serialize numpy values to Python floats for JSON
    def _json_friendly(obj: Any) -> Any:
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): _json_friendly(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_friendly(v) for v in obj]
        return obj

    json_data = _json_friendly(all_results)
    json_path = output_dir / "landscape_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON data to {json_path}")

    if not _MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available — only JSON data was saved.")
    else:
        print(f"Saved plots to {output_dir}")

    print("Done.")


if __name__ == "__main__":
    main()