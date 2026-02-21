from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .dataset import AdaCUTLatentDataset
    from .losses import AdaCUTObjective
    from .model import build_model_from_config
except ImportError:  # pragma: no cover
    from dataset import AdaCUTLatentDataset
    from losses import AdaCUTObjective
    from model import build_model_from_config


LOSS_KEYS = ("w_style", "w_moment", "w_identity", "w_structure", "w_tv")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    content = batch["content"].to(device, non_blocking=True)
    target_style = batch["target_style"].to(device, non_blocking=True)
    target_style_id = batch["target_style_id"].to(device, non_blocking=True).long()
    source_style_id = batch["source_style_id"].to(device, non_blocking=True).long()
    return content, target_style, target_style_id, source_style_id


def _sample_train_knobs(cfg: dict) -> tuple[int, float, float]:
    loss_cfg = cfg.get("loss", {})
    nmin = int(loss_cfg.get("train_num_steps_min", 1))
    nmax = int(loss_cfg.get("train_num_steps_max", nmin))
    smin = float(loss_cfg.get("train_step_size_min", 1.0))
    smax = float(loss_cfg.get("train_step_size_max", smin))
    tmin = float(loss_cfg.get("train_style_strength_min", 1.0))
    tmax = float(loss_cfg.get("train_style_strength_max", tmin))
    num_steps = max(1, int(round(0.5 * (nmin + nmax))))
    step_size = 0.5 * (smin + smax)
    style_strength = 0.5 * (tmin + tmax)
    return num_steps, step_size, style_strength


def _get_dec_out_weight(model: torch.nn.Module) -> torch.Tensor:
    if not hasattr(model, "dec_out") or not hasattr(model.dec_out, "weight"):
        raise RuntimeError("Model does not expose model.dec_out.weight; please update calibrate.py to target a valid layer.")
    return model.dec_out.weight


def calibrate(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    seed = int(config.get("training", {}).get("seed", 42))
    _set_seed(seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    data_cfg = config.get("data", {})
    dataset = AdaCUTLatentDataset(
        data_root=data_cfg.get("data_root", "../../latents"),
        style_subdirs=data_cfg.get("style_subdirs", ["photo", "monet"]),
        allow_hflip=bool(data_cfg.get("allow_hflip", True)),
        preload_to_gpu=bool(data_cfg.get("preload_to_gpu", False)),
        preload_max_vram_gb=float(data_cfg.get("preload_max_vram_gb", 0.0)),
        preload_reserve_ratio=float(data_cfg.get("preload_reserve_ratio", 0.35)),
        virtual_length_multiplier=max(1, int(data_cfg.get("virtual_length_multiplier", 1))),
        device=str(device),
    )

    style_count = len(dataset.style_subdirs)
    config.setdefault("model", {})
    config["model"]["num_styles"] = style_count

    model = build_model_from_config(
        config["model"],
        use_checkpointing=bool(config.get("training", {}).get("use_gradient_checkpointing", False)),
    ).to(device)
    model.train()

    cfg_batch_size = int(config.get("training", {}).get("batch_size", 8))
    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else cfg_batch_size

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda" and not bool(getattr(dataset, "preload_to_gpu", False))),
        drop_last=False,
    )

    objective = AdaCUTObjective(config)
    objective._ensure_modules(device)  # reuse the same loss modules used by training
    assert objective._style_loss_module is not None
    assert objective._content_loss_module is not None
    assert objective._tv_loss_module is not None

    num_steps, step_size, style_strength = _sample_train_knobs(config)
    dec_out_weight = _get_dec_out_weight(model)

    print("Starting gradient calibration")
    requested_steps = int(args.steps)
    total_batches = len(loader)
    run_all = requested_steps <= 0
    run_steps = total_batches if run_all else min(requested_steps, total_batches)

    print(f"Measure layer: model.dec_out.weight, steps={run_steps}, batch_size={batch_size}")
    print(
        "Forward knobs: "
        f"num_steps={num_steps}, step_size={step_size:.4f}, style_strength={style_strength:.4f}"
    )

    grad_sums = {k: 0.0 for k in LOSS_KEYS}
    grad_counts = {k: 0 for k in LOSS_KEYS}

    for i, batch in enumerate(tqdm(loader, total=run_steps, desc="Calibrating", leave=False)):
        if i >= run_steps:
            break
        content, target_style, target_style_id, source_style_id = _to_device(batch, device)

        # style
        model.zero_grad(set_to_none=True)
        pred = objective._apply_model(model, content, target_style_id, step_size, style_strength, num_steps)
        l_style, _ = objective._style_loss_module(pred, target_style)
        l_style.backward()
        if dec_out_weight.grad is not None:
            grad_sums["w_style"] += float(dec_out_weight.grad.norm().item())
            grad_counts["w_style"] += 1

        # moment
        model.zero_grad(set_to_none=True)
        pred = objective._apply_model(model, content, target_style_id, step_size, style_strength, num_steps)
        _, l_moment = objective._style_loss_module(pred, target_style)
        l_moment.backward()
        if dec_out_weight.grad is not None:
            grad_sums["w_moment"] += float(dec_out_weight.grad.norm().item())
            grad_counts["w_moment"] += 1

        # structure
        model.zero_grad(set_to_none=True)
        pred = objective._apply_model(model, content, target_style_id, step_size, style_strength, num_steps)
        l_struct = objective._content_loss_module(pred, content)
        l_struct.backward()
        if dec_out_weight.grad is not None:
            grad_sums["w_structure"] += float(dec_out_weight.grad.norm().item())
            grad_counts["w_structure"] += 1

        # identity (force source==target condition by using source_style_id)
        model.zero_grad(set_to_none=True)
        pred_id = objective._apply_model(model, content, source_style_id, step_size, style_strength, num_steps)
        l_id = F.smooth_l1_loss(pred_id, content)
        l_id.backward()
        if dec_out_weight.grad is not None:
            grad_sums["w_identity"] += float(dec_out_weight.grad.norm().item())
            grad_counts["w_identity"] += 1

        # total variation
        model.zero_grad(set_to_none=True)
        pred = objective._apply_model(model, content, target_style_id, step_size, style_strength, num_steps)
        l_tv = objective._tv_loss_module(pred)
        l_tv.backward()
        if dec_out_weight.grad is not None:
            grad_sums["w_tv"] += float(dec_out_weight.grad.norm().item())
            grad_counts["w_tv"] += 1

    grad_norms = {
        k: (grad_sums[k] / max(1, grad_counts[k]))
        for k in LOSS_KEYS
    }

    print("\nMeasured raw gradient norms:")
    for k in LOSS_KEYS:
        print(f"  {k:<12} = {grad_norms[k]:.8f} (n={grad_counts[k]})")

    if args.anchor_key not in LOSS_KEYS:
        raise ValueError(f"anchor_key must be one of {LOSS_KEYS}, got: {args.anchor_key}")
    if grad_norms[args.anchor_key] <= 1e-12:
        raise RuntimeError(f"Anchor gradient for {args.anchor_key} is too small; cannot calibrate.")

    target_force = float(args.anchor_weight) * grad_norms[args.anchor_key]
    bias = {
        "w_style": 1.0,
        "w_moment": float(args.moment_bias),
        "w_identity": float(args.identity_bias),
        "w_structure": float(args.structure_bias),
        "w_tv": float(args.tv_bias),
    }

    suggested = {}
    for k in LOSS_KEYS:
        g = grad_norms[k]
        if g <= 1e-12:
            suggested[k] = 0.0
            continue
        suggested[k] = (target_force / g) * bias[k]

    print(f"\nSuggested balanced weights (anchor: {args.anchor_key}={args.anchor_weight}):")
    for k in LOSS_KEYS:
        print(f"  {k:<12} = {suggested[k]:.6f}")

    out = {
        "config": str(config_path),
        "device": str(device),
        "layer": "model.dec_out.weight",
        "forward": {
            "num_steps": num_steps,
            "step_size": step_size,
            "style_strength": style_strength,
        },
        "grad_norms": grad_norms,
        "suggested_weights": suggested,
    }

    if args.save:
        save_path = Path(args.save).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved calibration report to: {save_path}")

    print("\nPaste suggested values into config.loss and start training.")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Calibrate multi-loss weights via gradient norm balancing.")
    p.add_argument("--config", type=str, default="overfit50.json", help="Path to config JSON.")
    p.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu/cuda:0 ...")
    p.add_argument("--batch-size", type=int, default=8, help="Calibration batch size. Use 0 to inherit config.training.batch_size.")
    p.add_argument("--steps", type=int, default=5, help="Number of batches to average. Use 0 to run all batches.")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    p.add_argument("--anchor-key", type=str, default="w_style", choices=LOSS_KEYS, help="Reference loss key.")
    p.add_argument("--anchor-weight", type=float, default=150.0, help="Reference weight for anchor loss.")
    p.add_argument("--structure-bias", type=float, default=0.5, help="Post-scale bias for structure weight.")
    p.add_argument("--identity-bias", type=float, default=0.2, help="Post-scale bias for identity weight.")
    p.add_argument("--moment-bias", type=float, default=1.0, help="Post-scale bias for moment weight.")
    p.add_argument("--tv-bias", type=float, default=1.0, help="Post-scale bias for TV weight.")
    p.add_argument("--save", type=str, default="", help="Optional JSON report path.")
    return p


if __name__ == "__main__":
    parser = build_argparser()
    raise SystemExit(calibrate(parser.parse_args()))
