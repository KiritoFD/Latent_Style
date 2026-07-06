#!/usr/bin/env python3
"""Analyze ablation design: show config parameter differences and trained experiments."""
from __future__ import annotations

import json
import sys
from pathlib import Path

CONFIG_DIR = Path("/mnt/i/Github/Latent_Style/SchrodingerBridge/ablation_620")
EXP_DIR = Path("/mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620")


def short_config(cfg: dict) -> dict:
    """Extract the most important config fields for ablation comparison."""
    m = cfg.get("model", {}) or {}
    b = cfg.get("bridge", {}) or {}
    t = cfg.get("training", {}) or {}
    return {
        "depth": m.get("depth", "?"),
        "base_dim": m.get("base_dim", "?"),
        "heads": m.get("heads", "?"),
        "style_gate_init": m.get("style_gate_init", "?"),
        "style_shortcut_alpha": m.get("style_shortcut_alpha", "?"),
        "style_embed_init": m.get("style_embed_init", "?"),
        "endpoint_velocity_floor": m.get("endpoint_velocity_floor", "?"),
        "endpoint_high_scale": m.get("endpoint_high_scale", "?"),
        "spectral_w_hh": m.get("spectral_w_hh", "?"),
        "objective_mode": b.get("objective_mode", "?"),
        "loss_type": b.get("loss_type", "?"),
        "semantic_supervision_family": b.get("semantic_supervision_family", "?"),
        "i2sb_predictor_time_floor": b.get("i2sb_predictor_time_floor", "?"),
        "i2sb_style_noise_amplitude_power": b.get("i2sb_style_noise_amplitude_power", "?"),
        "batch_size": t.get("batch_size", "?"),
        "num_epochs": t.get("num_epochs", "?"),
        "acc_grad_steps": t.get("acc_grad_steps", "?"),
        "use_amp": t.get("use_amp", "?"),
        "tf32": t.get("tf32", "?"),
    }


def main() -> int:
    # Find the baseline config to compare against
    # Look for infra_I0_baseline or DA01_backbone1
    baseline_path = CONFIG_DIR / "infra_I0_baseline" / "config.json"
    if not baseline_path.is_file():
        baseline_path = CONFIG_DIR / "DA01_backbone1" / "config.json"
    baseline_cfg = {}
    if baseline_path.is_file():
        with baseline_path.open() as f:
            baseline_cfg = short_config(json.load(f))
    print(f"Baseline: {baseline_path.parent.name}")
    print(f"  {baseline_cfg}")
    print()

    # Group by category
    categories = {"DA": "Architecture", "DD": "Data", "DI": "Infrastructure",
                  "DL": "Loss", "DN": "Inference", "infra_I0": "Baseline"}
    by_cat: dict[str, list[tuple[str, dict, bool]]] = {}

    for exp_dir in sorted(EXP_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        cfg_path = CONFIG_DIR / name / "config.json"
        if not cfg_path.is_file():
            continue
        with cfg_path.open() as f:
            cfg = short_config(json.load(f))
        has_ckpt = (exp_dir / "epoch_0003.pt").is_file()
        cat = "Baseline" if name == "infra_I0_baseline" else categories.get(name[:2], "Other")
        by_cat.setdefault(cat, []).append((name, cfg, has_ckpt))

    for cat in ["Architecture", "Data", "Infrastructure", "Loss", "Inference", "Baseline"]:
        if cat not in by_cat:
            continue
        print(f"\n=== {cat} ===")
        for name, cfg, has_ckpt in by_cat[cat]:
            # Show only fields that differ from baseline
            diffs = {}
            for k, v in cfg.items():
                bv = baseline_cfg.get(k, "?")
                if str(v) != str(bv):
                    diffs[k] = f"base={bv} -> {v}"
            ckpt = "OK" if has_ckpt else "NO_CKPT"
            print(f"  {name} [{ckpt}]")
            if diffs:
                for k, v in diffs.items():
                    print(f"    {k}: {v}")
            else:
                print("    (same as baseline)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
