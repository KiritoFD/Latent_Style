#!/usr/bin/env python3
"""Analyze ablation design - compare configs against baseline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

CONFIG_DIR = Path("/mnt/i/Github/Latent_Style/SchrodingerBridge/ablation_620")
EXP_DIR = Path("/mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620")


def flatten(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def main() -> int:
    # Use DA01_backbone1 as baseline (it's the "1 block" variant but functionally the closest to default)
    # Actually, let me check what the "default" mainline config looks like
    baseline_path = CONFIG_DIR / "DA01_backbone1" / "config.json"
    with baseline_path.open() as f:
        baseline_full = json.load(f)
    baseline = flatten(baseline_full)

    print(f"Baseline: DA01_backbone1")
    print(f"  num_res_blocks={baseline.get('model.num_res_blocks')}")
    print(f"  style_attn_num_heads={baseline.get('model.style_attn_num_heads')}")
    print(f"  style_cross_attn_gate_init={baseline.get('model.style_cross_attn_gate_init')}")
    print(f"  style_shortcut_alpha={baseline.get('model.style_shortcut_alpha')}")
    print(f"  style_embed_scale={baseline.get('model.style_embed_scale')}")
    print(f"  endpoint_delta_scale={baseline.get('model.endpoint_delta_scale')}")
    print(f"  endpoint_velocity_floor={baseline.get('model.endpoint_velocity_floor')}")
    print(f"  single_step_swd_weight={baseline.get('bridge.single_step_swd_weight')}")
    print(f"  single_step_edge_weight={baseline.get('bridge.single_step_edge_weight')}")
    print(f"  w_flow={baseline.get('bridge.w_flow')}")
    print(f"  loss_type={baseline.get('bridge.loss_type')}")
    print(f"  batch_size={baseline.get('training.batch_size')}")
    print()

    # Compare each experiment to baseline
    categories = {"DA": "Architecture", "DD": "Data", "DI": "Infrastructure",
                  "DL": "Loss", "DN": "Inference"}
    by_cat: dict[str, list] = {}

    for exp_dir in sorted(EXP_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        if name == "infra_I0_baseline":
            continue
        cfg_path = CONFIG_DIR / name / "config.json"
        if not cfg_path.is_file():
            continue
        with cfg_path.open() as f:
            cfg = flatten(json.load(f))
        has_ckpt = (exp_dir / "epoch_0003.pt").is_file()
        # Find differences
        diffs = {}
        for k, v in cfg.items():
            bv = baseline.get(k)
            if bv is not None and str(v) != str(bv):
                diffs[k] = (bv, v)
        cat = categories.get(name[:2], "Other")
        by_cat.setdefault(cat, []).append((name, diffs, has_ckpt))

    for cat in ["Architecture", "Data", "Infrastructure", "Loss", "Inference"]:
        if cat not in by_cat:
            continue
        print(f"\n=== {cat} ===")
        for name, diffs, has_ckpt in by_cat[cat]:
            ckpt = "OK" if has_ckpt else "NO_CKPT"
            if not diffs:
                print(f"  {name} [{ckpt}]: (no diff found - check nested fields)")
                continue
            diff_str = ", ".join(f"{k}={v[0]}->{v[1]}" for k, v in diffs.items())
            print(f"  {name} [{ckpt}]: {diff_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
