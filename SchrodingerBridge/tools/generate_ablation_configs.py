#!/usr/bin/env python3
"""
Generate 620 ablation configs from the current best baseline.

Usage:
    python tools/generate_ablation_configs.py \
        --base exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/config.json \
        --outdir configs/ablations \
        --spec configs/ablations/ablation_batch.json
"""
from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Current best baseline fields (used as reference)
BASELINE = {
    "style_attn_mode": "gated",
    "style_film_enabled": True,
    "endpoint_head_mode": "endpoint_lowhigh",
    "endpoint_film_enabled": True,
    "endpoint_style_hidden_dim": 512,
    "style_cross_attn_gate_init": 0.3,
    "base_dim": 64,
    "num_res_blocks": 4,
    "style_attn_num_heads": 4,
}

ABLATION_SPECS: list[dict] = [
    # Task 2.1: Attention mechanism
    {"name": "620_ablation_attn_gated_smoke", "overrides": {"style_attn_mode": "gated"}, "axis": "attention", "notes": "Gated attention (current best reproduction)"},
    {"name": "620_ablation_attn_softmax_smoke", "overrides": {"style_attn_mode": "softmax"}, "axis": "attention", "notes": "Standard softmax attention"},
    {"name": "620_ablation_attn_gated_raw_smoke", "overrides": {"style_attn_mode": "gated_raw"}, "axis": "attention", "notes": "Gated raw attention (historically harmful)"},
    {"name": "620_ablation_attn_relu2_smoke", "overrides": {"style_attn_mode": "relu2"}, "axis": "attention", "notes": "ReLU-squared attention (historically harmful)"},
    {"name": "620_ablation_attn_style_select_smoke", "overrides": {"style_attn_mode": "style_select"}, "axis": "attention", "notes": "Style-select attention (historically harmful)"},
    {"name": "620_ablation_attn_sparsemax_smoke", "overrides": {"style_attn_mode": "sparsemax"}, "axis": "attention", "notes": "Sparsemax attention"},

    # Task 2.2: StyleFiLM
    {"name": "620_ablation_stylefilm_on_smoke", "overrides": {"style_film_enabled": True}, "axis": "style_film", "notes": "Block-level StyleFiLM enabled (current best)"},
    {"name": "620_ablation_stylefilm_off_smoke", "overrides": {"style_film_enabled": False}, "axis": "style_film", "notes": "Block-level StyleFiLM disabled"},

    # Task 2.3: Endpoint structure
    {"name": "620_ablation_endpoint_velocity_smoke", "overrides": {"endpoint_head_mode": "velocity", "endpoint_film_enabled": False}, "axis": "endpoint", "notes": "Velocity endpoint head (pre-fix baseline)"},
    {"name": "620_ablation_endpoint_lowhigh_nofilm_smoke", "overrides": {"endpoint_head_mode": "endpoint_lowhigh", "endpoint_film_enabled": False}, "axis": "endpoint", "notes": "Low/high endpoint without FiLM"},
    {"name": "620_ablation_endpoint_lowhigh_hd128_smoke", "overrides": {"endpoint_head_mode": "endpoint_lowhigh", "endpoint_film_enabled": True, "endpoint_style_hidden_dim": 128}, "axis": "endpoint", "notes": "Low/high endpoint FiLM hidden dim 128"},
    {"name": "620_ablation_endpoint_lowhigh_hd256_smoke", "overrides": {"endpoint_head_mode": "endpoint_lowhigh", "endpoint_film_enabled": True, "endpoint_style_hidden_dim": 256}, "axis": "endpoint", "notes": "Low/high endpoint FiLM hidden dim 256"},
    {"name": "620_ablation_endpoint_lowhigh_hd512_smoke", "overrides": {"endpoint_head_mode": "endpoint_lowhigh", "endpoint_film_enabled": True, "endpoint_style_hidden_dim": 512}, "axis": "endpoint", "notes": "Low/high endpoint FiLM hidden dim 512 (current best)"},

    # Task 2.4: Gate init
    {"name": "620_ablation_gate_init005_smoke", "overrides": {"style_cross_attn_gate_init": 0.05}, "axis": "gate_init", "notes": "Cross-attention gate init 0.05"},
    {"name": "620_ablation_gate_init03_smoke", "overrides": {"style_cross_attn_gate_init": 0.3}, "axis": "gate_init", "notes": "Cross-attention gate init 0.3 (current best)"},
    {"name": "620_ablation_gate_init05_smoke", "overrides": {"style_cross_attn_gate_init": 0.5}, "axis": "gate_init", "notes": "Cross-attention gate init 0.5"},

    # Phase 3: Capacity
    {"name": "620_ablation_capacity_64x4_smoke", "overrides": {"base_dim": 64, "num_res_blocks": 4, "style_attn_num_heads": 4}, "axis": "capacity", "notes": "Base capacity 64x4 (current best)"},
    {"name": "620_ablation_capacity_64x6_smoke", "overrides": {"base_dim": 64, "num_res_blocks": 6, "style_attn_num_heads": 4}, "axis": "capacity", "notes": "Deeper capacity 64x6"},
    {"name": "620_ablation_capacity_128x4_smoke", "overrides": {"base_dim": 128, "num_res_blocks": 4, "style_attn_num_heads": 8}, "axis": "capacity", "notes": "Wider capacity 128x4"},
    {"name": "620_ablation_capacity_128x6_smoke", "overrides": {"base_dim": 128, "num_res_blocks": 6, "style_attn_num_heads": 8}, "axis": "capacity", "notes": "Wider and deeper 128x6"},

    # Phase 3: Loss
    {"name": "620_ablation_loss_swd0_smoke", "overrides": {"bridge.single_step_swd_weight": 0.0}, "axis": "loss", "notes": "Disable single-step SWD"},
    {"name": "620_ablation_loss_swd2_smoke", "overrides": {"bridge.single_step_swd_weight": 2.0}, "axis": "loss", "notes": "Single-step SWD weight 2.0"},
    {"name": "620_ablation_loss_swd8_smoke", "overrides": {"bridge.single_step_swd_weight": 8.0}, "axis": "loss", "notes": "Single-step SWD weight 8.0 (current best)"},
    {"name": "620_ablation_loss_swd16_smoke", "overrides": {"bridge.single_step_swd_weight": 16.0}, "axis": "loss", "notes": "Single-step SWD weight 16.0"},
    {"name": "620_ablation_loss_nosigma_smoke", "overrides": {"bridge.swd_noise_sigma": 0.0}, "axis": "loss", "notes": "Disable SWD noise (NSWD off)"},
    {"name": "620_ablation_loss_edge0_smoke", "overrides": {"bridge.single_step_edge_weight": 0.0}, "axis": "loss", "notes": "Disable single-step edge loss"},

    # Phase 3: DINO / conditioning source
    {"name": "620_ablation_dino_baseline_smoke", "overrides": {"style_condition_source": "target_dino_patches", "style_dino_adapter_enabled": False, "style_moe_enabled": False}, "axis": "condition_source", "notes": "DINO patches baseline (current best)"},
    {"name": "620_ablation_dino_adapter_smoke", "overrides": {"style_condition_source": "target_dino_patches", "style_dino_adapter_enabled": True, "style_moe_enabled": False}, "axis": "condition_source", "notes": "DINO patches with adapter"},
    {"name": "620_ablation_dino_moe_smoke", "overrides": {"style_condition_source": "target_dino_patches", "style_dino_adapter_enabled": False, "style_moe_enabled": True}, "axis": "condition_source", "notes": "DINO patches with MoE"},
    {"name": "620_ablation_intrinsic_latent_smoke", "overrides": {"style_condition_source": "latent", "style_dino_adapter_enabled": False, "style_moe_enabled": False}, "axis": "condition_source", "notes": "Intrinsic latent conditioning (no DINO)"},
]


def set_nested(cfg: dict, key_path: str, value) -> None:
    parts = key_path.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node:
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def make_config(base_cfg: dict, spec: dict) -> dict:
    cfg = deepcopy(base_cfg)
    for key, value in spec["overrides"].items():
        set_nested(cfg, key, value)

    # Update save dir
    save_dir = str(REPO_ROOT / "exp" / "620_spatial_bridge" / spec["name"])
    cfg.setdefault("checkpoint", {})["save_dir"] = save_dir

    # Update ablation metadata
    cfg["ablation"] = {
        "name": spec["name"],
        "axis": spec["axis"],
        "stage": "smoke",
        "notes": spec["notes"],
    }

    # Sanity: ensure training.num_epochs = 1 and save_interval = 1 for smoke
    cfg.setdefault("training", {})
    cfg["training"]["num_epochs"] = 1
    cfg["training"]["save_interval"] = 1

    # Ensure data paths exist in base config (they should)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Generate 620 ablation configs")
    parser.add_argument("--base", required=True, help="Path to baseline config JSON")
    parser.add_argument("--outdir", default="configs/ablations", help="Output directory for generated configs")
    parser.add_argument("--spec", default="configs/ablations/ablation_batch.json", help="Output batch spec JSON")
    parser.add_argument("--filter-axis", default=None, help="Only generate configs for given axis")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_path = Path(args.base)
    if not base_path.is_absolute():
        base_path = REPO_ROOT / base_path

    with open(base_path, "r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = REPO_ROOT / outdir

    if not args.dry_run:
        os.makedirs(outdir, exist_ok=True)

    specs = []
    generated = []
    for spec in ABLATION_SPECS:
        if args.filter_axis and spec["axis"] != args.filter_axis:
            continue
        cfg = make_config(base_cfg, spec)
        cfg_path = outdir / f"{spec['name']}.json"
        if not args.dry_run:
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        generated.append((spec["name"], cfg_path))
        specs.append({
            "name": spec["name"],
            "config": str(cfg_path),
            "epochs": ["epoch_0001"],
        })

    spec_path = Path(args.spec)
    if not spec_path.is_absolute():
        spec_path = REPO_ROOT / spec_path

    if not args.dry_run:
        os.makedirs(spec_path.parent, exist_ok=True)
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump({"experiments": specs}, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(generated)} configs in {outdir}")
    print(f"Batch spec written to {spec_path}")
    for name, path in generated:
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
