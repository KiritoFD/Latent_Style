#!/usr/bin/env python3
"""Create local GPU-friendly smoke configs for attention ablation."""
import json
from pathlib import Path

SRC = Path("g:/GitHub/Latent_Style/SchrodingerBridge/exp/620_spatial_bridge/620_film_v5_gated_5ep/config.json")
BASE_DIR = Path("g:/GitHub/Latent_Style/SchrodingerBridge/exp/620_spatial_bridge")

VARIANTS = {
    "620_film_v5_gated_raw_local_smoke": {
        "style_attn_mode": "gated_raw",
        "style_attn_temperature": 1.0,
        "notes": "Gated attention without renormalization",
    },
    "620_film_v5_relu2_local_smoke": {
        "style_attn_mode": "relu2",
        "style_attn_temperature": 1.0,
        "notes": "ReLU^2 attention without softmax",
    },
    "620_film_v5_style_select_local_smoke": {
        "style_attn_mode": "style_select",
        "style_attn_temperature": 1.0,
        "notes": "Top-k style token selection before softmax",
    },
}

with open(SRC) as f:
    base_cfg = json.load(f)

for name, delta in VARIANTS.items():
    dst_dir = BASE_DIR / name
    dst_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(json.dumps(base_cfg))
    cfg["model"]["style_attn_mode"] = delta["style_attn_mode"]
    cfg["model"]["style_attn_temperature"] = delta["style_attn_temperature"]

    cfg["training"]["batch_size"] = 4
    cfg["training"]["accumulation_steps"] = 16
    cfg["training"]["num_epochs"] = 1
    cfg["training"]["save_interval"] = 1
    cfg["training"]["full_eval_each_epoch"] = True
    cfg["training"]["full_eval_batch_size"] = 4
    cfg["training"]["full_eval_vae_decode_batch_size"] = 4

    cfg["full_eval"]["batch_size"] = 4
    cfg["full_eval"]["vae_decode_batch_size"] = 4
    cfg["full_eval"]["ref_feature_batch_size"] = 4
    cfg["full_eval"]["max_src_samples"] = 30
    cfg["full_eval"]["max_ref_compare"] = 30
    cfg["full_eval"]["max_ref_cache"] = 30

    cfg["checkpoint"]["save_dir"] = str(dst_dir)
    cfg["ablation"]["name"] = name
    cfg["ablation"]["notes"] = f"Local smoke test: {delta['notes']}"

    dst = dst_dir / "config.json"
    with open(dst, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Created {dst}")
