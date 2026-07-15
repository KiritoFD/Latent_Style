#!/usr/bin/env python3
"""Create a local GPU-friendly smoke config for film_v5_gated."""
import json
from pathlib import Path

src = Path("g:/GitHub/Latent_Style/SchrodingerBridge/exp/620_spatial_bridge/620_film_v5_gated_5ep/config.json")
dst_dir = Path("g:/GitHub/Latent_Style/SchrodingerBridge/exp/620_spatial_bridge/620_film_v5_gated_local_smoke")
dst_dir.mkdir(parents=True, exist_ok=True)

with open(src) as f:
    cfg = json.load(f)

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
cfg["ablation"]["name"] = "620_film_v5_gated_local_smoke"
cfg["ablation"]["notes"] = "Local smoke test for film_v5_gated with batch=4 accum=16"

dst = dst_dir / "config.json"
with open(dst, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"Created {dst}")
