#!/usr/bin/env python3
"""Read full runtime_observability for film_v4_gated epoch_0005."""
import json, os

base = "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge"

# Read film_v4_gated epoch_0005 summary
sj = os.path.join(base, "620_film_v4_gated_5ep/full_eval/epoch_0005/summary.json")
s = json.load(open(sj))
ro = s.get("runtime_observability", {})

# Get all_pairs_overview model_ keys
ap = ro.get("all_pairs_overview", {})
print("=== all_pairs_overview runtime_observability ===")
for k, v in sorted(ap.items()):
    if isinstance(v, float):
        print(f"  {k}: {v:.6f}")
    else:
        print(f"  {k}: {v}")

# Also check settings
settings = s.get("settings", {})
print("\n=== Eval settings (relevant) ===")
for k in ["num_steps", "batch_size", "target_chunk_size", "save_generated_images"]:
    print(f"  {k}: {settings.get(k)}")

# Check if config has film_enabled
cfg_path = os.path.join(base, "620_film_v4_gated_5ep/config.json")
c = json.load(open(cfg_path))
model = c.get("model", {})
print("\n=== Model config (relevant) ===")
for k in ["style_film_enabled", "style_attn_mode", "style_attn_temperature",
          "style_cross_attn_gate_init", "style_cross_attn_skip_coarse"]:
    print(f"  {k}: {model.get(k)}")

# Compare with film_v2
print("\n=== film_v2_5ep epoch_0005 for comparison ===")
sj2 = os.path.join(base, "620_film_v2_5ep/full_eval/epoch_0005/summary.json")
if os.path.exists(sj2):
    s2 = json.load(open(sj2))
    ro2 = s2.get("runtime_observability", {})
    ap2 = ro2.get("all_pairs_overview", {})
    for k, v in sorted(ap2.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
