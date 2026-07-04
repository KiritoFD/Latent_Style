#!/usr/bin/env python3
"""Check remote experiment state - run on remote machine."""
import json, os, sys, glob

base = "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge"

# 1. Check base config full_eval settings
print("=== Base config (620_intrinsic_v2) full_eval settings ===")
try:
    c = json.load(open(os.path.join(base, "620_intrinsic_v2/config.json")))
    fe = c.get("full_eval", {})
    print("full_eval keys:", list(fe.keys()))
    for k in ["save_generated_images", "only_lpips_clip_style", "num_steps",
              "batch_size", "test_dir", "cache_dir", "clip_hf_cache_dir",
              "target_dino_cache", "eval_enable_art_fid", "eval_enable_kid",
              "eval_enable_introstyle"]:
        print(f"  {k}: {fe.get(k)}")
except Exception as e:
    print(f"Error: {e}")

# 2. Check film_gate03_5ep epoch_0005 eval output
print("\n=== film_gate03_5ep epoch_0005 eval output ===")
eval_dir = os.path.join(base, "620_film_gate03_5ep/full_eval/epoch_0005")
if os.path.exists(eval_dir):
    print("Contents:", os.listdir(eval_dir))
    images_dir = os.path.join(eval_dir, "images")
    if os.path.exists(images_dir):
        imgs = os.listdir(images_dir)
        print(f"Images count: {len(imgs)}")
        if imgs:
            print(f"First 3: {imgs[:3]}")
    else:
        print("No images/ directory")
    # Check summary.json
    sj = os.path.join(eval_dir, "summary.json")
    if os.path.exists(sj):
        s = json.load(open(sj))
        print(f"summary.json keys: {list(s.keys())}")
        ap = s.get("analysis", {}).get("all_pairs_overview", {})
        print(f"all_pairs_overview: {json.dumps(ap, indent=2)[:800]}")
else:
    print(f"Eval dir does not exist: {eval_dir}")

# 3. List all film experiments and their eval status
print("\n=== All film experiments eval status ===")
for exp in sorted(os.listdir(base)):
    if "film" not in exp and "intrinsic" not in exp:
        continue
    exp_path = os.path.join(base, exp)
    if not os.path.isdir(exp_path):
        continue
    fe_dir = os.path.join(exp_path, "full_eval")
    ckpts = sorted(glob.glob(os.path.join(exp_path, "epoch_*.pt")))
    evals = sorted(glob.glob(os.path.join(fe_dir, "epoch_*"))) if os.path.exists(fe_dir) else []
    has_images = any(os.path.exists(os.path.join(e, "images")) for e in evals)
    print(f"  {exp}: ckpts={len(ckpts)}, evals={len(evals)}, has_images={has_images}")

# 4. Check film_v4_gated progress
print("\n=== film_v4_gated progress ===")
log = os.path.join(base, "620_film_v4_gated_5ep/train.log")
if os.path.exists(log):
    with open(log) as f:
        lines = f.readlines()
    # Show last 10 lines
    for line in lines[-10:]:
        print(f"  {line.rstrip()}")

# 5. Check film_v3 crash
print("\n=== film_v3 crash info ===")
log = os.path.join(base, "620_film_v3_5ep/train.log")
if os.path.exists(log):
    with open(log) as f:
        lines = f.readlines()
    # Show last 15 lines
    for line in lines[-15:]:
        print(f"  {line.rstrip()}")
