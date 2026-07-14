"""Batch DINO evaluation for all epoch eval directories."""
import subprocess
import sys
import os
import json
from pathlib import Path

base = Path("I:/Github/Latent_Style/SchrodingerBridge/exp/dino_s_break/brk_a_ll03_15ep/full_eval")
test_dir = "I:/datasets/wikiart_distinct5_512_images/test"
cache_dir = "I:/Github/Latent_Style/SchrodingerBridge/exp/eval_cache/hf"
script = "C:/Users/Administrator/SchrodingerBridge/src/utils/compute_dino_metrics.py"

results = []
for e in range(1, 16):
    eval_dir = base / f"epoch_{e:04d}"
    if not eval_dir.exists():
        continue
    dino_json = eval_dir / "dino_summary.json"
    if dino_json.exists():
        with open(dino_json) as f:
            d = json.load(f)
        results.append((e, d.get("all_dino_s", 0), d.get("all_dino_c", 0)))
        print(f"Epoch {e}: DINO-S={d.get('all_dino_s',0):.4f} DINO-C={d.get('all_dino_c',0):.4f} (cached)")
        continue
    images_dir = eval_dir / "images"
    if not images_dir.exists():
        print(f"Epoch {e}: no images dir, skipping")
        continue
    print(f"Epoch {e}: running DINO eval...", flush=True)
    cmd = [
        sys.executable, script,
        "--eval_dir", str(eval_dir),
        "--test_dir", test_dir,
        "--cache_dir", cache_dir,
        "--batch_size", "8",
        "--max_refs_per_style", "30",
        "--device", "cuda",
        "--allow_network",
    ]
    ret = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if ret.returncode != 0:
        print(f"  ERROR: {ret.stderr[:200]}")
        continue
    if dino_json.exists():
        with open(dino_json) as f:
            d = json.load(f)
        results.append((e, d.get("all_dino_s", 0), d.get("all_dino_c", 0)))
        print(f"  DINO-S={d.get('all_dino_s',0):.4f} DINO-C={d.get('all_dino_c',0):.4f}")

print("\n=== DINO Summary ===")
print(f"{'Ep':>3} | {'DINO-S':>7} | {'DINO-C':>7}")
print("-" * 30)
for e, ds, dc in sorted(results):
    print(f"{e:>3} | {ds:>7.4f} | {dc:>7.4f}")
