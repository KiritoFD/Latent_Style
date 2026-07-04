#!/usr/bin/env bash
set -e

UNIFIED=/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/unified_results.json

python3 << 'PYEOF'
import json

UNIFIED = "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/unified_results.json"

with open(UNIFIED, "r") as f:
    results = json.load(f)

# SeeDream metrics (from metrics.csv mean, 750 rows, distinct5_512)
# CLIP backend: HF transformers (consistent with other baselines, clip_style in 0.72 range matches)
seedream_metrics = {
    "clip_style": 0.7198476771513621,
    "content_lpips": 0.47671699916000004,
    "clip_s_delta_idt": 0.7198476771513621 - 0.6932711070378621,
    "n_pairs": 750
}

results["seedream"] = seedream_metrics

with open(UNIFIED, "w") as f:
    json.dump(results, f, indent=2)

print("Added seedream to unified_results.json")
print(f"  clip_style: {seedream_metrics['clip_style']:.4f}")
print(f"  content_lpips: {seedream_metrics['content_lpips']:.4f}")
print(f"  delta_idt: {seedream_metrics['clip_s_delta_idt']:.4f}")
print(f"  n_pairs: {seedream_metrics['n_pairs']}")
print()
print("All methods now:", sorted(results.keys()))
PYEOF
