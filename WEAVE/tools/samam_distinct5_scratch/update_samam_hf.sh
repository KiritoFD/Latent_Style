#!/usr/bin/env bash
set -e

# Update unified_results.json with HF CLIP SaMam (step=20000, converged)
python3 << 'PYEOF'
import json

UNIFIED = "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/baseline_v2/eval/unified_results.json"

with open(UNIFIED, "r") as f:
    results = json.load(f)

# SaMam HF transformers CLIP (step=20000, converged, distinct5_512, 750 pairs)
# From curve_eval_hf_750_batched/curve_metrics.csv
samam_hf = {
    "clip_style": 0.5816,
    "content_lpips": 0.2434,
    "clip_s_delta_idt": 0.5816 - 0.6932711070378621,
    "n_pairs": 750,
    "note": "HF transformers CLIP, step=20000 (converged), 20k train, batch=1, 512x512, distinct5"
}

results["samam"] = samam_hf

with open(UNIFIED, "w") as f:
    json.dump(results, f, indent=2)

print("Updated samam in unified_results.json")
print(f"  clip_style: {samam_hf['clip_style']}")
print(f"  content_lpips: {samam_hf['content_lpips']}")
print(f"  delta_idt: {samam_hf['clip_s_delta_idt']:.4f}")
PYEOF
