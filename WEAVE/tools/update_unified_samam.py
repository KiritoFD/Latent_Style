"""Update unified_results.json with SaMam results."""
import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

UNIFIED = Path(r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\unified_results.json')

with open(UNIFIED, 'r', encoding='utf-8') as f:
    data = json.load(f)

data['samam'] = {
    "clip_style": 0.7221691230138143,
    "content_lpips": 0.3281765048,
    "clip_s_delta_idt": 0.08226912301381428,
    "n_pairs": 745
}

with open(UNIFIED, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print(f"Updated {UNIFIED}")
print(f"Methods now: {len(data)}")
print(f"SaMam: clip_style={data['samam']['clip_style']:.4f}, lpips={data['samam']['content_lpips']:.4f}, delta_idt={data['samam']['clip_s_delta_idt']:+.4f}")
print("==UPDATE_SAMAM_DONE==")
