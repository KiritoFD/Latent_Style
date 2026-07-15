"""Update unified_results.json with CUT results."""
import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

UNIFIED = Path(r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\unified_results.json')

# Load existing
with open(UNIFIED, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Add CUT results
data['cut'] = {
    "clip_style": 0.7136621680657069,
    "content_lpips": 0.37425638182666665,
    "clip_s_delta_idt": 0.07376216806570687,
    "n_pairs": 745
}

# Write back
with open(UNIFIED, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print(f"Updated {UNIFIED}")
print(f"Methods now: {len(data)}")
print(f"CUT: clip_style={data['cut']['clip_style']:.4f}, lpips={data['cut']['content_lpips']:.4f}, delta_idt={data['cut']['clip_s_delta_idt']:+.4f}")
print("==UPDATE_DONE==")
