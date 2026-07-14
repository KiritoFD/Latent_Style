"""Collect DINO results from all probe experiments."""
import json
from pathlib import Path

probe_root = Path("I:/Github/Latent_Style/SchrodingerBridge/exp/model_probe")
results = []

for exp_dir in sorted(probe_root.iterdir()):
    if not exp_dir.is_dir():
        continue
    # Look for dino_summary.json in full_eval subdirectories
    for dino_path in exp_dir.rglob("dino_summary.json"):
        try:
            with open(dino_path) as f:
                d = json.load(f)
            rel = dino_path.relative_to(probe_root)
            results.append({
                "path": str(rel),
                "all_dino_s": float(d.get("all_dino_s", 0) or 0),
                "all_dino_c": float(d.get("all_dino_c", 0) or 0),
                "off_dino_s": float(d.get("off_dino_s", 0) or 0),
                "all_clip_s": float(d.get("all_clip_s", 0) or 0),
                "all_lpips": float(d.get("all_lpips", 0) or 0),
            })
        except Exception as ex:
            print(f"  ERROR reading {dino_path}: {ex}")

# Sort by DINO-S descending
results.sort(key=lambda x: x["all_dino_s"], reverse=True)

print(f"{'DINO-S':>8} | {'DINO-C':>8} | {'off-DS':>8} | {'CLIP-S':>7} | {'LPIPS':>7} | path")
print("-" * 90)
for r in results:
    print(f"{r['all_dino_s']:>8.4f} | {r['all_dino_c']:>8.4f} | {r['off_dino_s']:>8.4f} | {r['all_clip_s']:>7.4f} | {r['all_lpips']:>7.4f} | {r['path']}")

print(f"\nTotal: {len(results)} DINO eval results found")
