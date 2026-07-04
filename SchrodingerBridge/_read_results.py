"""Read all summary_*.json from clip073_results and plana_sweep_results."""
import json
from pathlib import Path

result_dirs = [
    ("R1_clip073", Path("I:/Github/Latent_Style/SchrodingerBridge/exp/clip073_results")),
    ("R1_plana", Path("I:/Github/Latent_Style/SchrodingerBridge/exp/plana_sweep_results")),
]

for label, result_dir in result_dirs:
    if not result_dir.exists():
        print(f"[{label}] dir not found: {result_dir}")
        continue
    print(f"\n=== {label}: {result_dir} ===")
    print(f"{'Name':<25} {'CLIP-S':>10} {'LPIPS':>10} {'Status':>14}")
    print("-" * 65)
    rows = []
    for f in sorted(result_dir.glob("summary_*.json")):
        try:
            with f.open("r", encoding="utf-8") as fh:
                s = json.load(fh)
            clip = s.get("analysis", {}).get("all_pairs_overview", {}).get("clip_style", 0)
            lpips = s.get("analysis", {}).get("all_pairs_overview", {}).get("content_lpips", 0)
        except Exception as e:
            print(f"  ERROR reading {f.name}: {e}")
            continue
        name = f.stem.replace("summary_", "")
        status = "?"
        if clip > 0.73 and lpips < 0.35:
            status = "*** DOUBLE WIN ***"
        elif clip > 0.73:
            status = "clip OK"
        elif lpips < 0.35:
            status = "lpips OK"
        else:
            status = "neither"
        rows.append((name, clip, lpips, status))

    rows.sort(key=lambda x: -x[1])
    for name, clip, lpips, status in rows:
        print(f"{name:<25} {clip:>10.4f} {lpips:>10.4f} {status:>14}")

print("\n=== T11 BASELINE ===")
print("T11 (8-step)            clip=0.7213 lpips=0.2868  (baseline)")
print("\nTarget: clip > 0.73 AND lpips < 0.35")
