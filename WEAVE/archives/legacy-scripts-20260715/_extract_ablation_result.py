"""Extract ablation results from summary.json and DINO results."""
import json
import sys
from pathlib import Path

exp_root = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp")
dino_root = exp_root / "_dino_results"

names = sys.argv[1:] if len(sys.argv) > 1 else ["abl_no_flow", "abl_flow_only"]

print(f"{'Config':<20} {'CLIP-S':>8} {'LPIPS':>8} {'DINO-C':>8} {'DINO-S':>8}")
print("-" * 60)

for name in names:
    summary_path = exp_root / name / "full_eval" / "epoch_0005" / "summary.json"
    dino_path = dino_root / f"{name}.json"

    clip_s = lpips = dino_c = dino_s = "N/A"

    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            s = json.load(f)
        o = s.get("analysis", {}).get("all_pairs_overview", {})
        clip_s = f"{o.get('clip_style', 0):.4f}"
        lpips = f"{o.get('content_lpips', 0):.4f}"
    else:
        print(f"  {name}: summary not found at {summary_path}")

    if dino_path.exists():
        with open(dino_path, encoding="utf-8") as f:
            d = json.load(f)
        dino_c = f"{d.get('dino_content', 0):.4f}"
        dino_s = f"{d.get('dino_style', 0):.4f}"
    else:
        print(f"  {name}: DINO results not found at {dino_path}")

    print(f"{name:<20} {clip_s:>8} {lpips:>8} {dino_c:>8} {dino_s:>8}")
