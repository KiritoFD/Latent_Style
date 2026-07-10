"""Extract CLIP-S and LPIPS from all M1 ablation summary.json files."""
import json
import os
from pathlib import Path

base = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp")
exps = ["abl_no_edge", "abl_no_terminal", "abl_no_ep_content", "abl_no_ll_fm"]
dino_dir = base / "_dino_results"

print(f"{'config':<20} {'clip_s':>8} {'lpips_1m':>9} {'dino_sty':>9} {'dino_con':>9} {'dino_str':>9}")
print("-" * 75)

for exp in exps:
    summary_path = base / exp / "full_eval" / "epoch_0015" / "summary.json"
    dino_path = dino_dir / f"{exp}.json"
    clip_s = lpips_1m = dino_sty = dino_con = dino_str = "N/A"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        overview = data.get("analysis", {}).get("all_pairs_overview", {})
        clip_s = f"{overview.get('clip_style', 0):.4f}"
        lpips_val = overview.get("content_lpips", 0)
        lpips_1m = f"{1.0 - lpips_val:.4f}" if lpips_val else "N/A"
    if dino_path.exists():
        with open(dino_path, "r", encoding="utf-8") as f:
            dino = json.load(f)
        dino_con = f"{dino.get('dino_content', 0):.4f}"
        dino_sty = f"{dino.get('dino_style', 0):.4f}"
        dino_str = f"{dino.get('dino_structure', 0):.4f}"
    print(f"{exp:<20} {clip_s:>8} {lpips_1m:>9} {dino_sty:>9} {dino_con:>9} {dino_str:>9}")
