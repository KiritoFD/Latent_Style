"""Extract CLIP-S / LPIPS / MUSIQ from experiment summary.json files."""
import json
import os

exps = [
    "hp_simple_swd12_15ep",
    "d1_gram_hf1_15ep",
    "d1_gram_hf5_15ep",
    "d2_moment_hf1_15ep",
]
base = r"I:\Github\Latent_Style\SchrodingerBridge\exp"
dino_base = os.path.join(base, "_dino_results")

print(f"{'config':<28} {'clip_s':>8} {'lpips':>8} {'1-LPIPS':>8} | {'dino_sty':>9} {'dino_con':>9} {'dino_str':>9}")
print("-" * 95)
for name in exps:
    summary_path = os.path.join(base, name, "full_eval", "epoch_0015", "summary.json")
    dino_path = os.path.join(dino_base, f"{name}.json")
    clip_s = lpips = lpips_1m = "?"
    dino_sty = dino_con = dino_str = "?"
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        ov = d.get("analysis", {}).get("all_pairs_overview", {})
        clip_s = ov.get("clip_style", "?")
        lpips = ov.get("content_lpips", "?")
        if isinstance(lpips, (int, float)):
            lpips_1m = 1.0 - lpips
    if os.path.exists(dino_path):
        with open(dino_path, "r", encoding="utf-8") as f:
            dd = json.load(f)
        dino_sty = dd.get("dino_style", "?")
        dino_con = dd.get("dino_content", "?")
        dino_str = dd.get("dino_structure", "?")
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
    print(f"{name:<28} {fmt(clip_s):>8} {fmt(lpips):>8} {fmt(lpips_1m):>8} | {fmt(dino_sty):>9} {fmt(dino_con):>9} {fmt(dino_str):>9}")
