"""Check progress of reeval_with_dino: count images and check dino_summary.json for each epoch."""
import json
from pathlib import Path

ckpt_dir = Path("I:/Github/Latent_Style/SchrodingerBridge/exp/dino_s_break/brk_a_ll03_15ep")
full_eval = ckpt_dir / "full_eval"

print(f"{'Ep':>3} | {'images':>6} | {'dino':>4} | {'clip_s':>7} | {'lpips':>7} | {'dino_s':>7} | {'dino_c':>7}")
print("-" * 60)
for e in range(1, 16):
    eval_dir = full_eval / f"epoch_{e:04d}"
    images_dir = eval_dir / "images"
    img_count = len(list(images_dir.iterdir())) if images_dir.exists() else 0
    dino_path = eval_dir / "dino_summary.json"
    summary_path = eval_dir / "summary.json"

    clip_s = lpips = dino_s = dino_c = 0.0
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                d = json.load(f)
            ov = d.get("analysis", {}).get("all_pairs_overview", {})
            clip_s = float(ov.get("clip_style", 0) or 0)
            lpips = float(ov.get("content_lpips", 0) or 0)
        except Exception as ex:
            print(f"  summary parse error: {ex}")
    if dino_path.exists():
        try:
            with open(dino_path) as f:
                d = json.load(f)
            dino_s = float(d.get("all_dino_s", 0) or 0)
            dino_c = float(d.get("all_dino_c", 0) or 0)
        except Exception as ex:
            print(f"  dino parse error: {ex}")
    dino_flag = "Y" if dino_path.exists() else "N"
    print(f"{e:>3} | {img_count:>6} | {dino_flag:>4} | {clip_s:>7.4f} | {lpips:>7.4f} | {dino_s:>7.4f} | {dino_c:>7.4f}")
