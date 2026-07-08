"""Extract CLIP-S and LPIPS from eval summaries."""
from pathlib import Path
import json

# Photo2Art-256
p2a = Path(r"I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_ipadapter\photo2art256\eval\summary.json")
if p2a.exists():
    s = json.loads(p2a.read_text())
    # Might be nested: s["aggregate"]["clip_style"]
    if "aggregate" in s:
        agg = s["aggregate"]
    else:
        agg = s
    cs = agg.get("clip_style", None) or agg.get("mean_clip_style", None)
    lp = agg.get("content_lpips", None) or agg.get("mean_lpips", None)
    # Try per-style
    if cs is None and "per_style" in s:
        cs_vals = [v.get("clip_style", 0) for v in s["per_style"].values() if isinstance(v, dict)]
        lp_vals = [v.get("content_lpips", 0) for v in s["per_style"].values() if isinstance(v, dict)]
        if cs_vals:
            cs = sum(cs_vals) / len(cs_vals)
            lp = sum(lp_vals) / len(lp_vals)
    print(f"Photo2Art-256 CLIP-S={cs:.4f} LPIPS={lp:.4f}" if cs else f"Photo2Art raw: {list(s.keys())[:10]}")
else:
    print("Photo2Art-256: no summary.json")

# Random5 - check file naming issue
r5_img = Path(r"I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_ipadapter\random5\eval\images")
if r5_img.exists():
    imgs = sorted(list(r5_img.glob("*.png")))[:5]
    print(f"Random5 eval images ({len(list(r5_img.glob('*.png')))}): {[f.name for f in imgs]}")
