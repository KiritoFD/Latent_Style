"""Check IP-Adapter results on remote."""
from pathlib import Path
import json

root = Path(r"I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_ipadapter")

for d in sorted(root.iterdir()):
    if not d.is_dir():
        continue
    imgs = list((d / "images").glob("*.png")) + list((d / "images").glob("*.jpg"))
    summary = d / "eval" / "summary.json"
    if summary.exists():
        s = json.loads(summary.read_text())
        cs = s.get("clip_style", "N/A")
        lp = s.get("content_lpips", "N/A")
        print(f"{d.name}: {len(imgs)} images | CLIP-S={cs:.4f} LPIPS={lp:.4f}" if isinstance(cs, float) else f"{d.name}: {len(imgs)} images | {s}")
    else:
        print(f"{d.name}: {len(imgs)} images | no summary.json")
