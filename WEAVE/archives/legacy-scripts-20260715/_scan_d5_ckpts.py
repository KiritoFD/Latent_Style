"""Scan all D5 summary.json files for CLIP-S and LPIPS values."""
import json
from pathlib import Path

root = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp")
results = []
for f in root.rglob("summary.json"):
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        clip = d.get("clip_style_global")
        lpips = d.get("lpips_global")
        if clip is not None and lpips is not None:
            # Extract relative path for identification
            rel = f.relative_to(root)
            results.append((rel.parent.parent.parent.name, rel.parent.parent.name, clip, lpips))
    except Exception:
        pass

# Sort by CLIP-S descending
results.sort(key=lambda x: -x[2])
print(f"{'exp_name':<35s} {'epoch':<15s} {'CLIP-S':>8s} {'LPIPS':>8s}")
print("-" * 70)
for name, epoch, clip, lpips in results:
    print(f"{name:<35s} {epoch:<15s} {clip:8.4f} {lpips:8.4f}")
