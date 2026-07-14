"""Scan all summary.json for CLIP-S/LPIPS keys."""
import json
from pathlib import Path

root = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp")
results = []
for f in root.rglob("summary.json"):
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        # Find clip-style and lpips values
        clip = None
        lpips = None
        for k, v in d.items():
            if "clip_style" in k.lower() and isinstance(v, (int, float)) and "delta" not in k.lower() and "identity" not in k.lower():
                clip = v
            if "lpips" in k.lower() and "global" in k.lower() and isinstance(v, (int, float)):
                lpips = v
            if k == "lpips_global" and isinstance(v, (int, float)):
                lpips = v
        if clip is not None and lpips is not None:
            rel = f.relative_to(root)
            results.append((str(rel.parent.parent.parent), str(rel.parent.parent), clip, lpips))
    except Exception:
        pass

results.sort(key=lambda x: -x[2])
print(f"{'exp_name':<40s} {'epoch':<20s} {'CLIP-S':>8s} {'LPIPS':>8s}")
print("-" * 80)
for name, epoch, clip, lpips in results:
    print(f"{name:<40s} {epoch:<20s} {clip:8.4f} {lpips:8.4f}")
print(f"\nTotal: {len(results)} entries")
