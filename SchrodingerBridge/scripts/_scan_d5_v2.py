"""Scan all summary.json for CLIP-S/LPIPS using correct nested keys."""
import json
from pathlib import Path

root = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp")
results = []
for f in root.rglob("summary.json"):
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
        a = d.get("analysis", {})
        ap = a.get("all_pairs_overview", {})
        st = a.get("style_transfer_ability", {})
        clip = ap.get("clip_style", st.get("clip_style"))
        lpips = ap.get("content_lpips", st.get("content_lpips"))
        if clip is not None and lpips is not None and 0 < clip < 1 and 0 < lpips < 1:
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
