"""Collect all spectral ensemble results from summary.json files."""
import json
from pathlib import Path

ensemble_root = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\spectral_ensemble")
results = []

for d in sorted(ensemble_root.iterdir()):
    if not d.is_dir():
        continue
    summary_path = d / "summary.json"
    if not summary_path.exists():
        print(f"  {d.name}: no summary.json")
        continue
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    analysis = data.get("analysis", {})
    overview = analysis.get("all_pairs_overview", {})
    transfer = analysis.get("style_transfer_ability", {})
    clip = overview.get("clip_style", 0)
    lpips = overview.get("content_lpips", 0)
    t_clip = transfer.get("clip_style", 0)
    t_lpips = transfer.get("content_lpips", 0)
    results.append({
        "name": d.name,
        "allpairs_clip": clip,
        "allpairs_lpips": lpips,
        "transfer_clip": t_clip,
        "transfer_lpips": t_lpips,
    })
    print(f"{d.name}: allpairs clip={clip:.4f} lpips={lpips:.4f} | transfer clip={t_clip:.4f} lpips={t_lpips:.4f}")

print("\n" + "=" * 80)
print("SUMMARY (all_pairs_overview)")
print(f"{'Experiment':<20} {'Alpha':<8} {'CLIP-S':<12} {'LPIPS':<12} {'1-LPIPS':<12}")
print("-" * 80)
for r in results:
    alpha_str = r["name"].replace("ensemble_a", "")
    alpha = int(alpha_str) / 10.0
    print(f"{r['name']:<20} {alpha:<8.1f} {r['allpairs_clip']:<12.4f} {r['allpairs_lpips']:<12.4f} {1-r['allpairs_lpips']:<12.4f}")

print("\n" + "=" * 80)
print("BASELINES:")
print(f"{'T11 (8-step)':<20} {'':<8} {0.7213:<12.4f} {0.2868:<12.4f} {1-0.2868:<12.4f}")
print(f"{'SaMam':<20} {'':<8} {0.7175:<12.4f} {0.2423:<12.4f} {1-0.2423:<12.4f}")
print(f"{'SeeDream':<20} {'':<8} {0.7198:<12.4f} {0.4767:<12.4f} {1-0.4767:<12.4f}")
