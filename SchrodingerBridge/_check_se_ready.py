"""Check if T11 baseline eval is ready and print metrics + filename matching with SaMam."""
import json
from pathlib import Path

t11_summary = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\630_local_t11_long30ep\full_eval\epoch_0001\summary.json")
t11_images = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\630_local_t11_long30ep\full_eval\epoch_0001\images")
samam_images = Path(r"I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\samam\images")

print("=" * 60)
print("T11 baseline eval summary check")
print("=" * 60)

if t11_summary.is_file():
    with t11_summary.open("r", encoding="utf-8") as f:
        data = json.load(f)
    analysis = data.get("analysis", {})
    overview = analysis.get("all_pairs_overview", {})
    transfer = analysis.get("style_transfer_ability", {})
    print(f"all_pairs_overview: clip_style={overview.get('clip_style')}, content_lpips={overview.get('content_lpips')}")
    print(f"style_transfer: clip_style={transfer.get('clip_style')}, content_lpips={transfer.get('content_lpips')}")
    print(f"generated_count: {data.get('generated_count')}")
else:
    print("summary.json NOT FOUND")

print()
t11_files = set(p.name for p in t11_images.glob("*.png")) if t11_images.is_dir() else set()
samam_files = set(p.name for p in samam_images.glob("*.png")) if samam_images.is_dir() else set()
print(f"T11 images: {len(t11_files)}")
print(f"SaMam images: {len(samam_files)}")
common = t11_files & samam_files
print(f"Common (matched): {len(common)}")
only_t11 = t11_files - samam_files
only_samam = samam_files - t11_files
print(f"Only in T11: {len(only_t11)}")
print(f"Only in SaMam: {len(only_samam)}")
if only_t11:
    print(f"  Sample T11-only: {sorted(only_t11)[:3]}")
if only_samam:
    print(f"  Sample SaMam-only: {sorted(only_samam)[:3]}")
print()
print("READY" if len(common) >= 700 else "NOT READY")
