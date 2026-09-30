"""Read summary.json and print key metrics + timings."""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
d = json.load(open(path, encoding="utf-8"))

timings = d.get("timings_sec", {}) or {}
wall = timings.get("wall_total", 0)
print(f"WALL_TOTAL: {wall:.3f}s")

# Sort timings by value descending
sorted_t = sorted(timings.items(), key=lambda x: -float(x[1]))
print("\nTIMINGS (top 10):")
for k, v in sorted_t[:10]:
    pct = 100 * float(v) / wall if wall > 0 else 0
    print(f"  {k:<30} {float(v):>8.3f}s  ({pct:>5.1f}%)")

# Metrics
analysis = d.get("analysis", {}) or {}
overview = analysis.get("all_pairs_overview", {}) or {}
print(f"\nCLIP_S: {overview.get('clip_style')}")
print(f"LPIPS:  {overview.get('content_lpips')}")

settings = d.get("settings", {}) or {}
print(f"\nbatch_size: {settings.get('batch_size')}")
print(f"generation_batch_size: {settings.get('generation_batch_size')}")
print(f"target_chunk_size: {settings.get('target_chunk_size')}")
print(f"vae_decode_batch_size: {settings.get('vae_decode_batch_size')}")
print(f"vae_compile_decoder: {settings.get('vae_compile_decoder')}")
print(f"vae_compile_mode: {settings.get('vae_compile_mode')}")
