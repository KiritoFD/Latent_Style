"""Extract SaMam curve data: join DINO-S (test refs) with CLIP-S/LPIPS (curve_metrics_hf.csv)."""
import csv
import json
import re
from pathlib import Path

# DINO-S from test refs
dino_data = {
    250: 0.297740,
    500: 0.222840,
    1000: 0.365621,
    2000: 0.454250,
    3000: 0.468658,
    5000: 0.475705,
    7000: 0.475826,
    20000: 0.415409,
}

# CLIP-S/LPIPS from curve_metrics_hf.csv
clip_path = Path(r"G:\GitHub\Latent_Style\WEAVE\aaai2027_v4\fig_data\curve_metrics_hf.csv")
clip_map = {}
with open(clip_path) as f:
    for row in csv.DictReader(f):
        img_dir = row["image_dir"]
        m = re.search(r"step_(\d+)", img_dir)
        if m:
            step = int(m.group(1))
            clip_map[step] = {
                "clip_s": float(row["clip_style"]),
                "lpips": float(row["content_lpips"]),
            }

# Join
print("step,dino_s,clip_s,lpips,x,y")
results = []
for step in sorted(dino_data.keys()):
    d = dino_data[step]
    c = clip_map.get(step, {})
    cs = c.get("clip_s", 0)
    lp = c.get("lpips", 0)
    x = 1.0 - lp
    y = 0.5 * (d + cs)
    print(f"{step},{d:.6f},{cs:.6f},{lp:.6f},{x:.6f},{y:.6f}")
    results.append({"step": step, "dino_s": d, "clip_s": cs, "lpips": lp})

# SaMam scatter point
print(f"\nSaMam scatter: dino_s=0.4771, clip_s=0.5816, lpips=0.2434")
print(f"SaMam scatter: x={1-0.2434:.4f}, y={0.5*(0.4771+0.5816):.4f}")
if results:
    last = results[-1]
    print(f"Curve endpoint: x={1-last['lpips']:.4f}, y={0.5*(last['dino_s']+last['clip_s']):.4f}")

# SaMST data
print("\n\n=== SaMST ===")
samst_dino = {
    5: 0.441664,
    10: 0.438931,
    15: 0.440354,
}
# From README: e5 CLIP-S=0.7276, LPIPS=0.6271; e15 CLIP-S=0.7247, LPIPS=0.6255
samst_clip = {
    5: {"clip_s": 0.7276, "lpips": 0.6271},
    15: {"clip_s": 0.7247, "lpips": 0.6255},
}
print("epoch,dino_s,clip_s,lpips,x,y")
for ep in sorted(samst_dino.keys()):
    d = samst_dino[ep]
    c = samst_clip.get(ep, {})
    cs = c.get("clip_s", 0)
    lp = c.get("lpips", 0)
    x = 1.0 - lp
    y = 0.5 * (d + cs)
    print(f"{ep},{d:.6f},{cs:.6f},{lp:.6f},{x:.6f},{y:.6f}")

print(f"\nSaMST scatter: dino_s=0.2710, clip_s=0.6183, lpips=0.7490")
print(f"SaMST scatter: x={1-0.7490:.4f}, y={0.5*(0.2710+0.6183):.4f}")
