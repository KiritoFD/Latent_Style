import json
import os
import sys

base = r"I:\Github\Latent_Style\SchrodingerBridge\exp"
results = []

for d in os.listdir(base):
    exp_dir = os.path.join(base, d)
    if not os.path.isdir(exp_dir):
        continue
    fe_dir = os.path.join(exp_dir, "full_eval")
    if not os.path.isdir(fe_dir):
        continue
    for ep in os.listdir(fe_dir):
        summary = os.path.join(fe_dir, ep, "summary.json")
        if not os.path.exists(summary):
            continue
        try:
            with open(summary, "r", encoding="utf-8") as f:
                data = json.load(f)
            apo = data.get("analysis", {}).get("all_pairs_overview", {})
            clip = apo.get("clip_style")
            lpips = apo.get("content_lpips")
            if clip is None or lpips is None:
                continue
            results.append((d, ep, clip, lpips))
        except Exception as e:
            pass

results.sort(key=lambda x: (x[2], -x[3]))
print(f"Total results: {len(results)}")
print(f"{'exp':40s} {'epoch':12s} {'clip':>8s} {'lpips':>8s}")
print("-" * 72)
for name, ep, clip, lpips in results:
    print(f"{name:40s} {ep:12s} {clip:8.4f} {lpips:8.4f}")
