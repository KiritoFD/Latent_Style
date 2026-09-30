import json
import os

base = r"G:\GitHub\Latent_Style\SchrodingerBridge\exp"
results = []

for d in os.listdir(base):
    exp_dir = os.path.join(base, d)
    if not os.path.isdir(exp_dir):
        continue
    for sub in ["full_eval", "quick_eval"]:
        fe_dir = os.path.join(exp_dir, sub)
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
                results.append((d, sub, ep, clip, lpips))
            except Exception:
                pass

results.sort(key=lambda x: (x[3], -x[4]))
print(f"Total results: {len(results)}")
print(f"{'exp':40s} {'type':12s} {'epoch':12s} {'clip':>8s} {'lpips':>8s}")
print("-" * 85)
for name, sub, ep, clip, lpips in results:
    print(f"{name:40s} {sub:12s} {ep:12s} {clip:8.4f} {lpips:8.4f}")
