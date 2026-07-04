import json, os, glob

base = "/mnt/i/Github/Latent_Style/exp"
results = []

for d in sorted(glob.glob(os.path.join(base, "aaai2027_eval_fiber_*"))):
    sj = os.path.join(d, "summary.json")
    if os.path.isfile(sj):
        try:
            data = json.load(open(sj))
            ov = data.get("analysis", {}).get("all_pairs_overview", {})
            results.append({
                "exp": os.path.basename(d),
                "clip_style": round(ov.get("clip_style", 0), 4),
                "lpips": round(ov.get("content_lpips", 0), 4),
                "checkpoint": os.path.basename(data.get("checkpoint", "")),
                "timestamp": data.get("timestamp", ""),
            })
        except Exception as e:
            print(f"ERROR:{d}:{e}", file=sys.stderr)

for r in results:
    print(f"{r['exp']}|{r['clip_style']}|{r['lpips']}|{r['checkpoint']}|{r['timestamp']}")
