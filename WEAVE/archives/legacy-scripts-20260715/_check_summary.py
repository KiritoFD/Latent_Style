"""Check summary.json format."""
import json
path = "exp/ablation_v2/a01_wo_endpoint_adain/eval/summary.json"
with open(path) as f:
    d = json.load(f)
print("Top keys:", list(d.keys())[:20])
agg = d.get("aggregate", d.get("summary", {}))
print("Agg keys:", list(agg.keys())[:20])
for k in ["clip_s", "lpips", "clip_style", "content_lpips"]:
    if k in agg:
        v = agg[k]
        print(f"  {k} = {v} (type={type(v).__name__})")
# Check if metrics are in a different structure
if "metrics" in d:
    print("Metrics:", list(d["metrics"].keys())[:20])
# Print first few items
for k, v in list(d.items())[:10]:
    if isinstance(v, dict):
        print(f"  {k}: {list(v.keys())[:10]}")
    else:
        print(f"  {k}: {v}")
