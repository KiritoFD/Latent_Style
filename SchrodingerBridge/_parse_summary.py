"""Parse summary.json and extract pool-level metrics."""
import json
import sys

path = r"I:\Github\Latent_Style\SchrodingerBridge\exp\clean_base_v2\full_eval\epoch_0010\summary.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=== Top-level keys ===")
for k in data.keys():
    v = data[k]
    if isinstance(v, dict):
        print(f"  {k}: dict({len(v)} keys)")
    elif isinstance(v, list):
        print(f"  {k}: list({len(v)} items)")
    else:
        print(f"  {k}: {v}")

print()
print("=== Looking for pool/overall metrics ===")
# Check common pool-level keys
for key in ["pool", "overall", "summary", "aggregate", "global", "pool_metrics"]:
    if key in data:
        print(f"\n--- {key} ---")
        pool = data[key]
        if isinstance(pool, dict):
            for k, v in pool.items():
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v}")

# Check if there's a per_style list with a pool entry
if "per_style" in data:
    print("\n--- per_style (first 3 + last) ---")
    ps = data["per_style"]
    if isinstance(ps, list):
        for item in ps[:3]:
            print(f"  {item.get('target_style', '?')}: clip_style={item.get('clip_style', '?')} content_lpips={item.get('content_lpips', '?')}")
        if len(ps) > 3:
            print(f"  ... ({len(ps)} total)")
            print(f"  {ps[-1].get('target_style', '?')}: clip_style={ps[-1].get('clip_style', '?')} content_lpips={ps[-1].get('content_lpips', '?')}")
    elif isinstance(ps, dict):
        for k, v in list(ps.items())[:5]:
            if isinstance(v, dict):
                print(f"  {k}: clip_style={v.get('clip_style', '?')} content_lpips={v.get('content_lpips', '?')}")

# Print all numeric top-level fields that look like metrics
print("\n=== All numeric top-level fields ===")
for k, v in data.items():
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        print(f"  {k}: {v}")

# Check rows for pool-level
if "rows" in data:
    rows = data["rows"]
    if isinstance(rows, list):
        print(f"\n=== rows: {len(rows)} items ===")
        # Check last row (often pool-level)
        if rows:
            last = rows[-1]
            print(f"Last row keys: {list(last.keys())[:10]}")
            for k in ["target_style", "clip_style", "content_lpips", "lpips", "clip_content", "clip_dir"]:
                if k in last:
                    print(f"  {k}: {last[k]}")

# Check clip_style_global and content_lpips
print("\n=== Key global metrics ===")
for k in ["clip_style_global", "clip_style", "content_lpips", "lpips", "clip_content", "clip_dir",
          "metric_lpips", "lpips_alex", "fid", "delta_fid"]:
    if k in data:
        print(f"  {k}: {data[k]}")
