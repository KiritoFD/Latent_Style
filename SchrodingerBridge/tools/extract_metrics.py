"""Extract key metrics from evaluation summary.json."""
import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

summary_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r'I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_v2\eval\cut\summary.json')

if not summary_path.exists():
    print(f"ERROR: {summary_path} not found")
    sys.exit(1)

with open(summary_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Try different possible key structures
print("=" * 60)
print(f"Summary: {summary_path}")
print("=" * 60)

# Print top-level keys
print(f"\nTop-level keys: {list(data.keys())}")

# Look for clip_style and content_lpips in various locations
def find_metrics(obj, path=""):
    results = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            if k in ('clip_style', 'content_lpips', 'clip_s_delta_idt', 'n_pairs', 'lpips_mean', 'clip_style_mean'):
                results.append((new_path, v))
            elif isinstance(v, (dict, list)):
                results.extend(find_metrics(v, new_path))
    elif isinstance(obj, list) and obj:
        # Only check first item
        results.extend(find_metrics(obj[0], f"{path}[0]"))
    return results

metrics = find_metrics(data)
if metrics:
    print("\nFound metrics:")
    for path, val in metrics:
        print(f"  {path} = {val}")
else:
    print("\nNo standard metrics found. Printing all top-level values:")
    for k, v in data.items():
        if not isinstance(v, (dict, list)):
            print(f"  {k} = {v}")
        elif isinstance(v, dict):
            print(f"  {k}: dict with keys {list(v.keys())[:10]}")
        elif isinstance(v, list):
            print(f"  {k}: list with {len(v)} items")

# Also look for 'all_pairs_overview' or similar aggregate keys
for key in ['all_pairs_overview', 'overview', 'aggregate', 'metrics', 'results']:
    if key in data:
        print(f"\n{key}:")
        agg = data[key]
        if isinstance(agg, dict):
            for k, v in agg.items():
                if not isinstance(v, (dict, list)):
                    print(f"  {k} = {v}")
                elif isinstance(v, dict):
                    # One level deeper
                    for k2, v2 in v.items():
                        if not isinstance(v2, (dict, list)):
                            print(f"  {k}.{k2} = {v2}")
                        elif isinstance(v2, dict) and 'mean' in v2:
                            print(f"  {k}.{k2}.mean = {v2['mean']}")

print("\n==EXTRACT_DONE==")
