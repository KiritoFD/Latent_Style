"""Extract key metrics from summary.json."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else r"I:\Github\Latent_Style\SchrodingerBridge\exp\t1_asg_5ep\full_eval\epoch_0005\summary.json"
with open(path, encoding="utf-8") as f:
    data = json.load(f)

# Print top-level keys
print("=== TOP-LEVEL KEYS ===")
for k in data.keys():
    v = data[k]
    if isinstance(v, dict):
        print(f"  {k}: dict with {len(v)} keys")
    elif isinstance(v, list):
        print(f"  {k}: list with {len(v)} items")
    else:
        print(f"  {k}: {v}")

# Look for metrics in common locations
print("\n=== METRICS ===")
for key in ["metrics", "aggregate_metrics", "scores", "results", "evaluation"]:
    if key in data:
        print(f"\n--- {key} ---")
        metrics = data[key]
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v}")
                elif isinstance(v, dict) and "mean" in v:
                    print(f"  {k}: mean={v['mean']}")
                elif isinstance(v, dict):
                    for k2, v2 in v.items():
                        if isinstance(v2, (int, float)):
                            print(f"  {k}.{k2}: {v2}")

# Search recursively for clip/lpips/dino keys
print("\n=== SEARCH RESULTS ===")
def search(obj, prefix="", depth=0):
    if depth > 3:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = k.lower()
            if any(t in kl for t in ["clip", "lpips", "dino", "style_sim", "content_sim", "musiq"]):
                if isinstance(v, (int, float)):
                    print(f"  {prefix}{k}: {v}")
                elif isinstance(v, dict) and "mean" in v:
                    print(f"  {prefix}{k}: mean={v['mean']}")
                elif isinstance(v, dict):
                    search(v, prefix=f"{prefix}{k}.", depth=depth+1)
            elif isinstance(v, dict):
                search(v, prefix=f"{prefix}{k}.", depth=depth+1)
    elif isinstance(obj, list) and len(obj) > 0:
        search(obj[0], prefix=f"{prefix}[0].", depth=depth+1)

search(data)
