"""Inspect 710 B0 summary.json nested structure."""
import json
from pathlib import Path

summary_path = Path(
    "/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/710_b0_t11/full_eval/epoch_0005/summary.json"
)
d = json.load(open(summary_path))

def walk(obj, prefix="", depth=0, max_depth=4):
    if depth > max_depth:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and depth < max_depth:
                print(f"{'  '*depth}{prefix}{k}:")
                walk(v, "", depth+1, max_depth)
            else:
                vs = str(v)
                if len(vs) > 100:
                    vs = vs[:100] + "..."
                print(f"{'  '*depth}{prefix}{k}: {vs}")
    elif isinstance(obj, list):
        if len(obj) > 5:
            print(f"{'  '*depth}[list of {len(obj)}], first: {obj[0]}")
        else:
            for i, v in enumerate(obj):
                print(f"{'  '*depth}[{i}]:")
                walk(v, "", depth+1, max_depth)

# Walk matrix_breakdown and analysis
print("=== matrix_breakdown ===")
walk(d.get("matrix_breakdown", {}), max_depth=3)
print("\n=== analysis ===")
walk(d.get("analysis", {}), max_depth=3)
print("\n=== timings_sec ===")
walk(d.get("timings_sec", {}), max_depth=2)
print("\n=== settings (partial) ===")
s = d.get("settings", {})
for k in ["save_generated_images", "only_lpips_clip_style", "transfer_only", "postprocess_mode"]:
    if k in s:
        print(f"  {k}: {s[k]}")
