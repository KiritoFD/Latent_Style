"""Inspect summary.json keys and find the right one for pairwise metrics."""
import json
from pathlib import Path

summary_path = Path(r"I:\Github\Latent_Style\final_works\CUT\summary.json")
data = json.loads(summary_path.read_text(encoding="utf-8"))

print("Top-level keys:", list(data.keys()))
for k in data.keys():
    v = data[k]
    if isinstance(v, dict):
        print(f"\n{k} keys (first 3): {list(v.keys())[:3]}")
        if v:
            first_key = list(v.keys())[0]
            sub = v[first_key]
            if isinstance(sub, dict):
                print(f"  {first_key} keys: {list(sub.keys())[:5]}")
