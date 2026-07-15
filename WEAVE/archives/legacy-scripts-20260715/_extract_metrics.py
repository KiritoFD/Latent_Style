"""Extract CLIP-S and LPIPS from summary.json."""
import json
import sys

path = sys.argv[1]
with open(path) as f:
    data = json.load(f)

apo = data.get("analysis", {}).get("all_pairs_overview", {})
print(f"clip_style={apo.get('clip_style', 'N/A')}")
print(f"content_lpips={apo.get('content_lpips', 'N/A')}")
