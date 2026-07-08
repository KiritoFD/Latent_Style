"""Extract CLIP-S and LPIPS from nested summary.json."""
from pathlib import Path
import json

def deep_get(d, keys):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return None
    return d

# Photo2Art-256
p2a = Path(r"I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_ipadapter\photo2art256\eval\summary.json")
if p2a.exists():
    s = json.loads(p2a.read_text())
    # Try common paths
    for path in [["aggregate", "clip_style"], ["analysis", "clip_style"], ["analysis", "clip_style", "mean"],
                 ["per_style", "__aggregate__", "clip_style"]]:
        v = deep_get(s, path)
        if v:
            print(f"P2A CLIP-S ({'->'.join(path)}): {v}")
            break
    for path in [["aggregate", "content_lpips"], ["analysis", "content_lpips"], ["analysis", "content_lpips", "mean"]]:
        v = deep_get(s, path)
        if v:
            print(f"P2A LPIPS ({'->'.join(path)}): {v}")
            break
    # Dump keys
    print(f"P2A top keys: {list(s.keys())}")
    if "analysis" in s:
        print(f"P2A analysis keys: {list(s['analysis'].keys()) if isinstance(s['analysis'], dict) else type(s['analysis'])}")
    if "matrix_breakdown" in s:
        mb = s["matrix_breakdown"]
        if isinstance(mb, dict):
            print(f"P2A matrix_breakdown keys: {list(mb.keys())[:10]}")
    # Just dump first 500 chars
    print(f"P2A snippet: {str(s)[:500]}")

# Random5: debug why eval fails
print("\n--- Random5 debug ---")
r5_eval = Path(r"I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_ipadapter\random5\eval")
imgs = list((r5_eval / "images").glob("*.png"))
print(f"Images in eval/images/: {len(imgs)}")
if imgs:
    print(f"  Sample: {imgs[0].name}")
