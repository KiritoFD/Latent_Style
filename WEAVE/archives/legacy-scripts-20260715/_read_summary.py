"""Read eval summary.json and print key metrics."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else r"I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\target_hf_subband_ft6\full_eval\adain20\summary.json"
with open(path, "r") as f:
    data = json.load(f)

print("TOP KEYS:", list(data.keys()))

# Extract CLIP-S and LPIPS from all_pairs_overview
apo = data.get("all_pairs_overview", {})
if apo:
    print("\n=== all_pairs_overview keys ===")
    print(list(apo.keys()))
    for k in ["clip_style", "clip_style_mean", "content_lpips", "content_lpips_mean"]:
        if k in apo:
            print(f"  {k} = {apo[k]}")
    # Try nested
    if "clip_style" in apo and isinstance(apo["clip_style"], dict):
        print(f"  clip_style dict: {apo['clip_style']}")
    if "content_lpips" in apo and isinstance(apo["content_lpips"], dict):
        print(f"  content_lpips dict: {apo['content_lpips']}")

# Print matrix_breakdown and analysis
for key in ["matrix_breakdown", "analysis", "idt_baselines", "settings"]:
    if key in data:
        print(f"\n=== {key} ===")
        print(json.dumps(data[key], indent=2)[:3000])

