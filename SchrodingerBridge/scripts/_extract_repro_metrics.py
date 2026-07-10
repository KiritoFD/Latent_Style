"""Extract clip_style and lpips from repro summary."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

path = sys.argv[1] if len(sys.argv) > 1 else "g:/GitHub/Latent_Style/SchrodingerBridge/exp/630_random20_heun_5ep/repro_d5_local/epoch_0005/summary.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

mn = data.get("metrics_note", {})
apo = data.get("analysis", {}).get("all_pairs_overview", {})

print("=== metrics_note ===")
for k in ["clip_style", "clip_content", "clip_dir", "clip_t", "fid", "delta_fid"]:
    v = mn.get(k, "N/A")
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

print("\n=== all_pairs_overview ===")
for k, v in apo.items():
    if isinstance(v, (int, float)):
        print(f"  {k}: {v}")
    elif isinstance(v, dict):
        print(f"  {k}: {v}")

print("\n=== matrix_breakdown (per style) ===")
mb = data.get("matrix_breakdown", {})
for style, info in mb.items():
    if isinstance(info, dict):
        cs = info.get("clip_style", info.get("mean_clip_style", "N/A"))
        lp = info.get("content_lpips", info.get("mean_content_lpips", "N/A"))
        print(f"  {style}: clip_style={cs}, lpips={lp}")
