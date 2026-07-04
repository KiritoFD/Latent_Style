#!/usr/bin/env python3
"""Check WFI report status in detail."""
import json, os, glob

base = "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge"

for exp in ["620_film_v4_gated_5ep", "620_film_v2_5ep", "620_film_gate03_5ep"]:
    wfi_dir = os.path.join(base, exp, "full_eval_wfi/epoch_0005")
    print(f"\n=== {exp} ===")
    if not os.path.exists(wfi_dir):
        print(f"  Directory does not exist")
        continue

    # List all files
    for f in sorted(os.listdir(wfi_dir)):
        fpath = os.path.join(wfi_dir, f)
        if os.path.isfile(fpath):
            print(f"  {f} ({os.path.getsize(fpath)} bytes)")
        elif os.path.isdir(fpath):
            print(f"  [DIR] {f}/ ({len(os.listdir(fpath))} files)")

    # Check wfi_eval_report.json
    report = os.path.join(wfi_dir, "wfi_eval_report.json")
    if os.path.exists(report):
        r = json.load(open(report))
        print(f"  REPORT: wfi_score={r.get('wfi_score')}, clip_style={r.get('clip_style')}, lpips={r.get('content_lpips')}")
    else:
        print(f"  NO wfi_eval_report.json")

    # Check wfi_benchmark.json
    wfi_json = os.path.join(wfi_dir, "wfi_benchmark.json")
    if os.path.exists(wfi_json):
        w = json.load(open(wfi_json))
        gen = w.get("generated_wfi", {})
        print(f"  WFI: score={gen.get('wfi_score',{}).get('mean')}, contrast={gen.get('contrast_ratio',{}).get('mean')}, sat={gen.get('saturation_mean',{}).get('mean')}")

    # Check summary.json for clip_style
    sj = os.path.join(wfi_dir, "summary.json")
    if os.path.exists(sj):
        s = json.load(open(sj))
        ap = s.get("analysis", {}).get("all_pairs_overview", {})
        print(f"  CLIP: style={ap.get('clip_style')}, lpips={ap.get('content_lpips')}")

# Check batch eval log
print("\n=== Batch eval log ===")
log = os.path.join(base, "batch_eval_wfi.log")
if os.path.exists(log):
    with open(log) as f:
        content = f.read()
    if content:
        print(content[-3000:])
    else:
        print("(empty)")
else:
    print("No log file")
