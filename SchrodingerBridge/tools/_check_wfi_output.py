#!/usr/bin/env python3
"""Check film_v4_gated WFI output in detail."""
import json, os

base = "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge"
wfi_dir = os.path.join(base, "620_film_v4_gated_5ep/full_eval_wfi/epoch_0005")

print(f"=== Contents of {wfi_dir} ===")
if os.path.exists(wfi_dir):
    for f in sorted(os.listdir(wfi_dir)):
        fpath = os.path.join(wfi_dir, f)
        if os.path.isdir(fpath):
            print(f"  [DIR] {f}/ ({len(os.listdir(fpath))} files)")
        else:
            size = os.path.getsize(fpath)
            print(f"  {f} ({size} bytes)")
            if f.endswith(".json") and size < 5000:
                with open(fpath) as fh:
                    print(f"    {json.dumps(json.load(fh), indent=2)[:2000]}")
else:
    print("  Directory does not exist")

# Check if summary.json has WFI data
sj = os.path.join(wfi_dir, "summary.json")
if os.path.exists(sj):
    s = json.load(open(sj))
    wfi = s.get("wfi_benchmark")
    if wfi:
        print("\n=== WFI Benchmark Results ===")
        gen = wfi.get("generated_wfi", {})
        for k, v in gen.items():
            print(f"  {k}: {v}")
    else:
        print("\n  No wfi_benchmark in summary.json")

# Check for wfi_benchmark.json
wfi_json = os.path.join(wfi_dir, "wfi_benchmark.json")
if os.path.exists(wfi_json):
    wfi = json.load(open(wfi_json))
    print("\n=== wfi_benchmark.json ===")
    gen = wfi.get("generated_wfi", {})
    for k, v in gen.items():
        print(f"  {k}: {v}")
    print(f"\n  generated_count: {wfi.get('generated_count')}")
    print(f"  source_count: {wfi.get('source_count')}")

# Check for wfi_eval_report.json
report = os.path.join(wfi_dir, "wfi_eval_report.json")
if os.path.exists(report):
    r = json.load(open(report))
    print("\n=== wfi_eval_report.json ===")
    print(json.dumps(r, indent=2))
