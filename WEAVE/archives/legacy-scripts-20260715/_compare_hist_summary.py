"""Compare historical full_eval summary with repro to find the discrepancy."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

hist_path = "g:/GitHub/Latent_Style/SchrodingerBridge/exp/630_random20_heun_5ep/full_eval/epoch_0005/summary.json"
repro_path = "g:/GitHub/Latent_Style/SchrodingerBridge/exp/630_random20_heun_5ep/repro_d5_local/epoch_0005/summary.json"

for label, path in [("HISTORICAL", hist_path), ("REPRO", repro_path)]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"\n=== {label}: {path} ===")
    apo = data.get("analysis", {}).get("all_pairs_overview", {})
    print(f"  clip_style: {apo.get('clip_style', 'N/A')}")
    print(f"  clip_content: {apo.get('clip_content', 'N/A')}")
    print(f"  clip_dir: {apo.get('clip_dir', 'N/A')}")
    print(f"  content_lpips: {apo.get('content_lpips', 'N/A')}")
    print(f"  clip_s_delta_idt: {apo.get('clip_s_delta_idt', 'N/A')}")

    # Check settings for test_dir, style mapping
    settings = data.get("settings", {})
    for k, v in settings.items():
        kl = k.lower()
        if "test" in kl or "style" in kl or "target" in kl or "num_style" in kl or "source" in kl:
            print(f"  setting {k}: {v}")

    # Check checkpoint info
    ckpt = data.get("checkpoint", {})
    print(f"  checkpoint: {ckpt}")

    # matrix_breakdown keys
    mb = data.get("matrix_breakdown", {})
    print(f"  matrix_breakdown styles: {list(mb.keys())}")
    # Show first style's keys
    if mb:
        first_style = list(mb.keys())[0]
        first_info = mb[first_style]
        if isinstance(first_info, dict):
            print(f"  {first_style} keys: {list(first_info.keys())[:10]}")
            # Look for clip_style in nested
            for k, v in first_info.items():
                if "clip" in k.lower() or "lpips" in k.lower():
                    print(f"    {k}: {v}")
