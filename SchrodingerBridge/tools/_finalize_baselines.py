#!/usr/bin/env python3
"""Run WFI on existing images for baseline experiments (film_formal, intrinsic_v2).
Also collect all results and update comparison table."""
import json, os, sys

base = "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge"
repo = "/mnt/i/Github/Latent_Style/SchrodingerBridge"

# 1. Run WFI on existing full_eval images for baselines
sys.path.insert(0, os.path.join(repo, "src"))
from utils.wfi import wfi_benchmark

print("=== Running WFI on baseline experiments ===")
for exp, epoch in [("620_film_formal", "epoch_0008"), ("620_intrinsic_v2", "epoch_0008")]:
    eval_dir = os.path.join(base, exp, "full_eval", epoch)
    images_dir = os.path.join(eval_dir, "images")
    if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0:
        print(f"  {exp}/{epoch}: {len(os.listdir(images_dir))} images found")
        wfi_benchmark(eval_dir, source_image_dir=None)
    else:
        print(f"  {exp}/{epoch}: NO images (need re-eval)")
        # Need to run eval with images
        import subprocess
        test_dir = "/mnt/i/wikiart_distinct5_samam_512_classview/test"
        cache_dir = "/mnt/i/Github/Latent_Style/eval_cache"
        cmd = [
            sys.executable,
            os.path.join(repo, "tools/run_eval_with_wfi.py"),
            "--checkpoint", os.path.join(base, exp, f"{epoch}.pt"),
            "--output", os.path.join(base, exp, "full_eval_wfi", epoch),
            "--test-dir", test_dir,
            "--cache-dir", cache_dir,
            "--clip-hf-cache-dir", os.path.join(cache_dir, "hf"),
            "--batch-size", "4",
            "--target-chunk-size", "2",
            "--vae-decode-batch-size", "8",
            "--num-steps", "8",
        ]
        print(f"    Launching subprocess...")
        subprocess.run(cmd)

# 2. Update comparison table
print("\n\n=== Final WFI/CLIP/LPIPS Comparison ===")
experiments = [
    ("620_intrinsic_v2", "epoch_0008", "Baseline (no FiLM)"),
    ("620_film_formal", "epoch_0008", "Early FiLM"),
    ("620_film_gate03_5ep", "epoch_0005", "Post-FiLM + gate=0.3"),
    ("620_film_v2_5ep", "epoch_0005", "Pre+Post FiLM"),
    ("620_film_v4_gated_5ep", "epoch_0005", "Gated attn (zero init)"),
]

print(f"{'Experiment':<30} {'Desc':<25} {'WFI':>7} {'Contrast':>8} {'Sat':>7} {'Clip-S':>7} {'LPIPS':>7} {'gamma':>7}")
print("-" * 100)

for exp, epoch, desc in experiments:
    # Check multiple dirs for WFI
    for d in [os.path.join(base, exp, "full_eval_wfi", epoch), os.path.join(base, exp, "full_eval", epoch)]:
        wfi_json = os.path.join(d, "wfi_benchmark.json")
        sj = os.path.join(d, "summary.json")
        if os.path.exists(wfi_json):
            w = json.load(open(wfi_json))
            gen = w.get("generated_wfi", {})
            wfi = gen.get("wfi_score", {}).get("mean")
            cr = gen.get("contrast_ratio", {}).get("mean")
            sat = gen.get("saturation_mean", {}).get("mean")
        else:
            wfi = cr = sat = None
        if os.path.exists(sj):
            s = json.load(open(sj))
            ap = s.get("analysis", {}).get("all_pairs_overview", {})
            cs = ap.get("clip_style")
            lp = ap.get("content_lpips")
            ro = s.get("runtime_observability", {})
            ap_ro = ro.get("all_pairs_overview", {}) if ro else {}
            gamma = ap_ro.get("model_film_gamma_abs", "N/A")
        else:
            cs = lp = gamma = None
        if wfi is not None or cs is not None:
            break

    wfi_s = f"{wfi:.4f}" if wfi else "N/A"
    cr_s = f"{cr:.3f}" if cr else "N/A"
    sat_s = f"{sat:.4f}" if sat else "N/A"
    cs_s = f"{cs:.4f}" if cs else "N/A"
    lp_s = f"{lp:.4f}" if lp else "N/A"
    g_s = f"{gamma:.4f}" if isinstance(gamma, float) else "N/A"
    print(f"{exp:<30} {desc:<25} {wfi_s:>7} {cr_s:>8} {sat_s:>7} {cs_s:>7} {lp_s:>7} {g_s:>7}")