#!/usr/bin/env python3
"""
Batch run eval+WFI on all completed experiments.
Runs sequentially to avoid GPU OOM.

Usage (on remote WSL):
  python3 tools/batch_eval_wfi.py [--experiments ...] [--dry-run]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = "/mnt/i/Github/Latent_Style/SchrodingerBridge"
BASE = "/mnt/i/Github/Latent_Style/exp/620_spatial_bridge"
TEST_DIR = "/mnt/i/wikiart_distinct5_samam_512_classview/test"
CACHE_DIR = "/mnt/i/Github/Latent_Style/eval_cache"
CLIP_HF_CACHE_DIR = "/mnt/i/Github/Latent_Style/eval_cache/hf"
SOURCE_DIR = TEST_DIR  # Use test images as source baseline for WFI

# Experiments to evaluate (name, epoch)
DEFAULT_EXPERIMENTS = [
    ("620_film_v4_gated_5ep", "epoch_0005"),     # Latest: gated attention
    ("620_film_v2_5ep", "epoch_0005"),            # Pre+post FiLM
    ("620_film_gate03_5ep", "epoch_0005"),        # Post-only FiLM
    ("620_film_formal", "epoch_0008"),            # Earlier FiLM
    ("620_intrinsic_v2", "epoch_0008"),           # Baseline (no FiLM)
]


def run_eval_with_wfi(exp_name, epoch, verbose=False):
    """Run eval+WFI for a single experiment."""
    ckpt = os.path.join(BASE, exp_name, f"{epoch}.pt")
    output = os.path.join(BASE, exp_name, "full_eval_wfi", epoch)

    if not os.path.exists(ckpt):
        print(f"  SKIP: checkpoint not found: {ckpt}")
        return None

    # Check if already done
    report_path = os.path.join(output, "wfi_eval_report.json")
    if os.path.exists(report_path):
        print(f"  Already done: {report_path}")
        with open(report_path) as f:
            return json.load(f)

    cmd = [
        sys.executable,
        os.path.join(REPO, "tools/run_eval_with_wfi.py"),
        "--checkpoint", ckpt,
        "--output", output,
        "--test-dir", TEST_DIR,
        "--cache-dir", CACHE_DIR,
        "--clip-hf-cache-dir", CLIP_HF_CACHE_DIR,
        "--source-dir", SOURCE_DIR,
        "--batch-size", "4",       # Small batch to be safe
        "--target-chunk-size", "2",
        "--vae-decode-batch-size", "8",
        "--eval-lpips-chunk-size", "4",
        "--num-steps", "8",
    ]
    if verbose:
        cmd.append("--verbose")

    print(f"\n{'='*60}")
    print(f"Running eval+WFI: {exp_name} / {epoch}")
    print(f"Checkpoint: {ckpt}")
    print(f"Output: {output}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=not verbose, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.1f}s)")
        if not verbose:
            print(f"  STDERR: {result.stderr[-2000:] if result.stderr else '(empty)'}")
        return None

    print(f"  Completed in {elapsed:.1f}s")

    # Read report
    if os.path.exists(report_path):
        with open(report_path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="*", default=None,
                       help="Specific experiments to run (format: name/epoch)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.experiments:
        experiments = []
        for item in args.experiments:
            if "/" in item:
                name, epoch = item.split("/", 1)
            else:
                name = item
                # Find latest epoch
                ckpts = sorted([f for f in os.listdir(os.path.join(BASE, name)) if f.startswith("epoch_") and f.endswith(".pt")])
                epoch = ckpts[-1].replace(".pt", "") if ckpts else "epoch_0001"
            experiments.append((name, epoch))
    else:
        experiments = DEFAULT_EXPERIMENTS

    print(f"=== Batch Eval+WFI: {len(experiments)} experiments ===")
    for name, epoch in experiments:
        print(f"  - {name} / {epoch}")

    if args.dry_run:
        print("\n(dry run, exiting)")
        return

    results = {}
    for name, epoch in experiments:
        report = run_eval_with_wfi(name, epoch, verbose=args.verbose)
        if report:
            results[f"{name}/{epoch}"] = report

    # Print summary table
    print(f"\n\n{'='*80}")
    print(f"=== SUMMARY: WFI / CLIP-style / LPIPS ===")
    print(f"{'='*80}")
    print(f"{'Experiment':<30} {'WFI':>8} {'Contrast':>10} {'DynRange':>10} {'Sat':>8} {'Clip-S':>8} {'LPIPS':>8}")
    print("-" * 80)
    for key, r in results.items():
        wfi = r.get("wfi_score")
        cr = r.get("wfi_contrast_ratio")
        dr = r.get("wfi_dynamic_range")
        sat = r.get("wfi_saturation")
        cs = r.get("clip_style")
        lp = r.get("content_lpips")
        print(f"{key:<30} {wfi:>8.4f} {cr:>10.4f} {dr:>10.4f} {sat:>8.4f} {cs:>8.4f} {lp:>8.4f}")

    # Write combined report
    combined_path = os.path.join(BASE, "wfi_comparison.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCombined report: {combined_path}")


if __name__ == "__main__":
    main()
