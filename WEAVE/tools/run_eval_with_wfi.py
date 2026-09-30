#!/usr/bin/env python3
"""
Run evaluation with image saving + WFI benchmark on a checkpoint.

This script wraps run_evaluation.py to:
1. Run eval with --save_generated_images (generates PNG images)
2. Run WFI benchmark on the generated images
3. Collect CLIP-style, LPIPS, WFI into a unified report

Usage (on remote WSL machine):
  python3 tools/run_eval_with_wfi.py \
    --checkpoint /mnt/i/.../epoch_0005.pt \
    --output /mnt/i/.../full_eval_wfi/epoch_0005 \
    --test-dir /mnt/i/wikiart_distinct5_samam_512_classview/test \
    --cache-dir /mnt/i/Github/Latent_Style/eval_cache \
    --clip-hf-cache-dir /mnt/i/Github/Latent_Style/eval_cache/hf \
    --source-dir /mnt/i/wikiart_distinct5_samam_512_classview/test
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure src/ is in path
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))


def run_evaluation(args):
    """Run run_evaluation.py with --save_generated_images."""
    cmd = [
        sys.executable,
        str(SRC_DIR / "utils" / "run_evaluation.py"),
        "--checkpoint", args.checkpoint,
        "--output", args.output,
        "--test_dir", args.test_dir,
        "--cache_dir", args.cache_dir,
        "--clip_hf_cache_dir", args.clip_hf_cache_dir,
        "--batch_size", str(args.batch_size),
        "--target_chunk_size", str(args.target_chunk_size),
        "--vae_decode_batch_size", str(args.vae_decode_batch_size),
        "--vae_compile_method", "pt2",
        "--vae_compile_mode", "reduce-overhead",
        "--skip_diffusers_vae_when_onnx",
        "--eval_lpips_chunk_size", str(args.eval_lpips_chunk_size),
        "--postprocess_mode", "none",
        "--postprocess_strength", "0.0",
        "--postprocess_mean_strength", "1.0",
        "--postprocess_std_strength", "1.0",
        "--postprocess_ref_limit", "64",
        "--latent_postprocess_mode", "none",
        "--latent_postprocess_strength", "0.0",
        "--latent_postprocess_mean_strength", "1.0",
        "--latent_postprocess_std_strength", "1.0",
        "--latent_postprocess_ref_limit", "64",
        "--no-eval_enable_introstyle",
        "--save_generated_images",
        "--save_summary_grid",
        "--keep_generated_on_device",
        "--source_latent_cache",
        "--no-eval_enable_art_fid",
        "--no-eval_enable_kid",
        "--eval_only_lpips_clip_style",
    ]

    if args.clip_style_idt_baseline:
        cmd += ["--clip_style_idt_baseline", str(args.clip_style_idt_baseline)]

    if args.num_steps:
        cmd += ["--num_steps", str(args.num_steps)]

    if args.force_regen:
        cmd += ["--force_regen"]

    print(f"\n{'='*60}")
    print(f"Running evaluation with image saving...")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd, capture_output=not args.verbose, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"ERROR: run_evaluation.py failed with code {result.returncode}")
        if not args.verbose:
            print("STDOUT:", result.stdout[-3000:] if result.stdout else "(empty)")
            print("STDERR:", result.stderr[-3000:] if result.stderr else "(empty)")
        sys.exit(1)

    print(f"Evaluation completed in {elapsed:.1f}s")
    return elapsed


def run_wfi_benchmark(args, eval_dir):
    """Run WFI benchmark on the generated images."""
    from utils.wfi import wfi_benchmark

    print(f"\n{'='*60}")
    print(f"Running WFI benchmark...")
    print(f"{'='*60}\n")

    result = wfi_benchmark(
        eval_dir,
        image_subdir="images",
        source_image_dir=args.source_dir,
    )
    return result


def collect_metrics(eval_dir):
    """Collect CLIP-style, LPIPS, WFI from summary.json."""
    summary_path = Path(eval_dir) / "summary.json"
    if not summary_path.exists():
        print(f"WARNING: {summary_path} not found")
        return {}

    with open(summary_path) as f:
        summary = json.load(f)

    ap = summary.get("analysis", {}).get("all_pairs_overview", {})
    idt = summary.get("analysis", {}).get("identity_reconstruction", {})
    transfer = summary.get("analysis", {}).get("style_transfer_ability", {})
    wfi = summary.get("wfi_benchmark", {})

    metrics = {
        # CLIP / LPIPS
        "clip_style": ap.get("clip_style"),
        "clip_content": ap.get("clip_content"),
        "content_lpips": ap.get("content_lpips"),
        "clip_s_delta_idt": ap.get("clip_s_delta_idt"),
        # Transfer-specific
        "transfer_clip_style": transfer.get("clip_style"),
        "transfer_content_lpips": transfer.get("content_lpips"),
        # Identity-specific
        "idt_clip_style": idt.get("clip_style"),
        "idt_content_lpips": idt.get("content_lpips"),
        # WFI
        "wfi_score": wfi.get("generated_wfi", {}).get("wfi_score", {}).get("mean") if wfi else None,
        "wfi_contrast_ratio": wfi.get("generated_wfi", {}).get("contrast_ratio", {}).get("mean") if wfi else None,
        "wfi_dynamic_range": wfi.get("generated_wfi", {}).get("dynamic_range", {}).get("mean") if wfi else None,
        "wfi_saturation": wfi.get("generated_wfi", {}).get("saturation_mean", {}).get("mean") if wfi else None,
        "wfi_brightness": wfi.get("generated_wfi", {}).get("brightness_mean", {}).get("mean") if wfi else None,
        "wfi_entropy": wfi.get("generated_wfi", {}).get("hist_entropy", {}).get("mean") if wfi else None,
        "wfi_transfer_score": wfi.get("transfer_wfi", {}).get("wfi_score", {}).get("mean") if wfi else None,
        "wfi_idt_score": wfi.get("identity_wfi", {}).get("wfi_score", {}).get("mean") if wfi else None,
        "source_wfi_score": wfi.get("source_wfi", {}).get("wfi_score", {}).get("mean") if wfi else None,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run eval + WFI benchmark")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--test-dir", required=True, type=str)
    parser.add_argument("--cache-dir", default="/mnt/i/Github/Latent_Style/eval_cache", type=str)
    parser.add_argument("--clip-hf-cache-dir", default="/mnt/i/Github/Latent_Style/eval_cache/hf", type=str)
    parser.add_argument("--source-dir", default=None, type=str, help="Source images for WFI baseline")
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--target-chunk-size", default=2, type=int)
    parser.add_argument("--vae-decode-batch-size", default=16, type=int)
    parser.add_argument("--eval-lpips-chunk-size", default=4, type=int)
    parser.add_argument("--num-steps", default=None, type=int)
    parser.add_argument("--clip-style-idt-baseline", default=None, type=float)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-eval", action="store_true", help="Skip eval, only run WFI on existing images")
    parser.add_argument("--force-regen", action="store_true", help="Force regenerate evaluation outputs/metrics")
    args = parser.parse_args()

    eval_dir = Path(args.output)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Run evaluation with image saving
    if not args.skip_eval:
        run_evaluation(args)
    else:
        print("Skipping evaluation (--skip-eval)")

    # Step 2: Run WFI benchmark
    wfi_result = run_wfi_benchmark(args, str(eval_dir))

    # Step 3: Collect all metrics
    metrics = collect_metrics(str(eval_dir))
    metrics["checkpoint"] = args.checkpoint
    metrics["eval_dir"] = str(eval_dir)

    # Write combined report
    report_path = eval_dir / "wfi_eval_report.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Combined Report: {report_path}")
    print(f"{'='*60}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
