#!/usr/bin/env python3
"""
Generate identity baseline images and run evaluation.

Identity baseline = copy source image as-is for every (src, tgt) pair.
This measures the floor/ceiling of metrics when no style transfer occurs.

Usage (on remote server):
  python I:\GitHub\Latent_Style\SchrodingerBridge\tools\gen_identity_baseline_remote.py
"""
import os
import shutil
import subprocess
import sys

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
TEST_DIR = r"I:\wikiart_distinct5_samam_512_classview\test"
EVAL_ROOT = r"I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_reeval"
EVAL_DIR = os.path.join(EVAL_ROOT, "identity_baseline")
IMAGES_DIR = os.path.join(EVAL_DIR, "images")
STYLE_SUBDIRS = ",".join(STYLES)
EVAL_SCRIPT = r"I:\GitHub\Latent_Style\SchrodingerBridge\src\utils\run_evaluation.py"


def generate_identity_images():
    """Copy source images into identity baseline directory."""
    os.makedirs(IMAGES_DIR, exist_ok=True)
    count = 0

    for src_style in STYLES:
        src_dir = os.path.join(TEST_DIR, src_style)
        if not os.path.isdir(src_dir):
            print(f"  WARNING: {src_dir} not found, skipping")
            continue

        files = sorted(
            f for f in os.listdir(src_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        )
        print(f"  {src_style}: {len(files)} source images")

        for fname in files:
            src_stem = os.path.splitext(fname)[0]
            src_path = os.path.join(src_dir, fname)

            for tgt_style in STYLES:
                out_name = f"{src_style}__{src_stem}__to__{tgt_style}.png"
                dst_path = os.path.join(IMAGES_DIR, out_name)
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                count += 1

    print(f"\nTotal identity images: {count}")
    return count


def run_evaluation():
    """Run run_evaluation.py on the identity baseline."""
    cmd = [
        sys.executable, EVAL_SCRIPT,
        EVAL_DIR,
        "--reuse_generated",
        "--save_generated_images",
        f"--style_subdirs={STYLE_SUBDIRS}",
        f"--test_dir={TEST_DIR}",
        "--eval_only_lpips_clip_style",
        "--clip_style_idt_baseline", "0.6399",
    ]

    print(f"\n{'=' * 60}")
    print(f"Running evaluation...")
    print(f"  eval_dir:  {EVAL_DIR}")
    print(f"  images:    {IMAGES_DIR}")
    print(f"  test_dir:  {TEST_DIR}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, text=True)
    return result.returncode


def main():
    print("Step 1: Generate identity baseline images")
    count = generate_identity_images()

    if count == 0:
        print("ERROR: No images generated. Aborting.")
        sys.exit(1)

    print(f"\nStep 2: Run evaluation on {count} identity images")
    rc = run_evaluation()

    if rc == 0:
        print("\nEvaluation complete!")
        summary_path = os.path.join(EVAL_DIR, "summary.json")
        if os.path.exists(summary_path):
            import json
            with open(summary_path) as f:
                summary = json.load(f)
            analysis = summary.get("analysis", {})
            overview = analysis.get("all_pairs_overview", {})
            transfer = analysis.get("style_transfer_ability", {})
            identity = analysis.get("identity_reconstruction", {})
            print(f"\n{'=' * 50}")
            print("Identity Baseline Results:")
            print(f"{'=' * 50}")
            print(f"  All pairs:   CLIP-S={overview.get('clip_style', 0):.4f}  LPIPS={overview.get('content_lpips', 0):.4f}")
            print(f"  Transfer:    CLIP-S={transfer.get('clip_style', 0):.4f}  LPIPS={transfer.get('content_lpips', 0):.4f}")
            print(f"  Identity:    CLIP-S={identity.get('clip_style', 0):.4f}  LPIPS={identity.get('content_lpips', 0):.4f}")
    else:
        print(f"\nEvaluation failed with exit code {rc}")
        sys.exit(rc)


if __name__ == "__main__":
    main()
