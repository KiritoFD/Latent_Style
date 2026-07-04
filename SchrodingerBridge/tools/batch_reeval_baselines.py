#!/usr/bin/env python3
"""
Batch unified evaluation for all external baseline images.
Copies baseline images into the format expected by run_evaluation.py,
then calls it with --reuse_generated for each baseline.

Usage (local GPU):
  python tools/batch_reeval_baselines.py --test_dir G:\GitHub\Latent_Style\Dataset\distinct5_512\test --output_root G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_reeval --device cuda
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

STYLE_NAMES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
STYLE_SUBDIRS = ",".join(STYLE_NAMES)

BASELINE_SOURCES = {
    "identity": {"path": r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_images\identity", "method": "Identity"},
    "samam_diag_step2250": {"path": r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_images\samam_diag_2250", "method": "SaMAM-diag-2250"},
    "samam_diag_step3000": {"path": r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_images\samam_diag_3000", "method": "SaMAM-diag-3000"},
    "samam_latent_step1000": {"path": r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_images\samam_latent_1000", "method": "SaMAM-latent-1000"},
    "samam_latent_step600": {"path": r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_images\samam_latent_600", "method": "SaMAM-latent-600"},
    "samam_latent_step300": {"path": r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_images\samam_latent_300", "method": "SaMAM-latent-300"},
    "samst_stepalign40": {"path": r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_images\samst_40", "method": "SaMST-40-diag"},
    "sdedit_010": {"path": r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_images\sdedit_str0.10", "method": "SDEdit-str0.10"},
    "sdedit_020": {"path": r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_images\sdedit_str0.20", "method": "SDEdit-str0.20"},
    "sdedit_035": {"path": r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_images\sdedit_str0.35", "method": "SDEdit-str0.35"},
    "sdedit_040": {"path": r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_images\sdedit_str0.40", "method": "SDEdit-str0.40"},
    "sdturbo": {"path": r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_images\sdturbo", "method": "SD-Turbo"},
    "styleid": {"path": r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_images\styleid", "method": "StyleID"},
    "s2wat": {"path": r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_images\s2wat", "method": "S2WAT"},
    "adain_v32k": {"path": r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_images\adain_v32k", "method": "AdaIN-v32k"},
    "adain_vgg19": {"path": r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_images\adain_vgg19", "method": "AdaIN-vgg19"},
    "adain_bad": {"path": r"G:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_images\adain_bad", "method": "AdaIN-bad"},
}

EVAL_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "src", "utils", "run_evaluation.py")


def normalize_filename(filename):
    """Convert baseline image filename to our standard __to__ format."""
    stem = Path(filename).stem
    ext = Path(filename).suffix

    # Already in __to__ format – ensure src_stem includes style prefix
    # so it matches the test image stem in src_lookup.
    if "__to__" in stem:
        left, tgt_style = stem.rsplit("__to__", 1)
        if "__" in left:
            src_style, src_name = left.split("__", 1)
            if src_style in STYLE_NAMES and tgt_style in STYLE_NAMES:
                # Check if src_name already starts with a style prefix
                has_style_prefix = any(src_name.startswith(s + "__") for s in STYLE_NAMES)
                if has_style_prefix:
                    # src_name is already a full test-file stem (Style__artist_title)
                    return f"{src_style}__{src_name}__to__{tgt_style}{ext}"
                # Re-embed style prefix in src_stem for src_lookup compatibility
                full_src_stem = f"{src_style}__{src_name}"
                return f"{src_style}__{full_src_stem}__to__{tgt_style}{ext}"
        return filename

    # Handle _to_ format
    if "_to_" in stem:
        left, tgt_style = stem.rsplit("_to_", 1)
        for src_style in sorted(STYLE_NAMES, key=lambda x: len(x), reverse=True):
            if left.startswith(src_style + "_"):
                src_stem = left[len(src_style) + 1:]
                # If src_stem already starts with "Style__", it already has the full test-file stem
                has_style_prefix = any(src_stem.startswith(s + "__") for s in STYLE_NAMES)
                if has_style_prefix:
                    return f"{src_style}__{src_stem}__to__{tgt_style}{ext}"
                full_src_stem = f"{src_style}__{src_stem}"
                return f"{src_style}__{full_src_stem}__to__{tgt_style}{ext}"

    # Handle directory-based format: tgt_style/filename.png
    # This is handled at the directory level
    return filename


def prepare_images(src_dir, dst_images_dir):
    """Copy and rename images from source to destination with standard naming."""
    src_dir = Path(src_dir)
    dst_images_dir = Path(dst_images_dir)
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    
    # Check for subdirectory structure (tgt_style/*.png)
    subdirs = [d for d in src_dir.iterdir() if d.is_dir()]
    if subdirs:
        for subdir in subdirs:
            tgt_style = subdir.name
            if tgt_style not in STYLE_NAMES:
                continue
            for img_path in sorted(subdir.glob("*.png")):
                # Parse filename to get src_style
                stem = img_path.stem
                src_style = None
                src_name = stem
                
                if "__" in stem:
                    parts = stem.split("__", 1)
                    if parts[0] in STYLE_NAMES:
                        src_style = parts[0]
                        src_name = parts[1]
                    else:
                        # Some files have format: Style__Style__artist_title
                        for s in STYLE_NAMES:
                            if stem.startswith(s + "__"):
                                src_style = s
                                src_name = stem[len(s) + 2:]
                                break
                
                if src_style is None:
                    # Try to infer from filename
                    for s in STYLE_NAMES:
                        if stem.startswith(s + "__") or stem.startswith(s + "_"):
                            src_style = s
                            src_name = stem[len(s) + 2:] if "__" in stem else stem[len(s) + 1:]
                            break
                
                if src_style and src_style != tgt_style:
                    # Keep full test-file stem (Style__artist_title) so _parse_generated_name
                    # produces src_stem matching the test image stem in src_lookup.
                    full_src_stem = f"{src_style}__{src_name}"
                    new_name = f"{src_style}__{full_src_stem}__to__{tgt_style}.png"
                    dst = dst_images_dir / new_name
                    if not dst.exists():
                        shutil.copy2(img_path, dst)
                    count += 1
    else:
        # Flat structure - just copy with normalization
        for img_path in sorted(src_dir.glob("*.png")):
            new_name = normalize_filename(img_path.name)
            dst = dst_images_dir / new_name
            if not dst.exists():
                shutil.copy2(img_path, dst)
            count += 1
    
    return count


def run_eval(eval_dir, test_dir, device="cuda"):
    """Run run_evaluation.py with --reuse_generated."""
    cmd = [
        sys.executable, EVAL_SCRIPT,
        str(eval_dir),
        "--reuse_generated",
        "--save_generated_images",
        f"--style_subdirs={STYLE_SUBDIRS}",
        f"--test_dir={test_dir}",
        "--eval_only_lpips_clip_style",
        "--eval_lpips_net", "alex",
        "--clip_style_idt_baseline", "0.6399",
    ]
    
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    
    if result.returncode != 0:
        print(f"ERROR: {result.stderr[-2000:]}")
        return None
    
    # Read the summary
    summary_path = Path(eval_dir) / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", required=True, help="Path to test/reference images root")
    parser.add_argument("--output_root", required=True, help="Root output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method", default=None, help="Only evaluate this method")
    args = parser.parse_args()
    
    results = {}
    
    for name, info in BASELINE_SOURCES.items():
        if args.method and name != args.method:
            continue
        
        src_path = info["path"]
        method = info["method"]
        
        if not os.path.isdir(src_path):
            print(f"\nSKIP {name}: source not found at {src_path}")
            continue
        
        eval_dir = os.path.join(args.output_root, name)
        images_dir = os.path.join(eval_dir, "images")
        
        # Step 1: Prepare images
        print(f"\nPreparing images for {name}...")
        n = prepare_images(src_path, images_dir)
        print(f"  Prepared {n} images")
        
        if n == 0:
            print(f"  SKIP: no images found")
            continue
        
        # Step 2: Run evaluation
        summary = run_eval(eval_dir, args.test_dir, args.device)
        
        if summary:
            overview = summary.get("analysis", {}).get("all_pairs_overview", {})
            cs = overview.get("clip_style", 0)
            lp = overview.get("content_lpips", 0)
            delta = overview.get("clip_s_delta_idt", 0)
            results[name] = {
                "method": method,
                "clip_style": cs,
                "content_lpips": lp,
                "one_minus_lpips": round(1 - lp, 4) if lp else None,
                "clip_s_delta_idt": delta,
                "n_images": n,
            }
            print(f"  {method}: clip_style={cs:.4f}, lpips={lp:.4f}, delta_idt={delta:+.4f}")
        else:
            print(f"  {method}: evaluation failed")
    
    # Save results
    results_path = os.path.join(args.output_root, "unified_eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    
    # Print comparison table
    print(f"\n{'='*70}")
    print(f"{'Method':30s} {'CLIP-S':>8s} {'LPIPS':>8s} {'1-LPIPS':>8s} {'delta_idt':>10s}")
    print(f"{'='*70}")
    for name, r in sorted(results.items(), key=lambda x: x[1].get("clip_style", 0), reverse=True):
        print(f"{r['method']:30s} {r['clip_style']:8.4f} {r['content_lpips']:8.4f} {r.get('one_minus_lpips',0):8.4f} {r.get('clip_s_delta_idt',0):+10.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
