"""
Redo inference for all baselines with correct naming: {src_style}_{stem}_to_{tgt_style}.jpg
Times each baseline and reports average seconds per image.
Writes timing results to timing_report.txt.
"""
import os
import sys
import time
import json
import shutil
import subprocess
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PIPELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent.parent
STYLE_DATA = REPO_ROOT / "style_data"
OVERFIT50 = STYLE_DATA / "overfit50"
RESULTS_DIR = PIPELINE_ROOT / "results"
PYTHON = sys.executable

CONTENT_STYLES = ["photo", "monet", "vangogh", "cezanne", "Hayao"]
TARGET_STYLES = ["monet", "vangogh", "cezanne", "Hayao"]
N_IMAGES = 30  # per content style
EXPECTED_TOTAL = len(CONTENT_STYLES) * len(TARGET_STYLES) * N_IMAGES  # 600


def prepare_content_dir(baseline):
    """Create a flat content dir with first 30 images per content style."""
    content_dir = PIPELINE_ROOT / "tmp" / f"{baseline}_content"
    content_dir.mkdir(parents=True, exist_ok=True)
    for cs in CONTENT_STYLES:
        src = OVERFIT50 / cs
        if not src.exists():
            continue
        for i, img in enumerate(sorted(src.glob("*.jpg"))[:N_IMAGES]):
            dst = content_dir / f"{cs}_{img.name}"
            if not dst.exists():
                shutil.copy2(str(img), str(dst))
    return content_dir


def clean_baseline_images(baseline):
    """Remove old images/ dir and create fresh one."""
    images_dir = RESULTS_DIR / baseline / "images"
    if images_dir.exists():
        shutil.rmtree(str(images_dir))
    images_dir.mkdir(parents=True, exist_ok=True)
    # Also clean old per-style dirs
    for s in TARGET_STYLES:
        d = RESULTS_DIR / baseline / s
        if d.exists():
            shutil.rmtree(str(d))
    return images_dir


def run_styleid(images_dir, timing_file):
    """Run StyleID inference (zero-shot)."""
    print(f"\n{'='*60}")
    print(f"[StyleID] Starting inference ({EXPECTED_TOTAL} images)")
    print(f"{'='*60}")

    from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler
    from PIL import Image
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16

    STYLE_PROMPTS = {
        "monet": "impressionist painting by Claude Monet, soft brushstrokes, water lilies",
        "vangogh": "post-impressionist painting by Vincent van Gogh, swirling bold brushstrokes",
        "cezanne": "post-impressionist painting by Paul Cezanne, geometric forms",
        "Hayao": "anime art by Hayao Miyazaki, Studio Ghibli style",
    }

    print("[StyleID] Loading SD1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=DTYPE, safety_checker=None
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()

    total_images = 0
    total_time = 0.0

    for target_style in TARGET_STYLES:
        style_prompt = STYLE_PROMPTS.get(target_style, f"painting in {target_style} style")
        style_img_path = sorted((OVERFIT50 / target_style).glob("*.jpg"))[0]
        style_img = Image.open(style_img_path).convert("RGB").resize((512, 512))

        for content_style in CONTENT_STYLES:
            content_dir = OVERFIT50 / content_style
            content_files = sorted(content_dir.glob("*.jpg"))[:N_IMAGES]

            for img_path in content_files:
                out_name = f"{content_style}_{img_path.stem}_to_{target_style}.jpg"
                out_path = images_dir / out_name
                if out_path.exists():
                    total_images += 1
                    continue

                t0 = time.time()
                try:
                    content_img = Image.open(img_path).convert("RGB").resize((512, 512))
                    with torch.no_grad():
                        result = pipe(
                            prompt=style_prompt,
                            negative_prompt="ugly, blurry, low quality, distorted",
                            image=content_img,
                            strength=0.65,
                            num_inference_steps=50,
                            guidance_scale=7.5,
                        ).images[0]
                    result.save(out_path)
                except Exception as e:
                    print(f"  [WARN] {img_path.name}: {e}")

                dt = time.time() - t0
                total_time += dt
                total_images += 1
                torch.cuda.empty_cache()

                if total_images % 50 == 0:
                    avg = total_time / total_images
                    print(f"  [{total_images}/{EXPECTED_TOTAL}] avg={avg:.2f}s/img")

    del pipe
    torch.cuda.empty_cache()

    avg_time = total_time / max(total_images, 1)
    report = f"StyleID: {total_images} images, {total_time:.1f}s total, {avg_time:.2f}s/img"
    print(f"\n{report}")
    timing_file.write(report + "\n")
    return total_images, total_time


def run_s2wat(images_dir, timing_file):
    """Run S2WAT inference."""
    print(f"\n{'='*60}")
    print(f"[S2WAT] Starting inference ({EXPECTED_TOTAL} images)")
    print(f"{'='*60}")

    S2WAT_ROOT = PIPELINE_ROOT.parent / "S2WAT-main"
    VGG_PATH = S2WAT_ROOT / "pre_trained_models" / "vgg_normalised.pth"
    content_dir = prepare_content_dir("s2wat")

    total_images = 0
    total_time = 0.0

    for target_style in TARGET_STYLES:
        ckpt_dir = PIPELINE_ROOT / "checkpoints" / "s2wat" / target_style
        candidates = list(ckpt_dir.glob("checkpoint_*_epoch.pkl"))
        if not candidates:
            print(f"  [SKIP] No S2WAT checkpoint for {target_style}")
            continue
        checkpoint = sorted(candidates)[-1]

        raw_dir = PIPELINE_ROOT / "tmp" / "s2wat_raw" / target_style
        raw_dir.mkdir(parents=True, exist_ok=True)
        for f in raw_dir.glob("*"):
            f.unlink()

        # Prepare test input
        test_input = PIPELINE_ROOT / "tmp" / "s2wat_test_input" / target_style
        (test_input / "Content").mkdir(parents=True, exist_ok=True)
        (test_input / "Style").mkdir(parents=True, exist_ok=True)
        # Copy content images
        for img in sorted(content_dir.glob("*.jpg")):
            dst = test_input / "Content" / img.name
            if not dst.exists():
                shutil.copy2(str(img), str(dst))
        # Copy style ref
        style_src = OVERFIT50 / target_style
        for img in sorted(style_src.glob("*.jpg"))[:1]:
            dst = test_input / "Style" / img.name
            if not dst.exists():
                shutil.copy2(str(img), str(dst))

        cmd = [
            PYTHON, str(S2WAT_ROOT / "test.py"),
            "--input_dir", str(test_input),
            "--output_dir", str(raw_dir),
            "--checkpoint_import_path", str(checkpoint),
        ]

        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(S2WAT_ROOT), capture_output=True)
        dt = time.time() - t0

        if result.returncode != 0:
            print(f"  [FAIL] S2WAT {target_style}: {result.stderr.decode()[-200:]}")
            continue

        # Rename outputs
        count = 0
        for f in sorted(raw_dir.glob("*.jpg")):
            name = f.stem
            if "_to_" in name:
                new_name = f.name
            else:
                new_name = f"{name}_to_{target_style}.jpg"
            dst = images_dir / new_name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))
            count += 1

        total_images += count
        total_time += dt
        avg = dt / max(count, 1)
        print(f"  S2WAT/{target_style}: {count} imgs in {dt:.1f}s ({avg:.2f}s/img)")

    avg_time = total_time / max(total_images, 1)
    report = f"S2WAT: {total_images} images, {total_time:.1f}s total, {avg_time:.2f}s/img"
    print(f"\n{report}")
    timing_file.write(report + "\n")
    return total_images, total_time


def run_samst(images_dir, timing_file):
    """Run SaMST inference."""
    print(f"\n{'='*60}")
    print(f"[SaMST] Starting inference ({EXPECTED_TOTAL} images)")
    print(f"{'='*60}")

    SAMST_REPO = PIPELINE_ROOT.parent / "SaMST-main"
    content_dir = prepare_content_dir("samst")

    total_images = 0
    total_time = 0.0

    for target_style in TARGET_STYLES:
        ckpt_dir = PIPELINE_ROOT / "checkpoints" / "samst" / target_style
        model_files = sorted(ckpt_dir.glob("epoch_*.model"))
        if not model_files:
            print(f"  [SKIP] No SaMST checkpoint for {target_style}")
            continue
        model_path = model_files[-1]

        # Prepare test input
        test_dir = SAMST_REPO / "content"
        test_dir.mkdir(parents=True, exist_ok=True)
        for img in sorted(content_dir.glob("*.jpg")):
            dst = test_dir / img.name
            if not dst.exists():
                shutil.copy2(str(img), str(dst))

        raw_output = SAMST_REPO / "outputs"
        if raw_output.exists():
            shutil.rmtree(str(raw_output))

        config = {
            "content_image_dir": str(test_dir),
            "content_scale": None,
            "output_image_dir": str(raw_output) + "/",
            "model": str(model_path),
            "style_num": 1,
            "cuda": 1,
        }
        config_path = SAMST_REPO / "test_model" / "test" / "test.yml"
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        cmd = [PYTHON, str(SAMST_REPO / "test_model" / "test" / "test.py")]

        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(SAMST_REPO / "test_model" / "test"), capture_output=True)
        dt = time.time() - t0

        if result.returncode != 0:
            print(f"  [FAIL] SaMST {target_style}: {result.stderr.decode()[-200:]}")
            continue

        # Rename: style1_{content_style}_{img}.jpg -> {content_style}_{img}_to_{target}.jpg
        count = 0
        for f in sorted(raw_output.glob("style1_*.jpg")):
            original = f.name[len("style1_"):]
            parts = original.split("_", 1)
            if len(parts) == 2:
                content_style, img_name = parts
                new_name = f"{content_style}_{img_name.replace('.jpg', '')}_to_{target_style}.jpg"
            else:
                new_name = original.replace(".jpg", f"_to_{target_style}.jpg")
            dst = images_dir / new_name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))
            count += 1

        total_images += count
        total_time += dt
        avg = dt / max(count, 1)
        print(f"  SaMST/{target_style}: {count} imgs in {dt:.1f}s ({avg:.2f}s/img)")

    avg_time = total_time / max(total_images, 1)
    report = f"SaMST: {total_images} images, {total_time:.1f}s total, {avg_time:.2f}s/img"
    print(f"\n{report}")
    timing_file.write(report + "\n")
    return total_images, total_time


def run_cut(images_dir, timing_file):
    """Copy CUT results (already trained) and rename _flip_ files."""
    print(f"\n{'='*60}")
    print(f"[CUT] Copying results (no re-inference needed)")
    print(f"{'='*60}")

    t0 = time.time()
    src_dir = RESULTS_DIR / "cut"
    count = 0

    # First check if there are existing correctly named files in per-style dirs
    for style in TARGET_STYLES:
        style_dir = src_dir / style
        if not style_dir.exists():
            continue
        for f in sorted(style_dir.glob("*.jpg")):
            name = f.name
            # Fix _flip_ naming
            name = name.replace("_flip_to_", "_to_")
            dst = images_dir / name
            if not dst.exists():
                shutil.copy2(str(f), str(dst))
            count += 1

    dt = time.time() - t0
    avg_time = dt / max(count, 1)
    report = f"CUT: {count} images, {dt:.1f}s total, {avg_time:.2f}s/img (copy only)"
    print(f"\n{report}")
    timing_file.write(report + "\n")
    return count, dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="all",
                       choices=["styleid", "s2wat", "samst", "cut", "all"])
    args = parser.parse_args()

    timing_path = RESULTS_DIR / "timing_report.txt"
    timing_file = open(timing_path, "w", encoding="utf-8")
    timing_file.write(f"Inference Timing Report\n{'='*60}\n")
    timing_file.write(f"Content styles: {CONTENT_STYLES}\n")
    timing_file.write(f"Target styles: {TARGET_STYLES}\n")
    timing_file.write(f"Images per combo: {N_IMAGES}\n")
    timing_file.write(f"Expected total: {EXPECTED_TOTAL}\n\n")

    baselines = ["styleid", "s2wat", "samst", "cut"] if args.baseline == "all" else [args.baseline]

    for bl in baselines:
        images_dir = clean_baseline_images(bl)
        if bl == "styleid":
            run_styleid(images_dir, timing_file)
        elif bl == "s2wat":
            run_s2wat(images_dir, timing_file)
        elif bl == "samst":
            run_samst(images_dir, timing_file)
        elif bl == "cut":
            run_cut(images_dir, timing_file)

    timing_file.close()
    print(f"\nTiming report saved to {timing_path}")


if __name__ == "__main__":
    main()
