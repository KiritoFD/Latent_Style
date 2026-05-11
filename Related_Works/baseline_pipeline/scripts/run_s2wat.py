"""
S2WAT Training + Inference Script
AAAI 2024 Wavelet Transformer Style Transfer

Correctly interfaces with S2WAT-main/train.py and test.py.

S2WAT train.py CLI:
    --content_dir, --style_dir, --vgg_dir, --base_lr, --batch_size,
    --img_size, --train_size, --precision, --grad_checkpoint,
    --epoch, --content_weight, --style_weight, --id1_weight, --id2_weight,
    --checkpoint_save_interval, --loss_count_interval, --resume_train,
    --checkpoint_save_path, --checkpoint_import_path

S2WAT test.py CLI:
    --input_dir (with Content/ and Style/ subdirs),
    --output_dir, --checkpoint_import_path
"""
import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PIPELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent.parent  # g:\GitHub\Latent_Style
S2WAT_ROOT = PIPELINE_ROOT.parent / "S2WAT-main"
STYLE_DATA = REPO_ROOT / "style_data"
TRAIN_DATA = STYLE_DATA / "train"
OVERFIT50 = STYLE_DATA / "overfit50"
VGG_PATH = S2WAT_ROOT / "pre_trained_models" / "vgg_normalised.pth"


ALL_STYLES = ["photo", "monet", "vangogh", "ukiyoe", "cezanne", "Hayao"]


def prepare_test_input(target_style):
    """Prepare S2WAT test input: all 5 overfit50 dirs as Content, 1 style ref."""
    test_input = PIPELINE_ROOT / "tmp" / "s2wat_test_input" / target_style
    content_dir = test_input / "Content"
    style_dir = test_input / "Style"
    content_dir.mkdir(parents=True, exist_ok=True)
    style_dir.mkdir(parents=True, exist_ok=True)

    # Content images from all 5 overfit50 directories (5*30=150)
    for content_style in ALL_STYLES:
        src = OVERFIT50 / content_style
        if not src.exists():
            continue
        for img in sorted(src.glob("*.jpg")):
            dst = content_dir / f"{content_style}_{img.name}"
            if not dst.exists():
                shutil.copy2(str(img), str(dst))

    # Style reference (first image from overfit50/{target_style})
    style_src = OVERFIT50 / target_style
    if style_src.exists():
        for img in sorted(style_src.glob("*.jpg"))[:1]:
            dst = style_dir / img.name
            if not dst.exists():
                shutil.copy2(str(img), str(dst))

    return test_input


def train_s2wat(style_name, epochs=1, batch_size=1, img_size=256):
    """Train S2WAT model"""
    ckpt_dir = PIPELINE_ROOT / "checkpoints" / "s2wat" / style_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    content_dir = TRAIN_DATA / "photo"
    style_dir = TRAIN_DATA / style_name

    if not content_dir.exists():
        print(f"[ERROR] Content dir not found: {content_dir}")
        return 1
    if not style_dir.exists():
        print(f"[ERROR] Style dir not found: {style_dir}")
        return 1
    if not VGG_PATH.exists():
        print(f"[ERROR] VGG weights not found: {VGG_PATH}")
        return 1

    # Use tiny resolution for smoke test to avoid OOM on 8GB VRAM
    train_size = 128 if epochs <= 1 else 0  # 0 = native resolution

    cmd = [
        sys.executable, str(S2WAT_ROOT / "train.py"),
        "--content_dir", str(content_dir),
        "--style_dir", str(style_dir),
        "--vgg_dir", str(VGG_PATH),
        "--epoch", str(epochs),
        "--batch_size", str(batch_size),
        "--img_size", str(img_size),
        "--train_size", str(train_size),
        "--grad_checkpoint",
        "--checkpoint_save_path", str(ckpt_dir),
        "--checkpoint_save_interval", str(max(1, epochs // 2)),
        "--loss_count_interval", str(max(1, epochs // 4)),
        "--precision", "bf16",
    ]

    print(f"\n[S2WAT TRAIN] style={style_name}, epochs={epochs}, bs={batch_size}")
    print(f"  content_dir={content_dir}")
    print(f"  style_dir={style_dir}")
    print(f"  ckpt_dir={ckpt_dir}")
    result = subprocess.run(cmd, cwd=str(S2WAT_ROOT))
    return result.returncode


def infer_s2wat(target_style, checkpoint_path=None):
    """Run S2WAT inference: all 5 content dirs -> target_style = 5*30=150 images."""
    output_dir = PIPELINE_ROOT / "results" / "s2wat" / target_style
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find checkpoint
    if checkpoint_path is None:
        ckpt_dir = PIPELINE_ROOT / "checkpoints" / "s2wat" / target_style
        candidates = list(ckpt_dir.glob("checkpoint_*_epoch.pkl"))
        if candidates:
            checkpoint_path = sorted(candidates)[-1]
        else:
            pretrained = S2WAT_ROOT / "pre_trained_models" / "checkpoint_bs1_256"
            candidates = list(pretrained.glob("*.pkl"))
            if candidates:
                checkpoint_path = candidates[0]
            else:
                print(f"[ERROR] No checkpoint found for S2WAT/{target_style}")
                return 1

    test_input = prepare_test_input(target_style)

    # S2WAT test outputs to a temp dir, then we rename
    raw_output_dir = PIPELINE_ROOT / "tmp" / "s2wat_raw_output" / target_style
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(S2WAT_ROOT / "test.py"),
        "--input_dir", str(test_input),
        "--output_dir", str(raw_output_dir),
        "--checkpoint_import_path", str(checkpoint_path),
    ]

    print(f"\n[S2WAT INFER] target={target_style}")
    print(f"  checkpoint={checkpoint_path}")
    result = subprocess.run(cmd, cwd=str(S2WAT_ROOT))
    if result.returncode != 0:
        return result.returncode

    # Rename outputs to {content_style}_{img}_to_{target_style}.jpg
    count = 0
    for f in sorted(raw_output_dir.glob("*.jpg")):
        name = f.stem  # e.g., "photo_xxx" or "monet_xxx"
        # If already has _to_ suffix from S2WAT, just ensure target matches
        if "_to_" in name:
            new_name = f.name
        else:
            new_name = f"{name}_to_{target_style}.jpg"
        dst = output_dir / new_name
        if not dst.exists():
            shutil.copy2(str(f), str(dst))
        count += 1

    shutil.rmtree(str(raw_output_dir), ignore_errors=True)
    print(f"[S2WAT INFER] {count} images -> {output_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="S2WAT Baseline")
    parser.add_argument("--style", type=str, required=True, help="Style name")
    parser.add_argument("--mode", type=str, default="all",
                       choices=["train", "infer", "all", "smoke"])
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "smoke":
        args.epochs = 1
        args.mode = "all"

    rc = 0
    if args.mode in ["train", "all"]:
        rc = train_s2wat(args.style, args.epochs, args.batch_size, args.img_size)
        if rc != 0:
            print(f"[FAIL] S2WAT training failed for {args.style}")
            return rc

    if args.mode in ["infer", "all"]:
        rc = infer_s2wat(args.style, args.checkpoint)
        if rc != 0:
            print(f"[FAIL] S2WAT inference failed for {args.style}")

    return rc


if __name__ == "__main__":
    sys.exit(main())
