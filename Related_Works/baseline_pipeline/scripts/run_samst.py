"""
SaMST Training + Inference Script
ACCV 2024 - Pluggable Style Representation Learning for Multi-Style Transfer
Repo: https://github.com/SYSU-SAIL/SaMST

Uses custom TransformerNet + VGG16, lightweight (no SD1.5/SDXL needed).
"""
import os
import sys
import shutil
import subprocess
import argparse
import yaml
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PIPELINE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PIPELINE_ROOT.parent.parent
STYLE_DATA = REPO_ROOT / "style_data"
OVERFIT50 = STYLE_DATA / "overfit50"
SAMST_REPO = PIPELINE_ROOT.parent / "SaMST-main"


def prepare_dataset(style_name):
    """Prepare dataset in SaMST format.
    SaMST expects: train_dataset/content/content/*.jpg and train_dataset/style/*.jpg
    """
    dataset_dir = SAMST_REPO / "train_dataset"
    content_dir = dataset_dir / "content" / "content"
    style_dir = dataset_dir / "style"
    content_dir.mkdir(parents=True, exist_ok=True)
    style_dir.mkdir(parents=True, exist_ok=True)

    # Copy content images (use train split for training)
    photo_src = STYLE_DATA / "train" / "photo"
    if photo_src.exists():
        for img in sorted(photo_src.glob("*.jpg"))[:500]:
            dst = content_dir / img.name
            if not dst.exists():
                shutil.copy2(str(img), str(dst))

    # Copy style images (use 1 image per style for single-style transfer)
    style_src = STYLE_DATA / "train" / style_name
    if style_src.exists():
        for img in sorted(style_src.glob("*.jpg"))[:1]:
            dst = style_dir / img.name
            if not dst.exists():
                shutil.copy2(str(img), str(dst))

    return dataset_dir


ALL_STYLES = ["photo", "monet", "vangogh", "ukiyoe", "cezanne", "Hayao"]


def prepare_test_input():
    """Prepare test content images from all 5 overfit50 directories."""
    test_dir = SAMST_REPO / "content"
    test_dir.mkdir(parents=True, exist_ok=True)

    for content_style in ALL_STYLES:
        src = OVERFIT50 / content_style
        if not src.exists():
            continue
        for img in sorted(src.glob("*.jpg")):
            # Prefix with content style to avoid name collisions
            dst = test_dir / f"{content_style}_{img.name}"
            if not dst.exists():
                shutil.copy2(str(img), str(dst))

    return test_dir


def train(style_name, epochs=100, smoke=False):
    """Train SaMST for one style."""
    if not SAMST_REPO.exists():
        print(f"[ERROR] SaMST repo not found at {SAMST_REPO}")
        return 1

    # Use train2 pipeline (fast convergence for few styles)
    train_dir = SAMST_REPO / "train_model" / "train2"
    train_script = train_dir / "train.py"
    if not train_script.exists():
        print(f"[ERROR] Training script not found: {train_script}")
        return 1

    prepare_dataset(style_name)

    n_epochs = 1 if smoke else epochs
    batch_size = 1 if smoke else 2  # conservative for 8GB VRAM

    # Create dynamic train.yml
    config = {
        "epochs": n_epochs,
        "batch_size": batch_size,
        "dataset": "../../train_dataset/content/",
        "style_image": "../../train_dataset/style/",
        "save_model_dir": str(PIPELINE_ROOT / "checkpoints" / "samst" / style_name),
        "image_size": 128 if smoke else 256,
        "style_size": 256 if smoke else 512,
        "cuda": 1,
        "seed": 7,
        "content_weight": 1e5,
        "style_weight": 1e10,
        "ae_weight": 1e3,
        "lr": 0.001,
        "weight_decay": 0.5,
        "step_size": 25,
        "save_interval": max(1, n_epochs),
        "log_interval": 10,
        "checkpoint_interval": 100,
        "checkpoint_model_dir": None,
        "begin_checkpoint": None,
        "begin_epoch": None,
    }

    ckpt_dir = Path(config["save_model_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Write train.yml to train_dir
    config_path = train_dir / "train.yml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n[SaMST TRAIN] style={style_name}, epochs={n_epochs}, batch_size={batch_size}")
    result = subprocess.run(
        [sys.executable, str(train_script)],
        cwd=str(train_dir),
    )
    return result.returncode


def infer(target_style, max_images=0):
    """Run SaMST inference: all 5 content dirs -> target_style = 5*30=150 images."""
    if not SAMST_REPO.exists():
        print(f"[ERROR] SaMST repo not found at {SAMST_REPO}")
        return 1

    test_dir_path = SAMST_REPO / "test_model" / "test"
    test_script = test_dir_path / "test.py"
    if not test_script.exists():
        print(f"[ERROR] Test script not found: {test_script}")
        return 1

    prepare_test_input()

    # Find trained model
    ckpt_dir = PIPELINE_ROOT / "checkpoints" / "samst" / target_style
    model_files = sorted(ckpt_dir.glob("epoch_*.model"))
    if not model_files:
        print(f"[ERROR] No trained model found in {ckpt_dir}")
        return 1
    model_path = model_files[-1]

    # SaMST test outputs to a temp dir, then we rename style1 -> final format
    raw_output_dir = SAMST_REPO / "outputs"
    final_output_dir = PIPELINE_ROOT / "results" / "samst" / target_style
    final_output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "content_image_dir": str(SAMST_REPO / "content"),
        "content_scale": None,
        "output_image_dir": str(raw_output_dir) + "/",
        "model": str(model_path),
        "style_num": 1,
        "cuda": 1,
    }

    config_path = test_dir_path / "test.yml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n[SaMST INFER] target={target_style}, model={model_path.name}")
    result = subprocess.run(
        [sys.executable, str(test_script)],
        cwd=str(test_dir_path),
    )
    if result.returncode != 0:
        return result.returncode

    # Rename: style1_{content_style}_{img}.jpg -> {content_style}_{img}_to_{target}.jpg
    # style0 is identity, style1 is the stylized output
    count = 0
    for f in sorted(raw_output_dir.glob("style1_*.jpg")):
        # style1_{content_style}_{original_name}.jpg
        original = f.name[len("style1_"):]
        # Extract content_style prefix (e.g., "monet_" from "monet_xxx.jpg")
        parts = original.split("_", 1)
        if len(parts) == 2:
            content_style, img_name = parts
            new_name = f"{content_style}_{img_name.replace('.jpg', '')}_to_{target_style}.jpg"
        else:
            new_name = original.replace(".jpg", f"_to_{target_style}.jpg")
        dst = final_output_dir / new_name
        if not dst.exists():
            shutil.copy2(str(f), str(dst))
        count += 1

    # Clean up raw output
    shutil.rmtree(str(raw_output_dir), ignore_errors=True)
    print(f"[SaMST INFER] {count} images -> {final_output_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, required=True)
    parser.add_argument("--mode", type=str, default="all",
                       choices=["train", "infer", "all", "smoke"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max_images", type=int, default=0, help="Max images (0=all)")
    args = parser.parse_args()

    if args.mode == "smoke":
        rc = train(args.style, smoke=True)
        if rc != 0:
            return rc
        rc = infer(args.style, args.max_images)
    elif args.mode == "train":
        rc = train(args.style, args.epochs)
    elif args.mode == "infer":
        rc = infer(args.style, args.max_images)
    else:  # all
        rc = train(args.style, args.epochs)
        if rc != 0:
            return rc
        rc = infer(args.style, args.max_images)

    return rc


if __name__ == "__main__":
    sys.exit(main())
