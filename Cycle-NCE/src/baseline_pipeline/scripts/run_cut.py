"""
CUT Execution Script
Contrastive Unpaired Translation (ECCV 2020)
"""
import os
import subprocess
import sys
from pathlib import Path

# Configuration
CUT_ROOT = "../../../../Related_Works/contrastive-unpaired-translation"
OUTPUT_DIR = "../results/cut"
CHECKPOINT_DIR = "../checkpoints/cut"

def prepare_dataset(style_name, content_dir, style_dir):
    """Prepare dataset in CUT required format"""
    dataset_dir = os.path.join(CUT_ROOT, "datasets", f"{style_name}2photo")
    os.makedirs(os.path.join(dataset_dir, "trainA"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "trainB"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "testA"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "testB"), exist_ok=True)
    
    # Symlink or copy test content to testA
    content_files = sorted([f for f in os.listdir(content_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    for fname in content_files[:30]:  # Use 30 test images
        src = os.path.abspath(os.path.join(content_dir, fname))
        dst = os.path.join(dataset_dir, "testA", fname)
        if not os.path.exists(dst):
            if sys.platform == "win32":
                import shutil
                shutil.copy(src, dst)
            else:
                os.symlink(src, dst)
    
    # Symlink or copy style images to trainA (use full style dataset for training)
    full_style_dir = f"../../../style_data/{style_name}"
    style_files = sorted([f for f in os.listdir(full_style_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    for fname in style_files:
        src = os.path.abspath(os.path.join(full_style_dir, fname))
        dst = os.path.join(dataset_dir, "trainA", fname)
        if not os.path.exists(dst):
            if sys.platform == "win32":
                import shutil
                shutil.copy(src, dst)
            else:
                os.symlink(src, dst)
    
    # Use existing COCO/photo dataset from style_data
    content_train_dir = "../../../style_data/photo/train"
    content_files = sorted([f for f in os.listdir(content_train_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    for fname in content_files[:1000]:  # Use 1000 content images for fast training
        src = os.path.abspath(os.path.join(content_train_dir, fname))
        dst = os.path.join(dataset_dir, "trainB", fname)
        if not os.path.exists(dst):
            if sys.platform == "win32":
                import shutil
                shutil.copy(src, dst)
            else:
                os.symlink(src, dst)
    
    print(f"Dataset prepared at {dataset_dir}")
    print(f"Training with {len(style_files)} style images, {min(len(content_files), 1000)} content images")
    return dataset_dir

def train_cut(style_name, dataset_dir):
    """Train CUT model"""
    cmd = [
        sys.executable,
        os.path.join(CUT_ROOT, "train.py"),
        "--dataroot", dataset_dir,
        "--name", f"{style_name}_cut",
        "--CUT_mode", "CUT",
        "--batch_size", "1",  # Optimized for 8GB VRAM
        "--load_size", "286",
        "--crop_size", "256",
        "--save_epoch_freq", "5",
        "--total_epochs", "50",  # Reduced for faster training
        "--gpu_ids", "0"
    ]
    
    print(f"Starting CUT training for {style_name}...")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=CUT_ROOT, check=True)

def test_cut(style_name, dataset_dir, output_dir):
    """Test CUT model and generate results"""
    output_dir = os.path.join(output_dir, style_name)
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        sys.executable,
        os.path.join(CUT_ROOT, "test.py"),
        "--dataroot", dataset_dir,
        "--name", f"{style_name}_cut",
        "--CUT_mode", "CUT",
        "--phase", "test",
        "--num_test", "30",
        "--results_dir", output_dir,
        "--gpu_ids", "0"
    ]
    
    print(f"Starting CUT testing for {style_name}...")
    subprocess.run(cmd, cwd=CUT_ROOT, check=True)
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, required=True, help="Style name to run")
    parser.add_argument("--mode", type=str, default="all", choices=["prepare", "train", "test", "all"], help="Execution mode")
    parser.add_argument("--content_dir", type=str, default="../datasets/test_content", help="Content images directory")
    parser.add_argument("--style_dir_prefix", type=str, default="../datasets/test_style", help="Style images directory prefix")
    args = parser.parse_args()
    
    style_dir = os.path.join(args.style_dir_prefix, args.style)
    
    if args.mode in ["prepare", "all"]:
        dataset_dir = prepare_dataset(args.style, args.content_dir, style_dir)
    
    if args.mode in ["train", "all"]:
        train_cut(args.style, dataset_dir)
    
    if args.mode in ["test", "all"]:
        test_cut(args.style, dataset_dir, OUTPUT_DIR)
