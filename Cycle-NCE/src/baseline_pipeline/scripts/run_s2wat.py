"""
S2WAT Execution Script
AAAI 2024 Wavelet Transformer Style Transfer
Optimized for 8GB VRAM
"""
import os
import sys
import subprocess
from pathlib import Path

# Configuration
S2WAT_ROOT = "../../../../Related_Works/repos/S2WAT-main"
OUTPUT_DIR = "../results/s2wat"
STYLE_DATA_ROOT = "../../../style_data"
TEST_CONTENT_DIR = "../../../style_data/overfit50/photo"

def run_s2wat(style_name, mode="all"):
    """Run S2WAT training and inference for a specific style"""
    output_dir = os.path.join(OUTPUT_DIR, style_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if mode in ["train", "all"]:
        print(f"Training S2WAT for {style_name}...")
        cmd = [
            sys.executable,
            os.path.join(S2WAT_ROOT, "train.py"),
            "--content_dir", os.path.join(STYLE_DATA_ROOT, "photo/train"),
            "--style_dir", os.path.join(STYLE_DATA_ROOT, style_name),
            "--batch_size", "2",  # Optimized for 8GB VRAM
            "--image_size", "256",
            "--epochs", "1",  # 1 epoch for smoke test
            "--save_dir", f"../checkpoints/s2wat/{style_name}",
            "--wavelet", "db4",
            "--num_workers", "2"
        ]
        subprocess.run(cmd, cwd=S2WAT_ROOT, check=True)
    
    if mode in ["test", "all"]:
        print(f"Generating S2WAT results for {style_name}...")
        cmd = [
            sys.executable,
            os.path.join(S2WAT_ROOT, "test.py"),
            "--content_dir", TEST_CONTENT_DIR,
            "--style_dir", os.path.join(STYLE_DATA_ROOT, style_name),
            "--checkpoint", f"../checkpoints/s2wat/{style_name}/latest.pt",
            "--output_dir", output_dir,
            "--image_size", "256"
        ]
        subprocess.run(cmd, cwd=S2WAT_ROOT, check=True)
        print(f"S2WAT results saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, required=True, help="Style name to run")
    parser.add_argument("--mode", type=str, default="all", choices=["train", "test", "all"])
    args = parser.parse_args()
    
    run_s2wat(args.style, args.mode)
