"""
B-LoRA Execution Script
2024 SDXL B-LoRA for Style Fine-tuning
Optimized for 8GB VRAM
"""
import os
import sys
import subprocess
from pathlib import Path

# Configuration
BLORA_ROOT = "../../../../Related_Works/B-LoRA"
OUTPUT_DIR = "../results/blora"
CHECKPOINT_DIR = "../checkpoints/blora"
STYLE_DATA_ROOT = "../../../style_data"
TEST_CONTENT_DIR = "../../../style_data/overfit50/photo"

def run_blora(style_name, mode="all"):
    """Run B-LoRA fine-tuning and inference for a specific style"""
    output_dir = os.path.join(OUTPUT_DIR, style_name)
    ckpt_dir = os.path.join(CHECKPOINT_DIR, style_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    style_train_dir = os.path.join(STYLE_DATA_ROOT, style_name)
    
    if mode in ["train", "all"]:
        print(f"Fine-tuning B-LoRA for {style_name} (expected time: ~15 minutes)...")
        cmd = [
            sys.executable,
            os.path.join(BLORA_ROOT, "train_blora.py"),
            "--pretrained_model_name_or_path", "stabilityai/stable-diffusion-xl-base-1.0",
            "--train_data_dir", style_train_dir,
            "--output_dir", ckpt_dir,
            "--resolution", "256",
            "--train_batch_size", "1",  # Optimized for 8GB VRAM
            "--gradient_accumulation_steps", "4",
            "--learning_rate", "1e-4",
            "--lr_scheduler", "constant",
            "--lr_warmup_steps", "0",
            "--max_train_steps", "500",
            "--checkpointing_steps", "100",
            "--enable_xformers_memory_efficient_attention",
            "--gradient_checkpointing",
            "--use_8bit_adam",
            "--style_lora_rank", "8",
            "--content_lora_rank", "8"
        ]
        subprocess.run(cmd, cwd=BLORA_ROOT, check=True)
    
    if mode in ["test", "all"]:
        print(f"Generating B-LoRA results for {style_name}...")
        cmd = [
            sys.executable,
            os.path.join(BLORA_ROOT, "inference.py"),
            "--base_model", "stabilityai/stable-diffusion-xl-base-1.0",
            "--style_lora_path", os.path.join(ckpt_dir, "pytorch_lora_weights.safetensors"),
            "--content_dir", TEST_CONTENT_DIR,
            "--output_dir", output_dir,
            "--image_size", "256",
            "--num_inference_steps", "20"
        ]
        subprocess.run(cmd, cwd=BLORA_ROOT, check=True)
        print(f"B-LoRA results saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, required=True, help="Style name to run")
    parser.add_argument("--mode", type=str, default="all", choices=["train", "test", "all"])
    args = parser.parse_args()
    
    run_blora(args.style, args.mode)
