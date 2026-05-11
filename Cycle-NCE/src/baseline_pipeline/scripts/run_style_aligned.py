"""
StyleAligned Execution Script
CVPR 2024 Google - Attention Sharing + ControlNet for Style Transfer
Optimized for 8GB VRAM
"""
import os
import sys
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import CannyDetector

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
OUTPUT_DIR = "../results/style_aligned"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
CONTROLNET_ID = "lllyasviel/sd-controlnet-canny"
INFERENCE_STEPS = 50
IMAGE_SIZE = 256

# Memory optimizations
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)

def run_style_aligned(content_dir, style_dir, style_name):
    """Run StyleAligned with ControlNet Canny for style transfer"""
    output_dir = os.path.join(OUTPUT_DIR, style_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    print("Loading ControlNet and Stable Diffusion...")
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=DTYPE)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_ID,
        controlnet=controlnet,
        torch_dtype=DTYPE,
        safety_checker=None
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    
    # Memory optimizations
    pipe.enable_attention_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()  # Critical for 8GB VRAM
    
    # Load Canny detector
    canny_detector = CannyDetector()
    
    # Load style reference
    style_files = sorted([f for f in os.listdir(style_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    style_path = os.path.join(style_dir, style_files[0])
    style_img = Image.open(style_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # Process content images
    content_files = sorted([f for f in os.listdir(content_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Processing {len(content_files)} content images for style {style_name}...")
    
    for fname in tqdm(content_files):
        content_path = os.path.join(content_dir, fname)
        content_img = Image.open(content_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        
        # Generate canny edge map
        canny_img = canny_detector(content_img, low_threshold=100, high_threshold=200)
        
        # Generate with style alignment
        with torch.no_grad(), torch.autocast(DEVICE, dtype=DTYPE):
            result = pipe(
                prompt=f"a painting in the style of {style_name}",
                negative_prompt="ugly, blurry, low quality",
                image=canny_img,
                num_inference_steps=INFERENCE_STEPS,
                guidance_scale=7.5,
                controlnet_conditioning_scale=0.7,
                output_type="pil"
            ).images[0]
        
        # Save result
        out_path = os.path.join(output_dir, f"{Path(fname).stem}_stylized.jpg")
        result.save(out_path)
        
        torch.cuda.empty_cache()
    
    del pipe, controlnet, canny_detector
    torch.cuda.empty_cache()
    print(f"StyleAligned completed for {style_name}, results saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, required=True, help="Style name to run")
    parser.add_argument("--content_dir", type=str, default="../../../style_data/overfit50/photo", help="Content images directory")
    parser.add_argument("--style_dir_prefix", type=str, default="../../../style_data/overfit50", help="Style images directory prefix")
    args = parser.parse_args()
    
    style_dir = os.path.join(args.style_dir_prefix, args.style)
    run_style_aligned(args.content_dir, style_dir, args.style)
