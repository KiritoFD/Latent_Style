"""
StyleID Execution Script
Optimized for 8GB VRAM
"""
import os
import sys
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# Add StyleID to path (from Related_Works)
sys.path.append("../../../../Related_Works/StyleID")
from diffusers import StableDiffusionPipeline, DDIMScheduler
from attention_injection import AttentionInjectionHook

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
MODEL_ID = "runwayml/stable-diffusion-v1-5"
INFERENCE_STEPS = 50  # Reduced from default 100 for speed
IMAGE_SIZE = 256
GUIDANCE_SCALE = 7.5
OUTPUT_DIR = "../results/styleid"

# Memory optimizations for 8GB VRAM
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)

def load_style_reference(style_path):
    """Load and preprocess style reference image"""
    img = Image.open(style_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    return img

def run_styleid(content_dir, style_dir, style_name):
    """Run StyleID for a specific style"""
    output_dir = os.path.join(OUTPUT_DIR, style_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading Stable Diffusion for StyleID...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        safety_checker=None,
        use_safetensors=True
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    
    # Enable memory optimizations
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing(slice_size="auto")
    pipe.enable_xformers_memory_efficient_attention()
    
    # Load style reference
    style_files = sorted([f for f in os.listdir(style_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    style_path = os.path.join(style_dir, style_files[0])  # Use first style reference
    style_img = load_style_reference(style_path)
    
    # Initialize attention hook
    hook = AttentionInjectionHook(
        pipe.unet,
        reference_image=style_img,
        injection_steps=20,
        injection_weight=0.8
    )
    
    # Process content images
    content_files = sorted([f for f in os.listdir(content_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Processing {len(content_files)} content images for style {style_name}...")
    
    for fname in tqdm(content_files):
        content_path = os.path.join(content_dir, fname)
        content_img = Image.open(content_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        
        # DDIM inversion first
        with torch.no_grad():
            # Invert content image
            inverted_latents = pipe.invert(
                content_img,
                num_inference_steps=INFERENCE_STEPS
            ).latents
            
            # Generate with style injection
            result = pipe(
                prompt="",
                negative_prompt="",
                latents=inverted_latents,
                num_inference_steps=INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                output_type="pil"
            ).images[0]
        
        # Save result
        out_path = os.path.join(output_dir, f"{Path(fname).stem}_stylized.jpg")
        result.save(out_path)
        
        # Clean up
        torch.cuda.empty_cache()
    
    hook.remove()
    del pipe, hook
    torch.cuda.empty_cache()
    print(f"StyleID completed for {style_name}, results saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, required=True, help="Style name to run")
    parser.add_argument("--content_dir", type=str, default="../datasets/test_content", help="Content images directory")
    parser.add_argument("--style_dir_prefix", type=str, default="../datasets/test_style", help="Style images directory prefix")
    args = parser.parse_args()
    
    style_dir = os.path.join(args.style_dir_prefix, args.style)
    run_styleid(args.content_dir, style_dir, args.style)
