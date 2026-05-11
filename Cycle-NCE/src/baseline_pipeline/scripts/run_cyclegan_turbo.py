"""
CycleGAN-Turbo Execution Script
Uses official pre-trained weights, optimized for 8GB VRAM
"""
import os
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
OUTPUT_DIR = "../results/cyclegan_turbo"
IMAGE_SIZE = 256

# Model paths
MODEL_PATHS = {
    "monet": "gaetanparmar/cyclegan-turbo-photo2monet",
    "vangogh": "gaetanparmar/cyclegan-turbo-photo2vangogh",
    "ukiyoe": "gaetanparmar/cyclegan-turbo-photo2ukiyoe",
    "cezanne": "gaetanparmar/cyclegan-turbo-photo2cezanne"
}

def run_cyclegan_turbo(content_dir, style_name):
    """Run CycleGAN-Turbo for a specific style"""
    if style_name not in MODEL_PATHS:
        print(f"Error: No pre-trained model available for style {style_name}")
        print(f"Available styles: {list(MODEL_PATHS.keys())}")
        return
    
    output_dir = os.path.join(OUTPUT_DIR, style_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading CycleGAN-Turbo model for {style_name}...")
    pipe = AutoPipelineForImage2Image.from_pretrained(
        MODEL_PATHS[style_name],
        torch_dtype=DTYPE,
        safety_checker=None
    )
    pipe = pipe.to(DEVICE)
    
    # Memory optimizations
    pipe.enable_attention_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    
    # Process content images
    content_files = sorted([f for f in os.listdir(content_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Processing {len(content_files)} content images for style {style_name}...")
    
    for fname in tqdm(content_files):
        content_path = os.path.join(content_dir, fname)
        content_img = load_image(content_path).resize((IMAGE_SIZE, IMAGE_SIZE))
        
        # 1-step inference
        with torch.no_grad(), torch.autocast(DEVICE, dtype=DTYPE):
            result = pipe(
                content_img,
                num_inference_steps=1,
                guidance_scale=0.0,
                output_type="pil"
            ).images[0]
        
        # Save result
        out_path = os.path.join(output_dir, f"{Path(fname).stem}_stylized.jpg")
        result.save(out_path)
        
        torch.cuda.empty_cache()
    
    del pipe
    torch.cuda.empty_cache()
    print(f"CycleGAN-Turbo completed for {style_name}, results saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, required=True, help="Style name to run (monet/vangogh/ukiyoe/cezanne)")
    parser.add_argument("--content_dir", type=str, default="../datasets/test_content", help="Content images directory")
    args = parser.parse_args()
    
    run_cyclegan_turbo(args.content_dir, args.style)
