import os
import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
from pathlib import Path

# 载入 VAE 编码器
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to("cuda")
vae.eval()

# 目录路径
root_dir = "/mnt/g/GitHub/Latent_Style/style_data/overfit50"
latent_dir = "../latents_overfit50"

# 支持的图片格式
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 确保保存目录存在
Path(latent_dir).mkdir(parents=True, exist_ok=True)

def save_latent_image(image_path, latent_path):
    """保存图片为 latent.pt"""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((256, 256))  # 调整为 512x512（SD VAE 需要的尺寸）
    image_tensor = torch.from_numpy(np.array(image)).unsqueeze(0).float().div(255.0).to("cuda")
    image_tensor = image_tensor.permute(0, 3, 1, 2) * 2.0 - 1.0

    with torch.inference_mode():
        # 获取 latent 表示 (match SD scaling)
        latent = vae.encode(image_tensor).latent_dist.sample() * 0.18215
    
    # 保存为 .pt 文件
    torch.save(latent, latent_path)

def process_directory():
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in EXTS:
                continue
            
            # 获取子目录路径
            relative_path = os.path.relpath(dirpath, root_dir)
            target_dir = os.path.join(latent_dir, relative_path)
            
            # 确保目标目录存在
            Path(target_dir).mkdir(parents=True, exist_ok=True)
            
            # 定义源图片路径和目标 latent 路径
            image_path = os.path.join(dirpath, filename)
            latent_path = os.path.join(target_dir, f"{os.path.splitext(filename)[0]}.pt")
            
            # 只处理未存在的 latent 文件
            if not os.path.exists(latent_path):
                save_latent_image(image_path, latent_path)
                print(f"Processed and saved: {latent_path}")

if __name__ == "__main__":
    process_directory()
