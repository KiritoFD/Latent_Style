import os
import torch
from PIL import Image
from diffusers import AutoencoderKL
from pathlib import Path
import argparse

# 载入 VAE 编码器
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to("cuda")

# 支持的图片格式
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def save_latent_image(image_path, latent_path, device):
    """保存图片为 latent.pt"""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((256, 256))  # 调整为 512x512（SD VAE 需要的尺寸）
    image_tensor = torch.tensor(image).unsqueeze(0).float().div(255).to(device)

    with torch.no_grad():
        # 获取 latent 表示
        latent = vae.encode(image_tensor).latent_dist.sample()  # 使用 latent_dist.sample() 获取latent

    # 保存为 .pt 文件
    torch.save(latent.cpu(), latent_path)

def process_directory(root_dir, latents_root, device):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in EXTS:
                continue
            
            # 获取子目录路径
            relative_path = os.path.relpath(dirpath, root_dir)
            target_dir = os.path.join(latents_root, relative_path)
            
            # 确保目标目录存在
            Path(target_dir).mkdir(parents=True, exist_ok=True)
            
            # 定义源图片路径和目标 latent 路径
            image_path = os.path.join(dirpath, filename)
            latent_path = os.path.join(target_dir, f"{os.path.splitext(filename)[0]}.pt")
            
            # 只处理未存在的 latent 文件
            if not os.path.exists(latent_path):
                save_latent_image(image_path, latent_path, device)
                print(f"Processed and saved: {latent_path}")

def main():
    # 使用 argparse 来指定根目录和目标 latent 存储路径
    parser = argparse.ArgumentParser(description="Convert images to latents using VAE.")
    parser.add_argument("--root_dir", default="/mnt/g/GitHub/Latent_Style/style_data/train/", type=str, required=True, help="Root directory of images to convert.")
    parser.add_argument("--latents_root", default="/mnt/g/GitHub/Latent_Style/data/latents/train/", type=str, required=True, help="Directory to save the latents.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run the VAE (default: 'cuda').")

    args = parser.parse_args()

    # 根据用户指定的设备选择 VAE
    device = torch.device(args.device)

    # 调用处理目录的函数
    process_directory(args.root_dir, args.latents_root, device)

if __name__ == "__main__":
    main()
