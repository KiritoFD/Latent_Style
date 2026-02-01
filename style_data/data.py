import os
import shutil
import random
import torch
import glob
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from diffusers import AutoencoderKL

# ================= 配置区域 =================
# 根目录配置
BASE_DIR = r"C:\Users\xy\style_data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
LATENTS_DIR = os.path.join(BASE_DIR, "latents")

# VAE配置 (请确认这是你想用的VAE，通常用SD官方的)
VAE_ID = "stabilityai/sd-vae-ft-mse" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4  # 显存够大可以开大，4070 8G 建议 4-8
IMG_SIZE = 256  # 按照之前讨论，锁定 256
SCALING_FACTOR = 0.18215 # SD 标准缩放因子

# ================= 功能函数 =================

def split_dataset(num_val=100):
    """
    从 train 中每个子文件夹随机移动 num_val 张图片到 test
    """
    print(f"[Info] Starting dataset split (Target: {num_val} images/style)...")
    
    # 获取所有风格子目录
    style_dirs = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    
    for style in style_dirs:
        src_style_dir = os.path.join(TRAIN_DIR, style)
        dst_style_dir = os.path.join(TEST_DIR, style)
        
        # 创建目标目录
        os.makedirs(dst_style_dir, exist_ok=True)
        
        # 获取所有图片文件
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        images = []
        for ext in extensions:
            images.extend(glob.glob(os.path.join(src_style_dir, ext)))
        
        # 检查数量
        if len(images) < num_val:
            print(f"[Warning] Style '{style}' has only {len(images)} images. Skipping split.")
            continue
            
        # 随机选择并移动
        selected_imgs = random.sample(images, num_val)
        for img_path in selected_imgs:
            shutil.move(img_path, os.path.join(dst_style_dir, os.path.basename(img_path)))
            
    print("[Info] Dataset split completed.")

def get_transforms():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE), # 保证正方形
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

@torch.no_grad()
def encode_and_save(src_root, dst_root):
    """
    遍历目录，编码图片并保存为 .pt 文件
    """
    print(f"[Info] Starting VAE encoding from {src_root}...")
    
    # 加载 VAE (FP16 优化)
    print(f"[Info] Loading VAE: {VAE_ID}")
    vae = AutoencoderKL.from_pretrained(VAE_ID).to(DEVICE, dtype=torch.float16)
    vae.eval()
    
    transform = get_transforms()
    
    # 遍历所有文件
    # 使用 os.walk 保持目录结构
    for root, dirs, files in os.walk(src_root):
        # 计算相对路径，用于在 latents 下重建结构
        rel_path = os.path.relpath(root, src_root)
        target_dir = os.path.join(dst_root, rel_path)
        os.makedirs(target_dir, exist_ok=True)
        
        # 筛选图片
        valid_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        if not valid_files:
            continue
            
        # Batch 处理循环
        for i in tqdm(range(0, len(valid_files), BATCH_SIZE), desc=f"Encoding {rel_path}"):
            batch_files = valid_files[i : i + BATCH_SIZE]
            batch_tensors = []
            save_paths = []
            
            # 数据加载 (CPU)
            for fname in batch_files:
                img_path = os.path.join(root, fname)
                try:
                    img = Image.open(img_path).convert("RGB")
                    tensor = transform(img)
                    batch_tensors.append(tensor)
                    
                    # 构造保存路径: change .jpg -> .pt
                    save_name = os.path.splitext(fname)[0] + ".pt"
                    save_paths.append(os.path.join(target_dir, save_name))
                except Exception as e:
                    print(f"[Error] Failed to load {fname}: {e}")
            
            if not batch_tensors:
                continue
                
            # 堆叠 Batch 并移至 GPU
            input_batch = torch.stack(batch_tensors).to(DEVICE, dtype=torch.float16)
            
            # VAE 推理 (GPU)
            # 使用 deterministic=True 或者 sample() 均可，这里用 sample() 保持分布特性
            posterior = vae.encode(input_batch).latent_dist
            latents = posterior.sample() * SCALING_FACTOR
            
            # 保存 (CPU)
            # 转换回 float32 存储以保证精度，并分离梯度
            latents = latents.detach().float().cpu()
            
            for idx, save_path in enumerate(save_paths):
                torch.save(latents[idx], save_path)

# ================= 主程序 =================

if __name__ == "__main__":
    # 1. 执行划分 (只运行一次，如果已经划分过，脚本逻辑会因文件移走而自动适应，但建议谨慎)
    # 检查 test 目录是否为空，避免重复运行导致训练集被掏空
    if not os.path.exists(TEST_DIR) or not os.listdir(TEST_DIR):
        split_dataset(num_val=100)
    else:
        print("[Info] Test directory not empty. Skipping split to prevent data loss.")

    # 2. 编码 Train 集
    encode_and_save(TRAIN_DIR, os.path.join(LATENTS_DIR, "train"))
    
    # 3. 编码 Test 集 (用于验证评估)
    encode_and_save(TEST_DIR, os.path.join(LATENTS_DIR, "test"))
    
    print("\n[Success] All tasks finished.")
    print(f"Latents saved to: {LATENTS_DIR}")