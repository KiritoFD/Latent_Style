import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 确保能导入 src 目录下的模块
sys.path.append(str(Path(__file__).parent / "src"))
from dataset import LatentDataset

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("ScaleAnalyzer")

def get_projected_distribution(latents, patch_size, num_projections=128, device='cuda'):
    """
    计算一组 Latent 在特定 Patch Size 下的投影分布 (Sorted Projections)。
    使用全卷积投影 + 排序，模拟 SWD Loss 的核心逻辑。
    """
    B, C, H, W = latents.shape
    
    # 1. 构造固定的随机投影矩阵 (Random Orthogonal Matrix)
    # 固定随机种子，确保不同风格使用同一组基向量进行投影
    rng_state = torch.get_rng_state()
    torch.manual_seed(42 + patch_size) 
    projections = torch.randn(num_projections, C, patch_size, patch_size, device=device)
    # 归一化投影向量
    projections = F.normalize(projections.view(num_projections, -1), dim=1).view(num_projections, C, patch_size, patch_size)
    torch.set_rng_state(rng_state) 
    
    # 2. 投影 (使用 Conv2d)
    # 为了防止显存溢出，分块处理
    chunk_size = 512 
    all_projections = []
    
    padding = patch_size // 2
    
    # 预计算归一化项 (Mean Subtraction) - 消除亮度影响，只看纹理
    if patch_size > 1:
        mean_kernel = torch.ones(1, C, patch_size, patch_size, device=device) / (C * patch_size**2)
        proj_sum = projections.sum(dim=(1, 2, 3)).view(1, -1, 1, 1)
    
    with torch.no_grad():
        for i in range(0, B, chunk_size):
            batch = latents[i : i + chunk_size].to(device)
            
            if patch_size > 1:
                batch_mean = F.conv2d(batch, mean_kernel, padding=padding)
                # Math: (Patch * Proj) - (Mean * Sum_Proj)
                proj = F.conv2d(batch, projections, padding=padding) - batch_mean * proj_sum
            else:
                proj = F.conv2d(batch, projections, padding=padding)
            
            # [B_chunk, N_proj, H, W] -> [Pixels, N_proj]
            proj_flat = proj.permute(0, 2, 3, 1).reshape(-1, num_projections)
            all_projections.append(proj_flat)
            
    # 3. 合并并排序 (全量分布)
    total_distribution = torch.cat(all_projections, dim=0)
    
    # 沿样本维度排序 [N_total, N_proj] -> Sorted
    sorted_dist, _ = torch.sort(total_distribution, dim=0)
    
    return sorted_dist

def compute_wasserstein_distance(dist_a, dist_b):
    """
    计算两个已排序分布之间的 SWD (L2 距离)
    """
    n_a = dist_a.shape[0]
    n_b = dist_b.shape[0]
    
    # 简单的重采样对齐
    if n_a != n_b:
        min_n = min(n_a, n_b)
        idx_a = torch.linspace(0, n_a - 1, min_n).long().to(dist_a.device)
        idx_b = torch.linspace(0, n_b - 1, min_n).long().to(dist_b.device)
        d_a = dist_a[idx_a]
        d_b = dist_b[idx_b]
    else:
        d_a = dist_a
        d_b = dist_b
        
    return F.mse_loss(d_a, d_b).item() * 1000  # 放大数值方便阅读

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    args = parser.parse_args()

    # 1. 加载配置和数据
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device} for analysis")
    
    # 初始化数据集
    dataset = LatentDataset(
        data_root=config['data']['data_root'],
        num_styles=config['model']['num_styles'],
        style_subdirs=config['data'].get('style_subdirs'),
        config=config
    )
    
    style_names = dataset.style_subdirs
    style_indices = dataset.style_indices
    
    # 2. 加载所有数据到内存 (Latent 很小，可以直接存)
    logger.info("Loading all latents into memory...")
    style_data = {}
    for sid, indices in style_indices.items():
        name = style_names[sid]
        # 获取该风格的所有 latents
        style_data[name] = dataset.latents_tensor[indices].to('cpu') # 先存 CPU
        logger.info(f"  Loaded {name}: {len(indices)} samples")

    # 3. 定义要扫描的尺度
    # 1=Pixel(Color), 3=Fine Texture, 5=Coarse Texture, 7=Structure, 9=Layout
    scales = list(range(1, 16, 1)) 
    
    out_dir = Path("analysis_results")
    out_dir.mkdir(exist_ok=True)
    
    logger.info("\n🚀 Starting Full Spectrum Analysis...")
    
    for scale in scales:
        logger.info(f"\n[Scale {scale}x{scale}] Computing Distributions...")
        
        # 4.1 计算每个风格的分布
        distributions = {}
        for name, latents in style_data.items():
            # 放到 GPU 计算
            dist = get_projected_distribution(latents, scale, device=device)
            distributions[name] = dist
            
        # 4.2 计算距离矩阵
        n = len(style_names)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j: continue
                d = compute_wasserstein_distance(
                    distributions[style_names[i]], 
                    distributions[style_names[j]]
                )
                dist_matrix[i, j] = d
        
        # 4.3 绘制热力图
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(dist_matrix, dtype=bool), k=1) # 遮挡一半
        
        sns.heatmap(dist_matrix, annot=True, fmt=".2f", cmap="magma",
                    xticklabels=style_names, yticklabels=style_names,
                    square=True)
        
        # 计算该尺度的“平均区分度” (非对角线均值)
        avg_sep = dist_matrix.sum() / (n * (n - 1))
        
        plt.title(f"Scale {scale}x{scale} Separability (Avg Dist: {avg_sep:.2f})")
        plt.tight_layout()
        
        save_path = out_dir / f"swd_scale_{scale}.png"
        plt.savefig(save_path)
        logger.info(f"  ✅ Saved heatmap: {save_path} | Avg Dist: {avg_sep:.4f}")

    logger.info(f"\nDone! Check the '{out_dir}' folder to pick your Golden Scales.")

if __name__ == "__main__":
    main()