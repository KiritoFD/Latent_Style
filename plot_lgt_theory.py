import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from tqdm import tqdm

def compute_gram_feature(x: torch.Tensor) -> torch.Tensor:
    # x: [C, H, W]
    c, h, w = x.shape
    x_flat = x.view(c, -1) # [C, HW]
    gram = torch.mm(x_flat, x_flat.t()) # [C, C]
    return gram.view(-1) / (c * h * w)

def compute_swd_feature(x: torch.Tensor, num_projections: int = 128) -> torch.Tensor:
    # x: [C, H, W] -> Patch-based distribution
    c, h, w = x.shape
    # 模拟 losses.py 中的 3x3 patch 采样
    patches = F.unfold(x.unsqueeze(0), kernel_size=3, padding=1).squeeze(0) # [C*9, L]
    patches = patches.t() # [L, dim]
    dim = patches.shape[-1]
    
    # 随机投影
    projections = torch.randn(dim, num_projections, device=x.device)
    projections = F.normalize(projections, p=2, dim=0)
    projected = torch.matmul(patches, projections) # [L, num_projections]
    
    # 排序作为特征向量 (Distribution Matching 的核心)
    sorted_proj, _ = torch.sort(projected, dim=0)
    return sorted_proj.t().reshape(-1)

@torch.no_grad()
def main():
    data_root = Path("sdxl-256")
    styles = ["photo", "Hayao", "monet", "cezanne", "vangogh"]
    samples_per_style = 200 # 每种风格采样 200 张，保证统计显著性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features_gram = []
    features_swd = []
    labels = []

    print(f"Reading latents from {data_root}...")
    for style_idx, style_name in enumerate(styles):
        style_dir = data_root / style_name
        files = sorted(list(style_dir.glob("*.pt")))[:samples_per_style]
        
        for f in tqdm(files, desc=f"Style: {style_name}"):
            latent = torch.load(f).to(device) # [4, 32, 32]
            
            # 1. 计算 Gram 特征
            f_gram = compute_gram_feature(latent)
            features_gram.append(f_gram.cpu().numpy())
            
            # 2. 计算 SWD 特征 (取均值分布以降低维度压力)
            f_swd = compute_swd_feature(latent)
            features_swd.append(f_swd.cpu().numpy())
            
            labels.append(style_name)

    # 可视化准备
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for ax, data, title in zip(axes, [features_gram, features_swd], ["Gram Matrix PCA", "SWD (Projected Sort) PCA"]):
        data_np = np.array(data)
        # PCA 降维到 2D
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(data_np)
        
        for i, style_name in enumerate(styles):
            idx = [j for j, l in enumerate(labels) if l == style_name]
            ax.scatter(reduced[idx, 0], reduced[idx, 1], c=colors[i], label=style_name, alpha=0.6, s=10)
        
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("style_feature_analysis.png")
    print("Done. Analysis saved to style_feature_analysis.png")

if __name__ == "__main__":
    main()