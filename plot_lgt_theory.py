import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# ==========================================
# 1. 定义各种特征提取算子 (The Extractors)
# ==========================================

def get_gram(x):
    # x: [B, C, H, W]
    b, c, h, w = x.shape
    f = x.view(b, c, -1)
    # 归一化非常重要，否则不同尺度的 Gram 数值无法比较
    gram = torch.bmm(f, f.transpose(1, 2)) / (h * w)
    # 取上三角 (去冗余)
    idx = torch.triu_indices(c, c)
    return gram[:, idx[0], idx[1]]

class FeatureExtractors:
    def __init__(self, device):
        self.device = device
        # 随机投影器 (固定随机种子以保证复现性)
        torch.manual_seed(42)
        self.rand_proj_linear = nn.Linear(4 * 32 * 32, 128).to(device).eval()
        
        self.rand_proj_conv = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        ).to(device).eval()

    def gram_1x(self, x):
        return get_gram(x)

    def gram_2x_up(self, x):
        # 2倍上采样
        x_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return get_gram(x_up)

    def gram_4x_up(self, x):
        # 4倍上采样 (模拟 128x128 像素空间)
        x_up = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return get_gram(x_up)

    def gram_unfolded(self, x):
        # 局部 3x3 Gram (Unfolded)
        # [B, 4, 32, 32] -> [B, 36, 1024]
        b, c, h, w = x.shape
        patches = F.unfold(x, kernel_size=3, padding=1)
        # 计算 36x36 的超大 Gram
        patches = patches - patches.mean(dim=2, keepdim=True) # 中心化
        gram = torch.bmm(patches, patches.transpose(1, 2)) / (h * w)
        # 取对角线特征 (Variance) 和部分非对角线，为了降维取前 128 个特征
        return gram.view(b, -1)[:, :128] 

    def fft_spectrum(self, x):
        # 2D FFT
        fft = torch.fft.rfft2(x, norm="ortho")
        mag = torch.abs(fft)
        return mag.view(x.shape[0], -1)

    def swd_feature(self, x, num_projections=64):
        # SWD 核心特征：排序后的投影
        b, c, h, w = x.shape
        f = x.view(b, c, -1).permute(0, 2, 1) # [B, HW, C]
        
        # 生成随机投影方向
        projections = torch.randn(c, num_projections, device=x.device)
        projections = F.normalize(projections, p=2, dim=0)
        
        # 投影 -> [B, HW, num_proj]
        projected = torch.matmul(f, projections)
        
        # 排序 (Wasserstein 距离的核心) -> [B, HW, num_proj]
        sorted_proj, _ = torch.sort(projected, dim=1)
        
        # 下采样以减少特征维度 (取分位数)
        # 取 32 个分位点
        indices = torch.linspace(0, h*w-1, 32).long().to(x.device)
        return sorted_proj[:, indices, :].reshape(b, -1)

    def random_linear(self, x):
        return self.rand_proj_linear(x.view(x.shape[0], -1))

    def random_conv(self, x):
        return self.rand_proj_conv(x)

# ==========================================
# 2. 主评测流程
# ==========================================
def run_comprehensive_proof():
    data_root = Path("sdxl-256") # 你的数据目录
    styles = ["photo", "Hayao", "monet", "cezanne", "vangogh"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples = 250 # 这里的样本数足够统计显著性

    print(f"🔥 Starting Comprehensive Failure Proof on {device}...")
    
    # 1. 加载数据
    all_lats, labels = [], []
    for s_idx, style in enumerate(styles):
        files = sorted(list((data_root/style).glob("*.pt")))[:n_samples]
        for f in files:
            all_lats.append(torch.load(f, map_location=device))
            labels.append(style)
    
    # Stack -> [N, 4, 32, 32]
    X_raw = torch.stack(all_lats)
    print(f"📦 Loaded {len(X_raw)} latents.")

    # 2. 执行评测
    extractors = FeatureExtractors(device)
    methods = {
        "Gram (1x)": extractors.gram_1x,
        "Gram (Upsample 2x)": extractors.gram_2x_up,
        "Gram (Upsample 4x)": extractors.gram_4x_up,
        "Gram (Unfolded 3x3)": extractors.gram_unfolded,
        "FFT (Spectrum)": extractors.fft_spectrum,
        "SWD (Sorted Proj)": extractors.swd_feature,
        "Rand Proj (Linear)": extractors.random_linear,
        "Rand Proj (Conv)": extractors.random_conv
    }

    results = {}
    
    # 可视化布局
    rows = 2
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(24, 12))
    axes = axes.flatten()

    print("\n" + "="*60)
    print(f"{'Method':<25} | {'Silhouette Score':<15} | {'Status'}")
    print("-" * 60)

    for i, (name, func) in enumerate(methods.items()):
        # 提取特征
        with torch.no_grad():
            # 分批处理防止显存溢出 (针对 3060)
            feats = []
            batch_size = 128
            for j in range(0, len(X_raw), batch_size):
                batch = X_raw[j:j+batch_size]
                f = func(batch).cpu().numpy()
                feats.append(f)
            feats = np.concatenate(feats)

        # 标准化 (对聚类至关重要)
        feats = StandardScaler().fit_transform(feats)

        # 计算分数
        score = silhouette_score(feats, labels)
        
        # 状态判定
        status = "✅ PASS" if score > 0.1 else "❌ FAIL"
        if score < 0: status = "💀 HARD FAIL"
        
        print(f"{name:<25} | {score: .4f}          | {status}")
        results[name] = score

        # PCA 可视化
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(feats)
        
        ax = axes[i]
        for style in styles:
            mask = [l == style for l in labels]
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=style, s=8, alpha=0.6)
        ax.set_title(f"{name}\nScore: {score:.4f}")
        if i == 0: ax.legend()

    plt.tight_layout()
    plt.savefig("failure_proof_matrix.png")
    print("="*60)
    print("\n📉 可视化矩阵已保存至 'failure_proof_matrix.png'")
    print("💡 结论: 如果以上所有分数均为负或接近0，证明无监督统计量在 Latent 空间彻底失效。")
    print("   这就构成了我们需要训练 'Style Probe' 的充分必要条件。")

if __name__ == "__main__":
    run_comprehensive_proof()