import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

class DifferentialGramExtractor(nn.Module):
    """
    微分格拉姆特征提取器 (Differential Gram).
    数学原理:
    1. Lift: 解决流形坍缩 (4->64)
    2. Instance Norm: 解决能量/锥体干扰 (Projection to Unit Sphere)
    3. Gradient: 解决内容泄漏 (High-pass Filter)
    4. Gram: 捕捉纹理的二阶共现统计量 (Second-order Statistics)
    """
    def __init__(self, in_channels=4, hidden_dim=64, scale_factor=0.13025):
        super().__init__()
        self.scale_factor = scale_factor
        
        # 1. 随机正交投影 (Random Orthogonal Projection)
        # 这里的 1x1 卷积仅仅起到 "混合通道 + 升维" 的作用
        self.projector = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        
        # 正交初始化保证各向同性，不引入额外的归纳偏置
        nn.init.orthogonal_(self.projector.weight)
        
        self.projector.eval()
        for p in self.projector.parameters(): p.requires_grad = False

    def forward(self, x):
        # x: [B, 4, H, W]
        x = x * self.scale_factor
        
        # [Step 1] Lift (4 -> 64)
        feats = self.projector(x)
        
        # [Step 2] Whiten (Instance Norm)
        # 这一步至关重要！它强制所有样本的特征均值为0，方差为1。
        # 这样 Gram 矩阵测量的就是纯粹的 "相关性 (Correlation)" 而非 "内积 (Inner Product)"
        # 从而彻底消除了 "锥体" 问题。
        feats = F.instance_norm(feats)
        
        # [Step 3] Differentiate (计算梯度)
        # 使用简单的差分算子，比 Sobel 更适合 32x32 的小图，避免边界伪影
        # dx: f(x+1) - f(x)
        # dy: f(y+1) - f(y)
        dx = feats[..., :, 1:] - feats[..., :, :-1]
        dy = feats[..., 1:, :] - feats[..., :-1, :]
        
        # 这一步丢弃了最后一行/一列，为了对齐维度我们取交集区域
        # 实际上风格是平移不变的，这点损失无所谓
        
        # 将梯度视为特征
        # 我们关心的是：通道 i 的梯度 和 通道 j 的梯度 是否同步变化？
        b, c, h, w = feats.shape
        
        # Reshape to [B, C, N]
        dx_flat = dx.reshape(b, c, -1)
        dy_flat = dy.reshape(b, c, -1)
        
        # [Step 4] Differential Gram
        # 计算梯度的协方差矩阵
        # G_ij = sum( dx_i * dx_j + dy_i * dy_j )
        # 这捕捉了 "纹理走向" 的相关性
        
        # [B, C, C]
        gram_x = torch.bmm(dx_flat, dx_flat.transpose(1, 2))
        gram_y = torch.bmm(dy_flat, dy_flat.transpose(1, 2))
        
        # 归一化 (虽然 InstanceNorm 已经做过了，但除以元素个数保持数值范围是个好习惯)
        gram = (gram_x + gram_y) / (h * w)
        
        # 取上三角并展平
        idx = torch.triu_indices(c, c, device=x.device)
        return gram[:, idx[0], idx[1]]

def run_analysis(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root) if args.data_root else (Path(__file__).resolve().parents[1] / "sdxl-256")
    styles = ["photo", "Hayao", "vangogh", "monet", "cezanne"]

    extractor = DifferentialGramExtractor(in_channels=4, hidden_dim=64).to(device)

    all_feats = []
    all_labels = []
    print(f"Running Differential Gram Analysis on {device}...")
    
    with torch.no_grad():
        for style in styles:
            files = list((data_root / style).glob("*.pt"))[:500]
            if not files: continue

            feats_list = []
            for i in range(0, len(files), 64):
                batch_files = files[i : i + 64]
                x = torch.stack([torch.load(f, map_location=device, weights_only=True) for f in batch_files])
                
                # 提取特征
                g = extractor(x)
                feats_list.append(g.cpu().numpy())

            if feats_list:
                all_feats.append(np.concatenate(feats_list))
                all_labels.extend([style] * len(files))
                print(f"Processed {style}")

    if not all_feats: return

    X = np.concatenate(all_feats)
    # 再次标准化 PCA 输入
    X_norm = StandardScaler().fit_transform(X)

    score = silhouette_score(X_norm, all_labels)
    print(f"\nSilhouette Score: {score:.4f}")

    pca = PCA(n_components=2).fit_transform(X_norm)
    plt.figure(figsize=(10, 8))
    labels_np = np.array(all_labels)
    for style in styles:
        mask = labels_np == style
        if mask.any():
            plt.scatter(pca[mask, 0], pca[mask, 1], label=style, s=15, alpha=0.6)

    plt.title(f"Differential Gram (Gradient Correlations)\nScore: {score:.4f}")
    plt.legend()
    plt.savefig("diff_gram_pca.png")
    print("Saved diff_gram_pca.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=None)
    args = parser.parse_args()
    run_analysis(args)