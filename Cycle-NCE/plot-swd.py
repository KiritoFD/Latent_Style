import argparse
import itertools
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings

# 忽略 sklearn 的一些除零警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 终极参数化特征提取器
# ==========================================
class ParametricExtractor(nn.Module):
    def __init__(self, in_channels=4, config=None):
        super().__init__()
        self.cfg = config or {}
        dim = self.cfg.get('dim', 64)
        kernel = self.cfg.get('kernel', 1)
        
        # 1. Lift Layer (Projection)
        # 1x1 卷积保持空间结构，3x3 卷积引入平滑
        padding = kernel // 2
        self.projector = nn.Conv2d(in_channels, dim, kernel_size=kernel, padding=padding, bias=False)
        
        # 初始化：正交初始化是目前表现最好的
        nn.init.orthogonal_(self.projector.weight)
            
        self.projector.eval()
        for p in self.projector.parameters(): p.requires_grad = False

    def forward(self, x):
        # Scale (Fixed SDXL factor)
        x = x * 0.13025
        
        # 1. Lift (升维)
        feats = self.projector(x)
        
        # 2. Activation (非线性激活) - 关键变量
        act = self.cfg.get('act', 'relu')
        if act == 'relu': feats = F.relu(feats)
        elif act == 'leaky': feats = F.leaky_relu(feats, 0.2)
        elif act == 'gelu': feats = F.gelu(feats)
        elif act == 'silu': feats = F.silu(feats) # Swish
        elif act == 'tanh': feats = torch.tanh(feats)
        
        # 3. Normalization (白化) - 关键变量
        # Instance Norm 已经证明是消除 "锥体效应" 的核心
        feats = F.instance_norm(feats)
        
        # 4. Feature Extraction (微分)
        scales = self.cfg.get('scales', [1, 2])
        grams = []
        
        # Differential Gram (Gradients)
        for s in scales:
            # 避免边界伪影，进行切片
            if s >= feats.shape[-1]: continue
            
            # dx: 宽度方向梯度
            dx = feats[..., :, s:] - feats[..., :, :-s]
            # dy: 高度方向梯度
            dy = feats[..., s:, :] - feats[..., :-s, :]
            
            # 计算梯度的 Gram 矩阵
            grams.append(self._gram(dx))
            grams.append(self._gram(dy))
                
        if len(grams) == 0:
            return torch.zeros(x.shape[0], 1, device=x.device)

        return torch.cat(grams, dim=1)

    def _gram(self, x):
        b, c, h, w = x.shape
        f = x.reshape(b, c, -1)
        # Normalized Gram matrix
        g = torch.bmm(f, f.transpose(1, 2)) / (h * w)
        
        # Flatten upper triangle
        idx = torch.triu_indices(c, c, device=x.device)
        return g[:, idx[0], idx[1]]


# ==========================================
# 2. 全量数据加载与搜索
# ==========================================
def run_full_grid_search():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path("../sdxl-256") # <--- 请再次确认你的路径
    styles = ["photo", "Hayao", "vangogh", "monet", "cezanne"]
    
    print(f"🚀 Initializing Full-Data Grid Search on {device}...")
    
    # --- A. 全量加载数据 ---
    all_latents = []
    all_labels = []
    
    total_files = 0
    print("Loading ALL latents into VRAM (this might take a moment)...")
    
    for s in styles:
        # 不再切片 [:400]，加载所有数据
        files = list((data_root / s).glob("*.pt"))
        if not files: 
            print(f"Warning: No files found for style '{s}'")
            continue
            
        print(f"  - Loading {len(files)} files for '{s}'...")
        
        # 批量加载以显示进度
        chunk_size = 200
        style_latents = []
        for i in range(0, len(files), chunk_size):
            chunk = files[i:i+chunk_size]
            lats = torch.stack([torch.load(f, map_location=device, weights_only=True) for f in chunk])
            style_latents.append(lats)
            
        all_latents.append(torch.cat(style_latents, dim=0))
        all_labels.extend([s] * len(files))
        total_files += len(files)
    
    if not all_latents:
        print("❌ Error: No data found!")
        return

    X_raw = torch.cat(all_latents, dim=0) # [Total, 4, 32, 32]
    labels_np = np.array(all_labels)
    print(f"✅ Data loaded successfully. Total shape: {X_raw.shape}")
    print(f"   Memory usage: {X_raw.element_size() * X_raw.nelement() / 1024**2:.2f} MB")
    
    # --- B. 定义精细化搜索空间 ---
    search_space = {
        # 维度：在 64 附近进行更密集的搜索
        'dim': [48, 64, 80, 96, 128],
        
        # 卷积核：1x1 是目前的王者，保留 3x3 做对照
        'kernel': [1, 3],
        
        # 激活函数：引入现代激活函数
        'act': ['relu', 'leaky', 'gelu', 'silu'],
        
        # 尺度：测试更大的感受野
        'scales': [[1, 2], [1, 2, 3], [1, 2, 4]],
        
        # 固定参数 (基于之前的胜者)
        'norm': ['instance'],
        'mode': ['diff'],
        'init': ['orthogonal']
    }
    
    keys = list(search_space.keys())
    values = list(search_space.values())
    combinations = list(itertools.product(*values))
    
    print(f"\n🔍 Starting Grid Search with {len(combinations)} configurations...")
    print("=" * 80)
    print(f"{'Score':<10} | {'Dim':<4} | {'K':<1} | {'Act':<6} | {'Scales':<10}")
    print("-" * 80)
    
    results = []
    batch_size = 256 # 全量数据推理，批次大一点
    
    # --- C. 暴力循环 ---
    for combo in tqdm(combinations, desc="Searching"):
        cfg = dict(zip(keys, combo))
        
        extractor = ParametricExtractor(in_channels=4, config=cfg).to(device)
        
        # Extract features
        feats_list = []
        with torch.no_grad():
            for i in range(0, len(X_raw), batch_size):
                batch = X_raw[i : i+batch_size]
                f = extractor(batch)
                feats_list.append(f.cpu().numpy())
                
        X_feats = np.concatenate(feats_list, axis=0)
        
        # Score Calculation
        if np.isnan(X_feats).any():
            score = -1.0
        else:
            try:
                # 标准化对距离度量至关重要
                scaler = StandardScaler()
                X_norm = scaler.fit_transform(X_feats)
                # 全量数据的 Silhouette Score
                score = silhouette_score(X_norm, labels_np)
            except:
                score = -1.0
        
        results.append((score, cfg))
        
        # 实时高亮显示突破 0.04 的结果
        if score > 0.038:
            tqdm.write(f"🌟 {score:.4f}  | {cfg['dim']:<4} | {cfg['kernel']} | {cfg['act']:<6} | {str(cfg['scales']):<10}")

    # --- D. 最终报告 ---
    print("\n" + "="*80)
    print("🏆 TOP 20 CONFIGURATIONS (FULL DATA)")
    print("="*80)
    
    # Sort by score descending
    results.sort(key=lambda x: x[0], reverse=True)
    
    for i, (score, cfg) in enumerate(results[:20]):
        print(f"Rank {i+1:02d}: Score={score:.5f}")
        print(f"  Config: {cfg}")
        print("-" * 40)
        
    # 保存最佳配置到文件，方便后续直接读取
    best_cfg = results[0][1]
    with open("best_style_config.txt", "w") as f:
        f.write(str(best_cfg))
    print(f"\n✅ Best configuration saved to 'best_style_config.txt'")

if __name__ == "__main__":
    run_full_grid_search()