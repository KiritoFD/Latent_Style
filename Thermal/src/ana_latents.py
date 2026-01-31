import torch
import json
import logging
import sys
from pathlib import Path
from torch.utils.data import DataLoader

# 挂载项目 src 目录
sys.path.append(str(Path(__file__).parent / "src"))
from dataset import LatentDataset

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("WeightEstimator")

def estimate_weights(config_path):
    # 1. 环境准备
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = LatentDataset(
        data_root=config['data']['data_root'],
        num_styles=config['model']['num_styles'],
        style_subdirs=config['data'].get('style_subdirs'),
        config=config
    )
    
    style_names = dataset.style_subdirs
    style_indices = dataset.style_indices
    variances = []

    logger.info(f"Using device: {device} | Analyzing {len(dataset.latents_tensor)} samples")

    # 2. 统计各风格方差 (Intra-class Variance)
    # 在计算机体系结构中，fp32 的求和比 fp16 更能减少舍入误差（Rounding Error）
    for style_id in range(len(style_names)):
        indices = style_indices[style_id]
        # [N, 4, 32, 32]
        latents = dataset.latents_tensor[indices].to(device).float()
        
        # 计算该风格全样本的全局方差
        # 物理意义：该风格在隐空间分布的“广度”
        var_i = torch.var(latents).item()
        variances.append(var_i)
        logger.info(f"Style '{style_names[style_id]}': Raw Variance = {var_i:.6f}")

    # 3. 计算逆方差权重 (Inverse Variance Weighting)
    # 目标：Weight_i * Var_i = Constant (使不同风格的 Loss 处于同一数量级)
    vars_tensor = torch.tensor(variances)
    avg_var = vars_tensor.mean()
    
    # 权重 = 全局平均方差 / 类内方差
    raw_weights = avg_var / (vars_tensor + 1e-8)
    
    # 归一化：使权重均值为 1.0，不改变全局学习率量级
    norm_weights = (raw_weights / raw_weights.mean()).tolist()
    
    # 4. 输出结果
    print("\n" + "="*40)
    print("📊 STATISTICAL WEIGHT RECOMMENDATION")
    print("="*40)
    print(f"{'Style Name':<15} | {'Weight':<10}")
    print("-" * 28)
    for name, weight in zip(style_names, norm_weights):
        print(f"{name:<15} | {weight:.4f}")
    
    print("\n[Action] Update your config.json:")
    print(f'\"style_weights\": {[round(w, 4) for w in norm_weights]}')
    print("="*40)

if __name__ == "__main__":
    estimate_weights("config.json")