import torch
import torch.nn.functional as F

def verify_pyramid_robustness():
    device = 'cuda'
    # 1. 模拟一个原始 Latent (结构)
    # 假设 B=1, C=4, H=W=32
    z_orig = torch.randn(1, 4, 32, 32, device=device)
    
    # 2. 模拟一个被“风格化”后的 Latent (加入极强的高频纹理干扰)
    # 我们加上强烈的随机高频噪声，模拟笔触错位
    noise = torch.randn_like(z_orig) * 0.8 
    z_stylized = z_orig + noise
    
    print(f"{'Resolution':<12} | {'MSE (Normalized)':<15}")
    print("-" * 30)

    # 3. 比较不同尺度下的偏差
    scales = [32, 16, 8, 4]
    for s in scales:
        # 使用 area 模式进行下采样
        p1 = F.interpolate(z_orig, size=(s, s), mode='area')
        p2 = F.interpolate(z_stylized, size=(s, s), mode='area')
        
        mse = F.mse_loss(p1, p2).item()
        # 归一化显示：相对于原图噪声水平的比例
        print(f"{s:>2}x{s:<2}      | {mse:.6f}")

if __name__ == "__main__":
    verify_pyramid_robustness()