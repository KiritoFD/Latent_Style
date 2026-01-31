import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader
import sys
import os
import logging
import json

# Add src to path to handle imports correctly (since trainer.py uses absolute imports)
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.trainer import LGTTrainer
from src.dataset import LatentDataset 

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_gradient_norm(model, loss):
    """
    计算特定 Loss 对模型参数产生的梯度范数 (L2 Norm)。
    这是衡量 Loss '拉力' 大小的物理量。
    """
    model.zero_grad()
    loss.backward(retain_graph=True)
    
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def auto_balance_weights(trainer, dataloader, device='cuda'):
    """
    [Strategy 1] 梯度范数平衡探测
    自动计算 w_mse 应该设定为多少，才能让结构梯度的力度 = 风格梯度的力度。
    """
    logger.info("🧪 Starting Gradient Norm Balancing...")
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        logger.error("Dataloader is empty!")
        return
    
    # 移动数据到设备
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    # 1. 获取各个 Loss 的原始值 (权重设为 1.0)
    trainer.model.eval() # 冻结 BN 层，保持确定性
    
    # 模拟 compute_energy_loss 的前向过程
    epoch = 0
    multipliers = (1.0, 1.0, 1.0) # 无调度干扰
    
    # --- 这是一个模拟的 forward pass，为了拿到独立的 loss ---
    # 强制 FP32 以获得准确梯度
    with torch.amp.autocast('cuda', enabled=False):
        ld = trainer.compute_energy_loss(batch, epoch, multipliers)
        
        # 提取原始 Loss
        loss_mse_raw = ld['mse'] 
        loss_style_raw = ld['style_swd']
        
    logger.info(f"Raw MSE Loss: {loss_mse_raw.item():.6f}")
    logger.info(f"Raw Style Loss: {loss_style_raw.item():.6f}")

    # 2. 计算梯度范数
    # 我们关注 UNet 中层的梯度，避免输入输出层的干扰
    norm_mse = compute_gradient_norm(trainer.model, loss_mse_raw)
    norm_style = compute_gradient_norm(trainer.model, loss_style_raw)
    
    logger.info(f"Gradient Norm (MSE):   {norm_mse:.6f}")
    logger.info(f"Gradient Norm (Style): {norm_style:.6f}")
    
    if norm_mse == 0 or norm_style == 0:
        logger.error("❌ Gradient is zero. Check model connectivity.")
        return

    # 3. 计算平衡比例
    # 目标： w_mse * norm_mse = w_style * norm_style
    # 假设 w_style = 1.0，推导 w_mse
    suggested_w_mse = norm_style / (norm_mse + 1e-8)
    
    print("\n" + "="*40)
    print(f"⚖️  Auto-Balancing Result")
    print(f"To balance gradients (assuming w_style=1.0):")
    print(f"Recommended w_mse = {suggested_w_mse:.4f}")
    print("="*40 + "\n")
    
    return suggested_w_mse

def lr_finder(trainer, dataloader, min_lr=1e-7, max_lr=1e-2, num_steps=100, device='cuda'):
    """
    [Strategy 2] 学习率探测器 (Leslie Smith Approach)
    从极小 LR 开始指数增加，找到 Loss 下降最快的点。
    """
    logger.info("🧪 Starting LR Finder...")
    trainer.model.train()
    optimizer = trainer.optimizer
    
    # 备份模型状态
    model_state = deepcopy(trainer.model.state_dict())
    optim_state = deepcopy(optimizer.state_dict())
    
    lrs = []
    losses = []
    
    lr_lambda = lambda step: (max_lr / min_lr) ** (step / num_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 初始化 LR
    for g in optimizer.param_groups:
        g['lr'] = min_lr
        
    iter_loader = iter(dataloader)
    best_loss = float('inf')
    
    for step in range(num_steps):
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(dataloader)
            batch = next(iter_loader)
            
        # 移动数据
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        optimizer.zero_grad()
        
        # 使用当前配置计算 Loss (模拟真实训练环境，使用 BF16)
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            ld = trainer.compute_energy_loss(batch, 0) 
            loss = ld['total']
        
        # 记录
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        losses.append(loss.item())
        
        # 检查发散
        if step > 10 and loss.item() > 4 * best_loss:
            logger.info(f"Loss diverged at step {step}, stopping early.")
            break
            
        if loss.item() < best_loss:
            best_loss = loss.item()
            
        # 反向传播
        trainer.scaler.scale(loss).backward()
        trainer.scaler.step(optimizer)
        trainer.scaler.update()
        scheduler.step()
        
    # 恢复模型
    trainer.model.load_state_dict(model_state)
    optimizer.load_state_dict(optim_state)
    
    # 绘图与分析
    def smooth(scalars, weight=0.8):
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    smoothed_losses = smooth(losses)
    
    # 寻找下降最陡峭的点（梯度最小的负值）
    grads = np.gradient(smoothed_losses)
    steepest_idx = np.argmin(grads)
    suggested_lr = lrs[steepest_idx]
    
    print("\n" + "="*40)
    print(f"🚀 LR Finder Result")
    print(f"Steepest descent LR: {suggested_lr:.2e}")
    print(f"Suggested Max LR (Safety 10x lower): {suggested_lr / 10:.2e}")
    print("="*40 + "\n")
    
    # 保存图表
    plt.figure()
    plt.plot(lrs, losses, label='Raw')
    plt.plot(lrs, smoothed_losses, label='Smoothed')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('LR Finder')
    plt.legend()
    plt.savefig('lr_finder_plot.png')
    logger.info("LR plot saved to lr_finder_plot.png")

if __name__ == "__main__":
    print("🚀 Auto-tune script starting...")
    
    # 加载配置
    config_path = "src/config.json"
    if not os.path.exists(config_path):
        logger.error(f"Config not found at {config_path}")
        sys.exit(1)
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化 Trainer (这里只做探测，不需要 resume checkpoint)
    config['training']['resume_checkpoint'] = None 
    # 🔥 关键修复：禁用编译，防止长时间无输出的等待
    config['training']['use_compile'] = False
    
    # 修正相对路径：如果 config 是为 src/ 目录设计的 (../../)，在 Thermal/ 目录下运行时需要调整为 (../)
    if config['data']['data_root'].startswith('../../'):
        config['data']['data_root'] = config['data']['data_root'].replace('../../', '../')
        logger.info(f"Adjusted data_root to: {config['data']['data_root']}")
    
    trainer = LGTTrainer(config, device=device)
    
    # 准备一个小 Batch 数据
    logger.info("Loading dataset for tuning...")
    dataset = LatentDataset(
        data_root=config['data']['data_root'],
        num_styles=config['model']['num_styles'],
        style_subdirs=config['data'].get('style_subdirs'),
        config=config
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0) # 小 Batch 即可
    
    # Initialize trainer dataset reference needed for sampling styles
    trainer.build_style_indices_cache(dataset)
    
    # 1. 测算权重平衡
    auto_balance_weights(trainer, dataloader, device)
    
    # 2. 测算学习率
    # 注意：在运行 LR Finder 前，最好把权重先设置为上面测算出的平衡值
    lr_finder(trainer, dataloader, device=device)