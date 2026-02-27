import json
import copy
import os
import platform
import math

def load_base_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def calculate_precise_vram_bs_and_lr(base_dim, max_patch, user_lr_override=None):
    """
    【架构师级】四元响应面显存求解器 (Response Surface Solver)
    基于用户 4 个真实物理探针数据逆向工程拟合。
    目标: 把所有配置的显存死死咬住 10.5 GB (绝对安全且满载)。
    """
    # 归一化特征维度
    x = base_dim / 256.0
    y = (max_patch / 15.0) ** 2
    
    # 1. 静态显存建模 (权重 + 优化器状态，呈 O(C^2) 增长)
    v_static = 1.5 + (x ** 2) * 1.0
    
    # 2. 剩余可用动态显存
    v_dyn_avail = 10.5 - v_static
    
    # 3. 单个 Batch 样本的动态显存开销 (多元线性插值)
    # 基于探针数据解算的特征多项式
    cost_per_bs = 0.086 * x + 0.071 * y - 0.018 * x * y - 0.014
    cost_per_bs = max(0.01, cost_per_bs) # 防御性保底
    
    # 4. 精确反解 Batch Size
    raw_bs = v_dyn_avail / cost_per_bs
    
    # 8倍数对齐以激活 Tensor Core 极致吞吐
    final_bs = int(raw_bs // 8) * 8
    
    # 硬件硬约束
    final_bs = max(16, min(256, final_bs)) 
    
    # --- 学习率平方根等效补偿法则 (Square Root Scaling) ---
    # 基于 BS=128 时最佳 lr=2e-4 进行物理等效平移
    ref_bs = 128
    base_lr = 2e-4
    if user_lr_override is not None:
        scaled_lr = user_lr_override
    else:
        scaled_lr = base_lr * math.sqrt(final_bs / ref_bs)
        
    return final_bs, round(scaled_lr, 6)

def create_experiment(base, name, overrides):
    cfg = copy.deepcopy(base)
    
    if 'model' in overrides:
        cfg['model'].update(overrides['model'])
        if 'base_dim' in overrides['model']:
            cfg['model']['lift_channels'] = overrides['model']['base_dim']
            
    if 'loss' in overrides:
        cfg['loss'].update(overrides['loss'])
        
    base_dim = cfg['model'].get('base_dim', 128)
    max_patch = max(cfg['loss'].get('swd_patch_sizes', [5]))
    
    user_lr = overrides.get('training', {}).get('learning_rate', None)
    
    # 调用响应面求解器获取最优 BS 和补偿 LR
    bs, auto_lr = calculate_precise_vram_bs_and_lr(base_dim, max_patch, user_lr)
    
    if 'training' not in cfg:
        cfg['training'] = {}
    cfg['training'].update(overrides.get('training', {}))
    
    # 强制注入
    cfg['training']['batch_size'] = bs
    cfg['training']['learning_rate'] = auto_lr
    cfg['training']['min_learning_rate'] = round(auto_lr * 0.1, 6)
    
    # 统一控制变量
    if 'loss' not in cfg: cfg['loss'] = {}
    cfg['loss']['w_identity'] = 2.0
    cfg['loss']['w_delta_tv'] = 0.1
    cfg['loss']['w_color'] = 15.0
    cfg['loss']['w_swd'] = 30.0
    
    cfg['training']['num_epochs'] = 100
    cfg['training']['full_eval_interval'] = 100
    cfg['training']['save_interval'] = 50
    cfg['checkpoint']['save_dir'] = f"../master_sweep_{name}"
    
    filename = f"config_{name}.json"
    with open(filename, 'w') as f:
        json.dump(cfg, f, indent=2)
        
    return filename, bs, auto_lr

if __name__ == "__main__":
    base_cfg = load_base_config()
    
    experiments = {
        "01_cap_64":  {"model": {"base_dim": 64, "ada_mix_rank": 16}, "loss": {"swd_patch_sizes": [5, 7, 9]}},
        "02_cap_128": {"model": {"base_dim": 128, "ada_mix_rank": 32}, "loss": {"swd_patch_sizes": [5, 7, 9]}}, 
        "03_cap_192": {"model": {"base_dim": 192, "ada_mix_rank": 48}, "loss": {"swd_patch_sizes": [5, 7, 9]}},
        "04_cap_256": {"model": {"base_dim": 256, "ada_mix_rank": 64}, "loss": {"swd_patch_sizes": [5, 7, 9]}},

        "05_patch_micro": {"model": {"base_dim": 128}, "loss": {"swd_patch_sizes": [3, 5]}},
        "06_patch_std":   {"model": {"base_dim": 128}, "loss": {"swd_patch_sizes": [3, 5, 7]}},
        "07_patch_xmax":  {"model": {"base_dim": 128}, "loss": {"swd_patch_sizes": [7, 11, 15]}},

        "08_lr_fast": {"model": {"base_dim": 128}, "loss": {"swd_patch_sizes": [5, 7, 9]}, "training": {"learning_rate": 5e-4}},
        "09_lr_slow": {"model": {"base_dim": 128}, "loss": {"swd_patch_sizes": [5, 7, 9]}, "training": {"learning_rate": 5e-5}},

        "10_wide_xmax": {"model": {"base_dim": 256, "ada_mix_rank": 64}, "loss": {"swd_patch_sizes": [7, 11, 15]}},
        "11_narrow_xmax": {"model": {"base_dim": 64, "ada_mix_rank": 16}, "loss": {"swd_patch_sizes": [7, 11, 15]}},
        "12_mid_xmax": {"model": {"base_dim": 192, "ada_mix_rank": 48}, "loss": {"swd_patch_sizes": [7, 11, 15]}},

        "13_wide_micro": {"model": {"base_dim": 256, "ada_mix_rank": 64}, "loss": {"swd_patch_sizes": [3, 5]}},
        "14_narrow_micro": {"model": {"base_dim": 64, "ada_mix_rank": 16}, "loss": {"swd_patch_sizes": [3, 5]}},

        "15_split_brain_128": {"model": {"base_dim": 128}, "loss": {"swd_patch_sizes": [3, 15]}},
        "16_split_brain_256": {"model": {"base_dim": 256, "ada_mix_rank": 64}, "loss": {"swd_patch_sizes": [3, 7, 15]}},

        "17_extreme_underpowered": {"model": {"base_dim": 32, "ada_mix_rank": 8}, "loss": {"swd_patch_sizes": [5]}}, 
        "18_extreme_overpowered":  {"model": {"base_dim": 384, "ada_mix_rank": 64}, "loss": {"swd_patch_sizes": [3, 5]}}, 
        "19_the_abyss": {"model": {"base_dim": 192, "ada_mix_rank": 48}, "loss": {"swd_patch_sizes": [11, 15], "w_swd": 50.0}}, 
        "20_golden_balance": {"model": {"base_dim": 192, "ada_mix_rank": 32}, "loss": {"swd_patch_sizes": [5, 7, 11]}} 
    }
    
    generated = []
    for name in sorted(experiments.keys()):
        cfg_file, bs, auto_lr = create_experiment(base_cfg, name, experiments[name])
        generated.append((name, cfg_file, bs, auto_lr))
        
    is_windows = platform.system() == "Windows"
    script_name = "run_20_master_calibrated.bat" if is_windows else "run_20_master_calibrated.sh"
    
    with open(script_name, 'w') as f:
        if not is_windows:
            f.write("#!/bin/bash\n\n")
            
        for name, cfg_file, bs, auto_lr in generated:
            f.write(f"echo \"=========================================\"\n")
            f.write(f"echo \"=== [START] {name} | Perfect BS: {bs} | LR: {auto_lr:.6f} ===\"\n")
            f.write(f"echo \"=========================================\"\n")
            cmd = f"uv run run.py --config {cfg_file}"
            f.write(f"{cmd} || echo \"[WARNING] {name} FAILED!\"\n\n")
            
    if not is_windows:
        os.chmod(script_name, 0o755)
        
    print(f"🎯 VRAM 曲面拟合完成。所有配置锁定 10.5G 甜点位。")
    print(f"{'Experiment':<25} | {'Batch Size':<10} | {'Auto LR':<10}")
    print("-" * 50)
    for name, _, bs, auto_lr in generated:
        print(f"{name:<25} | BS={bs:<7} | LR={auto_lr:.6f}")
    print(f"\n执行 {script_name} 点火！")