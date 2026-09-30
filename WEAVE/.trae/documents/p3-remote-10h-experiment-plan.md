# P3 远程 GPU 10 小时实验计划：突破平凡解、解决雾化

> 目标：基于 `docs/620/fog/theory/trivial_solution_unified.md` 理论框架，在远程 GPU 上系统性地探索未验证的高 ROI 方向，打破平凡解、降低雾化、提升 clip_style (>0.72) 和 LPIPS (<0.40)。

## 1. 当前状态分析

### 1.1 已验证的基线（本地 GPU，3 epoch）

| 实验 | clip_style (all/transfer) | LPIPS | velocity_abs | 模式 |
|------|---------------------------|-------|--------------|------|
| P3-A wflow_scale=0.5 | **0.7019 / 0.6787** | 0.506 | 0.89 | latent, fixed_one, FiLM |
| P3-E contrastive_w=0.5 | **0.7036 / 0.6687** | **0.262** | 0.35 | DINO, contrastive |

**关键观察：**
- P3-E 的 LPIPS (0.262) 远优于 P3-A (0.506)，但使用了 DINO 且在 epoch 1 后退化
- P3-A 的 velocity magnitude (0.89) 更健康，说明 FM reduction 有效
- 两者都只训练了 3 epoch，远未收敛
- **最优组合尚未测试**：将 P3-A 的好参数与 P3-E 的对比损失结合

### 1.2 理论文档中的未探索方向（按 ROI 排序）

| 优先级 | 方面 | 理论依据 | 实现状态 |
|--------|------|---------|----------|
| **P0** | FiLM 大初始化 | 打破第 4 层"零初始化"保守机制 | **未尝试** (film_init_std=0) |
| **P0** | 两阶段训练 | 打破"SWD 平坦条件"，先强注入再微调 | **未在远程跑过** |
| **P1** | 更长训练 (10+ ep) | 3 epoch 不够跳出保守盆地 | **未尝试** |
| **P1** | 最优组合叠加 | 乘积效应需要同时打破 >=2 个条件 | **部分尝试** |
| **P2** | AdaIN 推理后处理 | 零成本去雾（已验证有效） | **本地验证过，未集成到远程 eval** |

## 2. 远程环境配置

### 2.1 连接信息
```bash
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62
# WSL 环境，数据集在 /mnt/i/
```

### 2.2 路径映射
| 用途 | 本地路径 | 远程路径 |
|------|---------|----------|
| 项目代码 | `G:\GitHub\Latent_Style\SchrodingerBridge` | `/mnt/i/Github/Latent_Style/SchrodingerBridge` |
| 训练数据 | `I:\wikiart_distinct5_samam_512_latents_ema\train` | `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train` |
| 测试图片 | `I:\wikiart_distinct5_samam_512_classview\test` | `/mnt/i/wikiart_distinct5_samam_512_classview/test` |
| VAE 缓存 | `G:\GitHub\Latent_Style\eval_cache` | `/mnt/i/Github/Latent_Style/eval_cache` |

### 2.3 固定约束（来自 project_memory）
- `batch_size = 24`（12GB VRAM 安全）
- `num_workers = 0`, `pin_memory = False`, `persistent_workers = False`
- `virtual_length_multiplier = 1.0`
- `test_image_dir` = classview test（非 overfit50）
- `num_epochs`: 实验用 10，最终评估可调整
- `PYTHONPATH` 不要手动设置（由 run.py 处理）
- 运行脚本需有错误处理（if...then...else）
- 日志保存到 `exp/p3_remote_10h/<name>/focused.log`

## 3. 实验时间表（10 小时窗口）

### 时间分配策略

每个实验约 **2 小时**（含训练 + 评估 + 图片生成），共 **5 轮实验**。
每轮结束后必须：
1. 检查 `summary_grid.png` 是否生成
2. 目视检查生成图片的质量（雾化程度、风格强度、内容保持）
3. 记录定量指标（clip_style, LPIPS, velocity_ratio）
4. 根据诊断结果决定是否调整下一轮参数

```
时间线（假设 T=0 开始）：
T+0h   ┃ R1: 基线组合长训练 (10 ep)          ┃ 预计 1.5-2h
T+2h   ┃ [目视检查 R1 图片]                    ┃ 15 min
T+2.25 ┃ R2: FiLM 大初始化 (10 ep)           ┃ 预计 1.5-2h
T+4.25 ┃ [目视检查 R2 图片]                   ┃ 15 min
T+4.5  ┃ R3: 两阶段训练 (10 ep)              ┃ 预计 1.5-2h
T+6.5  ┃ [目视检查 R3 图片]                   ┃ 15 min
T+6.75 ┃ R4: 最优组合激进版 (10 ep)          ┃ 预计 1.5-2h
T+8.75 ┃ [目视检查 R4 图片]                  ┃ 15 min
T+9    ┃ R5: 最佳 checkpoint + AdaIN 评估     ┃ 预计 0.5-1h
T+10h  ┃ 汇总所有结果，输出最终报告            ┃
```

## 4. 各实验详细配置

### R1: 基线组合长训练（建立收敛基线）

**目的**：验证当前最优参数组合在 10 epoch 下的收敛表现，作为后续对比基准。

**基于**：P3-A (wflow_scale=0.5) 配置，扩展到 10 epoch。

**配置差异（相对于 P3-A）**：
```json
{
  "model": {
    "style_gate_mode": "fixed_one",
    "style_film_enabled": true,
    "endpoint_film_enabled": true,
    "endpoint_film_use_norm": false,
    "endpoint_film_init_std": 0.0,
    "style_condition_source": "latent",
    "film_init_std": 0.0,
    "endpoint_head_mode": "endpoint_lowhigh",
    "style_attn_num_tokens": 256,
    "inference_adain": false
  },
  "bridge": {
    "w_flow_scale": 0.5,
    "w_velocity_magnitude": 0.5,
    "w_style_strength_reg": 0.5,
    "w_contrast_preserve": 1.0,
    "w_channel_variance": 0.05,
    "w_hf_energy": 1.0,
    "w_pixel_color_match": 1.0,
    "single_step_swd_weight": 8.0,
    "training_objective_mode": "velocity"
  },
  "training": {
    "batch_size": 24,
    "num_epochs": 10,
    "learning_rate": 2e-4,
    "full_eval_each_epoch": true,
    "full_eval_save_summary_grid": true
  }
}
```

**成功标准**：
- clip_style (transfer) > 0.68
- LPIPS < 0.45
- velocity_ratio > 0.7
- 训练曲线不出现早期退化

**诊断重点**：
- 是否存在"先好后坏"的模式？（epoch 最好 → 后续退化）
- 各 style 的 clip_style 差异（Minimalism 是否仍然失败？）
- 雾化程度是否随训练轮次变化？

---

### R2: FiLM 大初始化（打破零初始化保守机制）

**目的**：验证理论文档第 4 层保守机制的修复效果。大初始化让 FiLM 层从一开始就有非平凡的 style 注入能力，避免从 identity 附近开始优化。

**理论依据**（trivial_solution_unified.md Section 3.1 Layer 4）：
> 零初始化使 model ≈ identity（恒等映射），Loss 景观在 identity 附近有局部最优。如果保守解附近的 loss 梯度很小，模型就"走不出去"。

**配置差异（相对于 R1）**：
```json
{
  "model": {
    "film_init_std": 0.05,
    "endpoint_film_init_std": 0.1
  }
}
```

**其他参数**：与 R1 完全相同（控制变量）

**预期效果**：
- 训练初期 style 注入更强 → 更快跳出保守盆地
- endpoint_alpha 应该更高（更大的初始位移）
- 风险：初始化过大可能导致训练初期不稳定

**成功标准**：
- 相比 R1，epoch 1-3 的 clip_style 提升 > 0.02
- 最终收敛指标不低于 R1
- 训练不出现 NaN/divergence

---

### R3: 两阶段训练（课程学习跳出保守盆地）

**目的**：实现理论文档方案 C1——先强制风格注入，再微调内容平衡。

**理论依据**（trivial_solution_unified.md Section 4.2 C1）：
> Stage 1：高 SWD / 低 FM → 强制 style 注入
> Stage 2：正常权重 → 微调内容平衡
> 类比：先把球踢出去，再调整准度

**配置差异（相对于 R1）**：
```json
{
  "bridge": {
    "two_stage_enabled": true,
    "two_stage_s1_epochs": 3,
    "two_stage_s1_w_endpoint_content": 0.3,
    "two_stage_s1_w_endpoint_style": 16.0,
    "two_stage_s1_w_style_strength_reg": 1.0,
    "two_stage_s2_w_endpoint_content": 1.0,
    "two_stage_s2_w_endpoint_style": 8.0,
    "two_stage_s2_w_style_strength_reg": 0.5,
    "training_objective_mode": "endpoint",
    "w_endpoint_velocity_reg": 0.1,
    "w_flow_scale": 0.3
  },
  "training": {
    "num_epochs": 10
  }
}
```

**注意**：R3 切换到 `endpoint` 训练模式（不是 velocity），因为两阶段设计是围绕 endpoint loss 的。

**预期效果**：
- Stage 1 (ep 1-3)：style 注入激进的，clip_style 快速上升，LPIPS 可能偏高
- Stage 2 (ep 4-10)：内容保持逐渐恢复，LPIPS 下降
- 整体应优于相同总 epoch 数的单阶段训练

**成功标准**：
- Stage 1 结束时 (ep 3) clip_style > R1 同期
- 最终 (ep 10) LPIPS < R1 最终值
- 两阶段交接处没有灾难性的指标跳变

**风险**：
- Endpoint 模式在之前短训练 (3ep) 中表现不如 velocity 模式
- 但 10 epoch 可能足够让 endpoint 模式收敛

---

### R4: 最优组合激进版（同时打破多个条件）

**目的**：选取 R1-R3 中表现最好的配置，进一步激进化，同时打破 >=2 个平凡解条件。

**策略**：如果 R2 (FiLM 大初始化) 表现最好，则以 R2 为基础叠加两阶段；如果 R1 最好，则增加对比损失权重。

**候选配置 A（如果 R2 最优）**：
```json
{
  "model": {
    "film_init_std": 0.05,
    "endpoint_film_init_std": 0.1,
    "inference_adain": false
  },
  "bridge": {
    "w_flow_scale": 0.3,
    "w_velocity_magnitude": 0.8,
    "w_style_strength_reg": 0.8,
    "w_contrast_preserve": 1.5,
    "w_hf_energy": 1.5,
    "w_pixel_color_match": 1.5,
    "w_style_contrastive": 0.3,
    "contrastive_margin": 0.05,
    "single_step_swd_weight": 10.0,
    "two_stage_enabled": false
  },
  "training": {
    "num_epochs": 10
  }
}
```

**候选配置 B（如果 R3 最优）**：
```json
{
  "model": {
    "film_init_std": 0.05,
    "endpoint_film_init_std": 0.1
  },
  "bridge": {
    "two_stage_enabled": true,
    "two_stage_s1_epochs": 4,
    "two_stage_s1_w_endpoint_style": 20.0,
    "two_stage_s1_w_style_strength_reg": 1.5,
    "w_flow_scale": 0.2,
    "w_style_contrastive": 0.2,
    "single_step_swd_weight": 12.0
  },
  "training": {
    "num_epochs": 10
  }
}
```

**具体配置将在 R1-R3 结果出来后动态决定**（见第 6 节决策树）。

---

### R5: 最佳 checkpoint + AdaIN 评估

**目的**：对 R1-R4 中最好的 checkpoint 开启 AdaIN 推理后处理，零成本去雾。

**理论基础**（来自之前的实验验证）：
> AdaIN 将 target latent 的 channel-wise 统计量迁移到 generated latent 上，零成本显著去雾。饱和度提升 40-70%，雾化 score 从 9/10 降到 ~3/10。

**方法**：
```json
{
  "model": {
    "inference_adain": true
  }
}
```

只需在最佳 checkpoint 的 config 中设置 `inference_adain: true`，重新运行 evaluation（不需要重新训练！）。

**预期效果**：
- LPIPS 可能略微上升（AdaIN 改变 pixel 分布）
- clip_style 应该显著提升（更饱和的色彩 = 更强的风格信号）
- 视觉上雾化明显减轻

## 5. 运行脚本模板

### 5.1 单实验运行脚本

```bash
#!/bin/bash
# p3_remote_run.sh <experiment_name> <config_json>
# Usage: bash p3_remote_run.sh r1_baseline_long exp/p3_remote_10h/r1_baseline_long/config.json

set -e
EXP_NAME=$1
CONFIG_PATH=$2
LOG_FILE="exp/p3_remote_10h/${EXP_NAME}/focused.log"
PROJECT_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge"

cd "$PROJECT_DIR"

echo "========================================" | tee -a "$LOG_FILE"
echo "Starting experiment: $EXP_NAME" | tee -a "$LOG_FILE"
echo "Time: $(date)" | tee -a "$LOG_FILE"
echo "Config: $CONFIG_PATH" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# 运行训练（含每 epoch 自动评估）
if python run.py --config "$CONFIG_PATH" >> "$LOG_FILE" 2>&1; then
    echo "[$(date)] Training COMPLETED for $EXP_NAME" | tee -a "$LOG_FILE"
else
    echo "[$(date)] Training FAILED for $EXP_NAME" | tee -a "$LOG_FILE"
    exit 1
fi

# 确保生成了 summary_grid
EVAL_DIR=$(ls -d exp/p3_remote_10h/${EXP_NAME}/full_eval/epoch_* 2>/dev/null | tail -1)
if [ -n "$EVAL_DIR" ]; then
    echo "Latest eval dir: $EVAL_DIR" | tee -a "$LOG_FILE"
    if [ -f "${EVAL_DIR}/summary_grid.png" ]; then
        echo "summary_grid.png EXISTS ✓" | tee -a "$LOG_FILE"
    else
        echo "WARNING: summary_grid.png NOT FOUND, regenerating..." | tee -a "$LOG_FILE"
        # 尝试重新运行评估以生成图片
        python run.py --config "$CONFIG_PATH" --eval-only >> "$LOG_FILE" 2>&1 || true
    fi
fi

echo "========================================" | tee -a "$LOG_FILE"
echo "Finished experiment: $EXP_NAME" | tee -a "$LOG_FILE"
echo "Time: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
```

### 5.2 tmux 会话管理

```bash
# 创建持久会话（SSH 断开不影响运行）
tmux new-session -d -s p3_remote_10h

# 在 tmux 内运行
tmux send-keys -t p3_remote_10h 'cd /mnt/i/Github/Latent_Style/SchrodingerBridge' Enter
tmux send-keys -t p3_remote_10h 'bash p3_remote_run.sh r1_baseline_long exp/p3_remote_10h/r1_baseline_long/config.json' Enter

# 查看进度
tmux attach -t p3_remote_10h
# Ctrl+B D 脱离
```

### 5.3 一键全部运行（可选，串行执行）

```bash
#!/bin/bash
# run_all_experiments.sh — 串行运行所有 5 轮实验

EXPERIMENTS=(
  "r1_baseline_long:exp/p3_remote_10h/r1_baseline_long/config.json"
  "r2_film_large_init:exp/p3_remote_10h/r2_film_large_init/config.json"
  "r3_two_stage:exp/p3_remote_10h/r3_two_stage/config.json"
  "r4_aggressive_combo:exp/p3_remote_10h/r4_aggressive_combo/config.json"
)

for item in "${EXPERIMENTS[@]}"; do
    EXP_NAME="${item%%:*}"
    CONFIG="${item##*:}"
    
    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║  Launching: $EXP_NAME"
    echo "╚══════════════════════════════════════════╝"
    echo ""
    
    bash p3_remote_run.sh "$EXP_NAME" "$CONFIG"
    
    echo ""
    echo "--- Waiting 30s before next experiment ---"
    sleep 30
done

# R5 是特殊评估，不在循环中
echo "All training experiments completed. Run R5 (AdaIN eval) manually."
```

## 6. 动态决策流程

### 6.1 每轮结束后的检查清单

```
□ 1. 训练是否正常完成（无 OOM，无 NaN）？
□ 2. full_eval/ 下是否有每个 epoch 的 summary.json？
□ 3. 是否生成了 summary_grid.png？
□ 4. 目视检查图片：
     - 雾化程度（1-10 分，10=完全白雾）
     - 风格强度（是否能分辨目标风格？）
     - 内容保持（source image 结构是否保留？）
     - 异常 artifacts？
□ 5. 定量指标趋势：
     - clip_style 随 epoch 变化趋势
     - LPIPS 随 epoch 变化趋势
     - velocity_ratio 是否接近 1.0？
□ 6. 各 style 的表现差异（特别是 Minimalism）
```

### 6.2 R4 配置决策树

```
R1-R3 结果分析
│
├─ 哪个 clip_style 最高？
│  ├─ R1 (baseline) → R4 用候选配置 A（在 R1 基础上加量）
│  ├─ R2 (FiLM init) → R4 以 R2 为基础 + 对比损失
│  └─ R3 (two-stage) → R4 用候选配置 B（强化两阶段）
│
├─ LPIPS 是否都 < 0.40？
│  ├─ 是 → 保持或略微增加 style 权重
│  └─ 否 → 增加 content preservation 权重
│
└─ 是否存在"先好后坏"？
   ├─ 是 → 减少总 epoch 或启用 early stopping
   └─ 否 → 可以安全地训练更久
```

### 6.3 降级方案

如果某个实验失败：

| 失败原因 | 处理方式 |
|----------|---------|
| OOM | batch_size 降到 16 |
| 训练 NaN | 降低对应 loss 的 weight（减半） |
| 评估失败 | 降低 eval_batch_size 到 1 |
| 无 summary_grid | 手动运行 eval 脚本重新生成 |
| SSH 断连 | tmux 会话保持运行，重连后 `tmux attach` |

## 7. 输出产物

### 7.1 每轮实验必须产生的文件

```
exp/p3_remote_10h/<r_name>/
├── config.json              # 完整配置
├── focused.log              # 完整训练日志
├── epoch_*.pt               # 每个 epoch 的 checkpoint
├── full_eval/
│   ├── round2_convergence.json
│   ├── curve_summary.json
│   ├── clip_lpips_curve.csv
│   └── epoch_*/
│       ├── summary.json     # 定量指标
│       └── summary_grid.png # 可视化网格图（必须！）
```

### 7.2 最终汇总（R5 结束后生成）

```
docs/620/fog/p3_remote_10h_report.md
├── 所有实验的定量指标对比表
├── 各实验 summary_grid.png 的视觉对比
├── 最佳配置和对应的 checkpoint 路径
├── 关键发现和下一步建议
```

## 8. 关键文件索引

| 文件 | 用途 |
|------|------|
| [losses620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/losses620.py) | 所有 loss 实现（velocity_mag, contrastive, pixel_color 等） |
| [config_schema.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/config_schema.py) | 所有配置参数定义 |
| [model620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py) | AdaIN 实现 + FiLM endpoint head |
| [blocks620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/blocks620.py) | Style gate 模式 (fixed_one/film_only/tanh_gate) |
| [trainer.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/trainer.py) | 训练循环 + 两阶段权重调度 |
| [trivial_solution_unified.md](file:///g:/GitHub/Latent_Style/SchrodingerBridge/docs/620/fog/theory/trivial_solution_unified.md) | 理论基础文档 |
