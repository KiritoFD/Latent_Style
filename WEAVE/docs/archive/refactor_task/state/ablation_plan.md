# 调参消融实验设计

## 基线
- **Config**: t1_asg_5ep (adaptive_style_gate=True, 5 epochs, batch=24)
- **Metrics**: CLIP-S=0.7261, LPIPS=0.3354, DINO-S=0.4843, DINO-C=0.7692
- **有效组件**: Wavelet (DWT) + Flow Matching + Endpoint AdaIN + ASG

## 实验组设计

### Group A: 推理时参数扫描 (无需重训练, 快速)
基于 t1_asg_5ep checkpoint, 通过 config_override 修改推理参数:

| ID | 参数 | 值 | 理论 |
|----|------|-----|------|
| A1 | num_steps | 1 | 单步推理极限 |
| A2 | num_steps | 4 | 中等步数 |
| A3 | num_steps | 8 (baseline) | 基线 |
| A4 | num_steps | 12 | 多步精度 |
| A5 | style_extrap_alpha | 0.0 | 无外推 |
| A6 | style_extrap_alpha | 0.1 (baseline) | 基线 |
| A7 | style_extrap_alpha | 0.2 | 强外推 |
| A8 | style_extrap_alpha | 0.5 | 极端外推 |
| A9 | endpoint_adain_scale | 0.5 | 弱AdaIN |
| A10 | endpoint_adain_scale | 1.0 (baseline) | 基线 |
| A11 | endpoint_adain_scale | 1.5 | 强AdaIN |

### Group B: 训练时参数扫描 (需重训练, 慢)
基于 t1_asg_5ep 配置, 修改训练参数:

| ID | 参数 | 值 | 理论 |
|----|------|-----|------|
| B1 | w_endpoint_style | 4.0 | 弱风格权重 |
| B2 | w_endpoint_style | 8.0 (baseline) | 基线 |
| B3 | w_endpoint_style | 16.0 | 强风格权重 |
| B4 | spectral_w_ll | 0.1 | 弱LL权重 |
| B5 | spectral_w_ll | 0.3 (baseline) | 基线 |
| B6 | spectral_w_ll | 1.0 | 强LL权重 |
| B7 | style_cross_attn_gate_init | 0.01 | 弱gate |
| B8 | style_cross_attn_gate_init | 0.05 (baseline) | 基线 |
| B9 | style_cross_attn_gate_init | 0.2 | 强gate |
| B10 | learning_rate | 0.0001 | 慢学习 |
| B11 | learning_rate | 0.0002 (baseline) | 基线 |
| B12 | learning_rate | 0.0005 | 快学习 |

## 执行策略
1. 先执行 Group A (推理扫描, ~2 min/config, 共 ~22 min)
2. 再执行 Group B (训练扫描, ~3 min/config, 共 ~36 min)
3. 收集所有4指标数据
4. 生成散点图可视化
5. 撰写完整报告

## 成功标准
- 保持或超过 baseline 性能 (CLIP-S≥0.7261, DINO-S≥0.4843)
- 识别出最优参数组合
- 散点图清晰展示参数-指标关系
