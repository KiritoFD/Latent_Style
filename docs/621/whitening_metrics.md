# 621 白化/雾化定量指标体系

> 建立日期: 2026-06-21  
> 目标: 从图像空间和潜空间两个维度建立完备的白化诊断指标

---

## 1. 图像空间指标 (WFI - Whitening Fog Index)

### 1.1 核心指标

| 指标 | 公式 | 健康值(Seedream) | 白化信号 | 计算方式 |
|------|------|-----------------|----------|----------|
| contrast_ratio | σ(luminance) / μ(luminance) | ≈0.42 | <0.30 | RGB→灰度，计算std/mean |
| dynamic_range | (P95-P5)/(P95+P5) | ≈0.62 | <0.45 | 灰度百分位数 |
| saturation_mean | mean(HSV saturation) | ≈0.36 | <0.25 | RGB→HSV，取S通道均值 |
| edge_energy | mean(√(dx²+dy²)) | — | <0.01 | 灰度梯度能量 |
| luminance_std | σ(luminance) | — | <0.05 | 灰度标准差 |
| wfi_score | 1-(0.4*cr_norm+0.3*sr_norm+0.3*dr_norm) | ≈0.16 | >0.35 | 归一化复合指标 |

### 1.2 WFI计算细节

```python
# 归一化
cr_norm = min(contrast_ratio / 0.5, 1.0)
sr_norm = min(saturation_mean / 0.4, 1.0)
dr_norm = min(dynamic_range / 0.6, 1.0)

# 复合WFI
wfi_score = 1.0 - (0.4 * cr_norm + 0.3 * sr_norm + 0.3 * dr_norm)
```

### 1.3 对比指标 (Source vs Generated)

| 指标 | 公式 | 健康值 | 白化信号 |
|------|------|--------|----------|
| contrast_retention | gen_cr / src_cr | >0.8 | <0.5 |
| dr_retention | gen_dr / src_dr | >0.8 | <0.5 |
| sat_retention | gen_sr / src_sr | >0.8 | <0.5 |
| wfi_delta | gen_wfi - src_wfi | <0.1 | >0.2 |

---

## 2. 潜空间指标

### 2.1 Endpoint Probe 指标

| 指标 | 公式 | 健康值 | 白化信号 | 测量位置 |
|------|------|--------|----------|----------|
| endpoint_alpha | ‖ẑ₁-x‖₂ / ‖y-x‖₂ | >0.5 | <0.3 | predict_endpoint(t=0) |
| high_alpha | ‖ẑ₁_high-(x-x_lp)‖₂ / ‖y_high-x_high‖₂ | >0.3 | <0.0 | 高频endpoint |
| low_alpha | 同上但低频 | >0.3 | <0.1 | 低频endpoint |
| style_sensitivity | std(v(s₁),v(s₂),...) | >5.0 | <1.0 | 多style velocity差异 |

### 2.2 统计探针指标

| 指标 | 公式 | 健康值 | 白化信号 | 测量位置 |
|------|------|--------|----------|----------|
| cos_sim_to_mean | cos(v, mean(v_all)) | <0.6 | >0.75 | metrics.csv clip vectors |
| velocity_abs | mean(|v|) | >0.2 | <0.1 | forward输出 |
| endpoint_pred_abs | mean(|ẑ₁|) | >0.3 | <0.1 | endpoint输出 |
| endpoint_low_abs | mean(|ẑ₁_low|) | — | <0.05 | 低频endpoint |
| endpoint_high_abs | mean(|ẑ₁_high|) | — | <0.01 | 高频endpoint |

### 2.3 Block级统计探针

| 指标 | 公式 | 位置 | 白化信号 |
|------|------|------|----------|
| block_i_output_mean | μ(h_i) | 每个block输出 | 跨block趋同 |
| block_i_output_std | σ(h_i) | 每个block输出 | 被GN压缩 |
| cross_attn_entropy | -Σ attn·log(attn) | cross-attn后 | ≈ln(N)均匀 |
| cross_attn_delta_abs | mean(|style_delta|) | cross-attn后 | <0.01 style无效 |
| film_gamma_abs | mean(|γ|) | FiLM后 | <0.01 FiLM无效 |
| film_beta_abs | mean(|β|) | FiLM后 | <0.01 FiLM无效 |
| style_gate_value | tanh(gate) | 每个block | <0.1 style被压制 |

### 2.4 Style信号保留率

$$R_\text{style} = \frac{\| \text{GN}(x''(s_1)) - \text{GN}(x''(s_2)) \|_2}{\| x''(s_1) - x''(s_2) \|_2}$$

- 健康值: >0.5
- 白化信号: <0.2 (style被GN洗掉)

---

## 3. 探针运行方案

### 3.1 本地Probe (RTX 4070)

```bash
# 基线WFI
python tools/probe_620_fog_whiteness_index.py \
  --eval_dir exp/620_spatial_bridge/<run>/full_eval_wfi/epoch_0001/ \
  --output probe_results/baseline_wfi.json

# Hypothesis probe (endpoint alpha)
python tools/probe_620_hypothesis_metrics.py \
  --checkpoint exp/620_spatial_bridge/<run>/epoch_0001.pt \
  --config exp/620_spatial_bridge/<run>/config.json \
  --output probe_results/hypothesis_alpha.json

# Gradient probe (SWD gradient direction)
python tools/probe_620_solver_trace.py \
  --checkpoint exp/620_spatial_bridge/<run>/epoch_0001.pt \
  --config exp/620_spatial_bridge/<run>/config.json \
  --output probe_results/gradient_trace.json
```

### 3.2 远程Probe (3060 WSL)

```bash
# 同步checkpoint到远程
scp -P 2222 <local_checkpoint> administrator@100.115.18.62:/mnt/g/GitHub/Latent_Style/SchrodingerBridge/exp/

# 远程运行
ssh -p 2222 administrator@100.115.18.62 \
  "wsl -d Ubuntu-26.04 -- bash -c 'cd /mnt/g/GitHub/Latent_Style/SchrodingerBridge && python tools/probe_620_fog_whiteness_index.py ...'"
```

---

## 4. 指标阈值决策矩阵

| 条件 | 判定 | 行动 |
|------|------|------|
| WFI < 0.20 且 clip_style ≥ 0.70 | 健康 | 继续优化 |
| WFI 0.20-0.40 且 clip_style ≥ 0.70 | 可接受 | 需进一步压低WFI |
| WFI > 0.40 | 白化严重 | 必须修复 |
| endpoint_alpha < 0.3 | shrinkage | 修复endpoint head |
| cross_attn_entropy > 5.0 | attention collapse | 修复style注入 |
| R_style < 0.2 | norm collapse | 修复归一化 |
| velocity_abs < 0.1 | velocity collapse | 修复loss/初始化 |
