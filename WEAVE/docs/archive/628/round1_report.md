# 628 破坏性消融 Round 1 报告 (64/128)

**时间**: 2026-06-29 02:54 (北京时间)
**Baseline**: T5 ep7 — ap_clip=0.7307, ap_lpips=0.3403

## 1. 进度概览
- 完成: 64/128 (50%)
- 待跑: 64 (含关键的 L13_no_flow, P13_wflow sweep)
- 当前训练: E17_w_anisotropic_kinetic
- GPU: 100% 满载, 7.3GB/12GB VRAM
- 预计剩余: ~6.1h (~09:00 完成)

## 2. 已完成实验分类汇总

### 2.1 显著偏离 baseline 的实验 (5 组)
| 实验 | clip | lpips | d_clip | d_lpips | 含义 |
|------|------|-------|--------|---------|------|
| D1_spectral_ode_off | 0.7136 | 0.3303 | -0.0171 | -0.0100 | 频域 ODE 是核心 |
| D2_adain_scale_0 | 0.7161 | 0.3270 | -0.0146 | -0.0133 | endpoint ADAIN 重要 |
| D4_avg_pool | 0.7303 | 0.3898 | -0.0004 | **+0.0495** | DWT 关键，avg_pool 灾难 |
| L7_no_spectral_ll | 0.7261 | 0.3232 | -0.0046 | -0.0171 | LL 频带是 clip 主要来源 |
| L9_no_spectral_lh_hl | 0.7317 | 0.3431 | +0.0010 | +0.0028 | LH/HL 反而是噪声 |
| L10_no_spectral_all | 0.7311 | 0.3400 | +0.0004 | -0.0003 | 全去反而略升 |

### 2.2 参数扫描有信号 (6 组)
| 实验 | clip | lpips | 含义 |
|------|------|-------|------|
| P1_adain_025 (0.25) | 0.7307 | 0.3527 | scale↓→lpips↑ |
| P1_adain_050 (0.5) | 0.7309 | 0.3498 | 中间档 |
| P1_adain_075 (0.75) | 0.7306 | 0.3437 | 接近 baseline |
| P2_alpha_005 (0.05) | 0.7297 | 0.3408 | alpha↓→clip↓ |
| P2_alpha_020 (0.2) | 0.7300 | 0.3483 | alpha↑→lpips↑ |
| P2_alpha_030 (0.3) | 0.7282 | 0.3620 | 极端 alpha 损失内容 |

### 2.3 保守吸引子稳定 (52 组, ≈0.7303/0.3411)
所有以下修改在 3 epoch 续训后对 clip/lpips 无显著影响：

**架构组件级 (D5-D9, D10-D18)**: 14 组
- skip_clean/skip_blur/decoder_highpass/residual_gain/no_residual
- style_gate_film_only/affine_gamma_0/affine_beta_0/global_gate_0
- tokenizer_residual_0/sharpen_0/endpoint_high_0/skip_residual_0/kinetic_off

**mode 切换 (D19-D30)**: 12 组
- attention mode: gated_raw/relu2/style_select/sparsemax
- endpoint mode: lowhigh
- transport mode: endpoint
- target_proj: dwt
- kinetic mode: per_band
- terminal_swd: hf
- bridge_path: tri_band
- swd_distance: squared
- t_sampling: logit_normal

**损失关闭 (L1-L6, L11-L12)**: 8 组
- endpoint_content/endpoint_style/terminal_swd/single_step_swd
- single_step_edge/kinetic/swd_high_freq/coupling_structure

**损失启用 (E10-E16)**: 7 组
- style_energy_floor/hsv_saturation/output_variance
- directional_cosine/freq_split_cosine/endpoint_velocity_reg/spectral_amplitude

**参数扫描无影响 (P4-P6)**: 11 组
- w_style × 3 档
- w_single_step_swd × 3 档
- gate_init × 3 档

## 3. 关键理论发现

### 3.1 "保守吸引子"的稳定性
**52/64 (81%) 的修改在 3 epoch 续训后无显著影响**。这强烈支持"保守吸引子"假设：
- T5 ep7 是一个**深度收敛点**，模型权重已到达局部最优
- 3 epoch 续训不足以让模型脱离该吸引子
- 唯一能改变结果的修改是**架构级破坏**（D1/D2/D4）和**频带级破坏**（L7）

### 3.2 DWT 频带贡献的非对称性
- LL 频带 (L7): clip 主要来源 (-0.0046) 但也增加 lpips (+0.0171 去除后 lpips 反而改善)
- HH 频带 (L8): 几乎无影响 (+0.0000 clip, +0.0018 lpips)
- LH/HL 频带 (L9): 移除反升 clip (+0.0010) — **是噪声项**
- 全部移除 (L10): clip 反升 (+0.0004) — **spectral losses 整体可能过约束**

### 3.3 ADAIN 的双刃剑效应
- D2_adain_scale_0: clip=0.7161 (-0.0146), lpips=0.3270 (-0.0133)
- P1_adain_025: clip=0.7307, lpips=0.3527 (+0.0124)
- P1_adain_075: clip=0.7306, lpips=0.3437
- 默认 scale=1.0: clip=0.7307, lpips=0.3403 (baseline)

**ADAIN 同时贡献 clip 提升和 lpips 降低**，是少数能"双赢"的机制。但 scale<1 时 lpips 反而变差，这表明 ADAIN 不是简单的"风格注入"，而是**内容锚定+风格注入耦合机制**。

### 3.4 alpha (style_extrap) 的纯权衡特性
- alpha=0: clip=0.7287 (-0.0020), lpips=0.3431
- alpha=0.05: clip=0.7297, lpips=0.3408
- alpha=0.1 (baseline): clip=0.7307, lpips=0.3403
- alpha=0.2: clip=0.7300, lpips=0.3483
- alpha=0.3: clip=0.7282, lpips=0.3620

**alpha=0.1 是 Pareto 最优点**。alpha<0.1 失去风格外推能力，alpha>0.1 损失内容保真度。

## 4. 7 命题初步验证

| # | 命题 | 当前证据 | 状态 |
|---|------|---------|------|
| 1 | Gate Collapse 必然性 | D10 (film_only) 无影响, gate_init (P6) 无影响 | ⚠️ 待 P18 大 gate |
| 2 | GN 白化定理 | D11/D12 (gamma/beta=0) 无影响 | ⚠️ 待 E1/E2 启用 |
| 3 | SWD 梯度正交性 | L3/L4/L11 关闭无影响 | ⚠️ 待 P9 + E24 |
| 4 | 训练-输出不匹配 | D24 (transport_endpoint) 无影响 | ❌ 推翻 (3ep 内无影响) |
| 5 | 有效 style 维度极低 | P6 (gate_init) 无影响 | ⚠️ 待 P16 num_tokens |
| 6 | 三难困境 | D1/D2/D4 证明架构核心 | ✅ 强支持 |
| 7 | FM 主导条件 | — | ⏳ 待 L13 + P13 |

## 5. 待跑实验的预期价值

### 高优先级 (期待显著信号)
- **L13_no_flow (w_flow=0)**: 验证命题 7。如果 clip 大幅下降，证明 FM 是主稳定器；如果不下降，证明 FM 主导是错觉。
- **P13_wflow sweep (0.1/0.3/0.5/2.0)**: 验证降低 FM 权重能否突破 clip 天花板。
- **P7 spectral_w_hh sweep (0.5/1.0/3.0/6.0)**: 验证高频权重饱和点。
- **P9 terminal_swd_weight sweep**: 验证 SWD 权重饱和点。

### 中优先级
- E1-E9 (剩余损失启用)
- P8 spectral_w_ll sweep
- P10 w_kinetic sweep
- P14 w_endpoint_content sweep

### 低优先级
- P15-P18 (耦合/OT/token/sharpen sweep)

## 6. 下一步规划

1. **等待剩余 64 组训练完成** (~6h)
2. **执行 10 组推理消融** (I9-I12)
3. **完整结果收集 + 7 命题验证**
4. **基于完整结果找新消融方向**：
   - 如果 L13_no_flow 显著降 clip → FM 主导确认，研究 FM 替代方案
   - 如果 L13_no_flow 无影响 → FM 主导推翻，研究真实主导损失
   - 如果 P13_wflow 扫描有信号 → 找到 w_flow 最优点
   - 如果 L9/L10 升 clip → 探索"反 spectral"路径
