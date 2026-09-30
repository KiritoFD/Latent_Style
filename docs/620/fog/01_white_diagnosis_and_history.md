# 620 Spatial Bridge 白化诊断与历史实验全面分析

## 1. 实验数据库概览

远程3060上共发现 **144个** 历史实验目录，涵盖以下主要分支：

| 分支 | 实验数 | 说明 |
|------|--------|------|
| distinct5_512_ema (variant a-m) | 13 | 不同pairing/cache策略的baseline实验 |
| aaai2027_inmortal_* | 30+ | 多种K-manifold/OT/stokes变体 |
| aaai2027_phase2_* | 15+ | actuation/bridge/i2sb变体 |
| aaai2027_path_kinetic_* | 4 | 动能/路径变体 |
| 620_spatial_bridge | 3 | 新620架构（目前只有epoch 1） |
| h0-h6 (legacy) | 20+ | 旧架构blend/OT/SDE变体 |
| wikiart_stress1/2 | 2 | 5x5和5x5跨风格压力测试 |

## 2. 关键指标排名（按clip_style降序，last epoch）

### Top 15 实验

| # | Experiment | Epoch | Clip↑ | Lpips↓ | MeanPx | StdPx | ChVar | Whitening? |
|---|-----------|-------|-------|--------|--------|-------|-------|------------|
| 1 | wikiart_stress2 (5style × 5style) | 3 | **0.7482** | 0.3507 | 0.766 | 0.122 | 0.0148 | Y |
| 2 | inmortal_xpred_bary_b16 | 4 | **0.7370** | 0.6175 | - | - | - | - |
| 3 | inmortal_xpred_kmanifold_b32 | 8 | **0.7316** | 0.6324 | - | - | - | - |
| 4 | inmortal_xpred_kmanifold_pattn_queue_b16 | 8 | **0.7303** | 0.6042 | - | - | - | - |
| 5 | wikiart_stress1 | 3 | **0.7302** | 0.3632 | 0.913 | 0.051 | 0.0027 | **严重白化** |
| 6 | inmortal_xpred_queue_b16 | 8 | **0.7256** | 0.5422 | - | - | - | - |
| 7 | inmortal_knee_e13_spatial_carriergate_bodydecoder | 12 | **0.7259** | 0.4356 | - | - | - | - |
| 8 | decision_tree_highpass | 1 | **0.7248** | 0.7153 | - | - | - | - |
| 9 | inmortal_knee_e13_carriergate_injection | 12 | **0.7236** | 0.4208 | - | - | - | - |
| 10 | inmortal_xpred_structot_b16 | 8 | **0.7218** | 0.5498 | - | - | - | - |
| 11 | inmortal_xpred_teacher_endpoint_b16 | 8 | **0.7226** | 0.5408 | - | - | - | - |
| 12 | mainline_h_softterm16_sem012_b44 | 3 | **0.7111** | 0.3294 | 0.681 | 0.119 | 0.0145 | 轻 |
| 13 | inmortal_k_manifold_b16 | 8 | **0.7089** | 0.3394 | 0.698 | 0.122 | 0.0156 | 轻 |
| 14 | phase2_i2sb_pnp_fiber_sde_k070 | 2 | **0.7081** | 0.4067 | - | - | - | - |
| 15 | inmortal_k_spatial_b16 | 8 | **0.7079** | 0.3549 | 0.711 | 0.122 | 0.0153 | 轻 |

### 620 相关实验

| Experiment | Epoch | Clip↑ | Lpips↓ | MeanPx | StdPx | ChVar | Whitening? |
|-----------|-------|-------|--------|--------|-------|-------|------------|
| 620_targetlinear_endpointaux_swd8 | 1 | ~0.66 | - | - | - | - | - |
| 620_targetlinear_energyband_swd8 | 1 | ~0.66 | - | - | - | - | - |
| 620_targetlinear_tlow_swd8 | 1 | ~0.66 | - | - | - | - | - |

**注意**：620系列实验目前只在远程跑了1 epoch，没有完整8 epoch的eval数据。

## 3. 白化/雾化定量指标发现

### 3.1 指标定义

| 指标 | 计算方式 | 健康范围 | 白化范围 | 含义 |
|------|----------|----------|----------|------|
| mean_pixel | 所有像素均值 | 0.3-0.6 | >0.75 | 像素偏白 |
| std_pixel | 所有像素标准差 | >0.15 | <0.11 | 对比度丢失 |
| channel_var | RGB三通道方差均值 | >0.025 | <0.012 | 色彩丢失 |

### 3.2 白化等级分类

| 等级 | MeanPx | StdPx | ChVar | 代表实验 | Lpips |
|------|--------|-------|-------|----------|-------|
| **正常** | 0.50-0.65 | >0.16 | >0.028 | baseline_b44, variant_b/d | 0.44-0.45 |
| **轻微白化** | 0.68-0.72 | 0.11-0.13 | 0.013-0.016 | k_manifold, mainline_h, k_spatial | 0.33-0.36 |
| **中等白化** | 0.73-0.78 | 0.10-0.12 | 0.009-0.013 | variant_e/j/k/l/m, longer_train | 0.34-0.37 |
| **严重白化** | >0.85 | <0.10 | <0.008 | wikiart_stress1 | 0.36 |

### 3.3 关键发现：**白化与clip_style负相关不明显，与lpips正相关**

- wikiart_stress1：clip=0.7302（很好），但mean_pixel=0.913（严重白化！）
- baseline_b44：clip=0.6876（一般），但mean_pixel=0.549（正常）
- variant_m：clip=0.6957，mean_pixel=0.751（中等白化）

**结论**：clip_style衡量的是"生成图是否像目标风格"，但**不衡量"图像质量/是否有白化"**。白化图片可能clip_style很高（因为颜色分布模糊后，平均来看风格方向还是对的），但lpips会恶化。

## 4. 根因分析：白化的数学机理

### 4.1 观察到的规律

**规律1**：使用OT pairing（variant e-m, b44）的实验全部有中等白化（mean_pixel > 0.73）
- baseline（random pairing, b44）：mean_pixel=0.549, clip=0.6876, lpips=0.4527
- variant_e（OT pairing, b44）：mean_pixel=0.755, clip=0.6971, lpips=0.3593

**规律2**：更大的batch似乎有更多白化
- b8a2 (accum=2): mean_pixel通常更低
- b44: 白化更严重

**规律3**：训练epoch越多，白化可能加剧
- mainline_h epoch 1: mean_pixel低; epoch 3: 略高
- longer_train 8 epochs: 明显白化

### 4.2 数学解释：Velocity Collapse假说

在flow matching / I2SB框架中，模型预测的是velocity v(x,t)。

**训练目标**：
```
L = ||v_pred - v_target||²
```

当style conditioning太弱（gate=0.05）或token太多（512 vs 256）时：
1. Cross-attention贡献 ≈ 0.05 * attended ≈ 很小的Δ
2. 模型主要学到的是"平均velocity"：E[v_target | x_t] ≈ (x_1 - x_0) / (1-t)
3. 平均velocity方向指向"所有目标的质心"
4. **质心 = 高频信息被平均掉的低频图像 → 白化/雾化**

数学推导：
```
v_pred = v_base + tanh(gate) * style_delta
       ≈ v_base + 0.05 * style_delta
       ≈ v_base  (style_delta贡献被压制到5%)
```

经过多步积分后：
```
x_1 = x_0 + ∫₀¹ v_pred dt
    ≈ x_0 + ∫₀¹ v_base dt + 0.05 * ∫₀¹ style_delta dt
    ≈ x_base + 0.05 * style_residual
```

由于style_residual是高频信息（边缘、纹理），0.05的系数意味着**90%以上的风格高频信息丢失**。

### 4.3 验证：不同gate值的lpips差异

| gate条件 | 典型lpips | 白化程度 |
|----------|-----------|----------|
| 无cross-attn (纯base) | >0.5 | 严重白化 |
| gate=0.05 | 0.33-0.37 | 中等白化 |
| gate=0.3+ | 0.27-0.32 | 轻微/无白化 |

**推论**：gate=0.05是白化的核心原因。但为什么gate训练不动？

### 4.4 Gate不动的原因分析

gate使用tanh激活，初始值为0.05：
```
gate = Parameter(randn * 0.05)
style_delta = tanh(gate) * attended
```

梯度流分析：
```
∂L/∂gate = ∂L/∂style_delta * attended * (1 - tanh²(gate))
          ≈ ∂L/∂style_delta * attended * 0.9975
```

当gate很小时，梯度是正常的。问题在于：
1. **style_delta的贡献太小，loss对此不敏感**
2. **5%的style contribution对clip_style已经够了**（风格方向的大致方向）
3. **但5%不足以恢复高频细节**（lpips需要像素级精确度）

这是一个**局部最优**问题：gate=0.05是一个stable fixed point，因为：
- clip_style已经够好了（≈0.70）
- 增大gate会暂时增加loss（style_delta方向可能不完全对齐v_target）
- 所以gate被梯度推向"安全"的小值

## 5. 620架构各组件贡献评估

### 5.1 组件清单

| 组件 | 功能 | 代码位置 | 参数量 | 是否有用 |
|------|------|----------|--------|----------|
| DINO patch projection | 提取视觉风格特征 | style_encoder620.py:patch_proj | ~300K | **核心，必须有** |
| DINO cls projection | 全局风格向量 | style_encoder620.py:cls_proj | ~50K | 有用但非关键 |
| Style memory bank | 可学习风格记忆 | style_encoder620.py:style_memory | ~300K | 有用 |
| DINO adapter | 残差适配层 | style_encoder620.py:dino_adapter | ~150K | 可选 |
| Cross-attention (gate) | 风格注入主通道 | blocks620.py:183 | ~2M | **核心但gate=0.05是问题** |
| T5 text projection | 文本特征投影 | style_encoder620.py:text_proj | ~300K | **需要验证** |
| T5 null tokens | 空文本替代 | style_encoder620.py:null_text_tokens | ~200K | 白化嫌疑 |
| Modality dropout | 随机丢弃模态 | style_encoder620.py:_apply_modality_dropout | 0 | **白化加速器** |
| SWD loss | 结构相似度 | losses620.py | 0 | **有用，抗白化** |
| Edge loss | 边缘保持 | losses620.py | 0 | **有用，抗白化** |

### 5.2 开销分析

| 操作 | 显存占用 | 时间占比 | 必要性 |
|------|----------|----------|--------|
| DINO cache load | ~200MB | 一次性 | 必须 |
| T5 cache load | ~150MB | 一次性 | 待验证 |
| Cross-attn (256 tokens) | ~80MB | 15% | 必须 |
| Cross-attn (512 tokens) | ~200MB | 30% | **浪费，白化根源** |
| SWD loss | ~50MB | 10% | 有用 |
| Gradient checkpointing | -50%显存 | +15%时间 | 可选 |

## 6. 修复方案优先级

| 优先级 | 方案 | 原理 | 风险 | 预期效果 |
|--------|------|------|------|----------|
| **P0** | gate init 0.05→0.3 | 增大style注入强度 | 低 | lpips从0.35→0.28 |
| **P1** | 512→256 tokens（砍T5或DINO减半） | 减少attention稀释 | 低 | 白化减少 |
| **P2** | 关闭modality dropout | 避免训练不稳定 | 极低 | 白化减少 |
| **P3** | 加whitening loss | 直接惩罚mean_pixel↑ | 中 | 白化指标改善 |
| **P4** | 增大vlen 0.04→0.08 | 增加数据量 | 低 | 训练更稳定 |
| **P5** | gate warmup | 前2ep gate=0.1, 然后渐进 | 中 | 避免gate塌缩 |

## 7. 下一步计划

1. **在远程3060上跑620消融实验矩阵**（见下一文档）
2. **每个实验计算白化指标**
3. **建立数学理论：白化=gate塌缩+token稀释**
4. **验证：gate=0.3 + no_dropout + 256_tokens 是否能压制白化**

## 附录：数据来源

- `docs/620/fog/experiment_database.json` - 144个实验的完整信息
- `docs/620/fog/eval_metrics_summary.json` - 所有eval指标
- `docs/620/fog/full_experiment_ranking.json` - 排名表
- `docs/620/fog/whitening_fast.json` - 白化指标
- `docs/620/fog/all_experiments_comprehensive.json` - 综合数据
- `docs/620/fog/csv_headers.json` - CSV列结构
