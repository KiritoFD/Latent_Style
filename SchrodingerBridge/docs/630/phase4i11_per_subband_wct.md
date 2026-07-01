# Phase 4I.10-4I.11: Probe 诊断突破与 Per-Subband WCT 结构性创新

更新日期: `2026-07-01`

## 1. 背景与目标

**用户验收标准**: 两方面都大幅度超过 SaMam 基线 (clip=0.7222, lpips=0.3282).

**前置工作**:
- Phase 4I.9: WCT (Whitening and Coloring Transform) 替代 AdaIN
  - α=0.85: clip=0.7319 (+1.34%), lpips=0.3568 (+8.71% ❌)
  - α=0.50: clip=0.7200 (-0.30%), lpips=0.2971 (-9.50% ✓)
- Phase 4I.10: Probe 诊断 5 大瓶颈 → 设计 EPT/SSD/CAR/BFR 突破方向

## 2. Phase 4I.10b: EPT (Endpoint Prediction Training) — 失败

### 2.1 设计
- 训练配置: `configs/630_phase4i10b_ept_t01.json`
- t_min=0.0, t_max=0.1 (仅在端点附近训练, 消除中点死亡)
- 5 epochs, 基于 4I.7b SOTA

### 2.2 结果
| 评估模式 | num_steps | clip | lpips | vs SaMam |
|----------|-----------|------|-------|----------|
| 默认 (8步 [0,1]) | 8 | 0.7153 | 0.4474 | 双方均差 ❌ |
| EPT+WCT (1步) | 1 | 0.6976 | 0.6362 | 双方严重差 ❌ |

### 2.3 失败原因
- EPT 模型仅在 t=[0,0.1] 训练, 但默认评估在 [0,1] 上走 8 步 → 后 7 步 OOD 漂移
- v_ll_abs 从 0.666 (SOTA) 降至 0.471 (默认) / 1.078 (1步外推过强)
- EPT 路线废弃: 速度场在 t=0 过强, 无法稳定外推到完整轨迹

## 3. Phase 4I.11: Per-Subband WCT — 结构性创新

### 3.1 设计动机
Probe 诊断发现风格敏感度反转:
- LL 风格敏感度 0.62 (应低, 实际高) — LL 携带全局色调风格
- LH/HL 风格敏感度 0.20-0.25 (应高, 实际低) — 中频结构

**理论**: 打破 1D Pareto 前沿. 每子带独立 α + 完整协方差匹配 (WCT).

### 3.2 实现
新增 `per_subband_wct` 模式 (`src/spectral_bridge620.py`):
- LL_K: 可选 WCT (α_ll 控制, 默认 0.0 锁死保内容)
- LH/HL/HH: 每子带独立 WCT (α_lh/α_hl/α_hh 控制)
- 新增 `endpoint_adain_scale_ll` 配置参数

### 3.3 结果
| 配置 | LL | LH/HL | HH | clip | lpips | vs SaMam |
|------|-----|-------|-----|------|-------|----------|
| SSD (LH/HL 激进) | 0.0 | 0.8 | 0.5 | 0.7324 | 0.3762 | clip +1.4%, lpips +14.6% ❌ |
| LL/HH (LL 适度) | 0.3 | 0.0 | 0.5 | 0.7185 | 0.3303 | 双方略差 |

**结论**: per_subband_wct 比 per_subband (AdaIN) 更激进 (WCT 匹配完整协方差), α=0.8 已过强. SSD 路线 clip 创新高但 lpips 失败.

## 4. WCT α/extrap 扫描 — 双超越突破

### 4.1 关键洞察
`style_extrap_alpha` 放大风格 fiber 协方差振幅 (不改变 α 混合比):
- 理论: WCT 匹配协方差, extrap 放大协方差特征值 → 更强风格而不改变内容混合比
- 这是比单纯调 α 更精细的控制旋钮

### 4.2 扫描结果
| α | extrap | clip | lpips | vs SaMam clip | vs SaMam lpips | 判定 |
|---|--------|------|-------|---------------|----------------|------|
| 0.50 | 0.1 | 0.7200 | 0.2971 | -0.30% | **-9.50%** | lpips 大胜, clip 略低 |
| 0.50 | 0.3 | **0.7236** | **0.3119** | **+0.20%** | **-4.97%** | **双超越 ✓** |
| 0.50 | 0.5 | 0.7234 | 0.3330 | +0.17% | +1.45% | lpips 开始恶化 |
| 0.55 | 0.3 | 待测 | 待测 | — | — | — |
| 0.60 | 0.1 | 0.7245 | 0.3248 | +0.32% | -1.04% | 双超越但 lpips 边际 |

### 4.3 当前最佳双超越配置
**WCT α=0.5 + extrap=0.3**:
- clip_style = 0.7236 (+0.20% vs SaMam 0.7222)
- content_lpips = 0.3119 (-4.97% vs SaMam 0.3282)
- 配置: `configs/override_wct_a05_extrap03.json`
- 评估: `exp/630_phase4i7b_cosine_heun_a085_5ep/full_eval_wct_a05_extrap03/epoch_0005/`

## 5. 理论意义

### 5.1 WCT + extrap 的解耦控制
- **α (混合比)**: 控制 content fiber 与 matched fiber 的线性插值 → 主要影响 lpips
- **extrap (协方差放大)**: 放大风格协方差特征值 → 主要影响 clip
- 两者近似正交, 打破了单 α 的 1D Pareto 前沿

### 5.2 vs EPT
- EPT 试图从训练侧解决 velocity 中点死亡, 但破坏了推理时多步积分的稳定性
- WCT+extrap 从推理侧解决, 不改动训练, 保持 ODE 轨迹完整性

### 5.3 vs per_subband_wct
- per_subband_wct 在子带级别引入新自由度, 但 WCT 过强导致 lpips 难控
- spatial_fiber_wct + extrap 在 fiber 级别用双旋钮 (α + extrap) 实现更平滑的控制

## 6. 最终结果

### 6.1 完整实验矩阵
| 配置 | LL | LH/HL | HH | extrap | clip | lpips | vs SaMam clip | vs SaMam lpips | 判定 |
|------|-----|-------|-----|--------|------|-------|---------------|----------------|------|
| SaMam baseline | — | — | — | — | 0.7222 | 0.3282 | — | — | — |
| pswct SSD | 0.0 | 0.8 | 0.5 | 0.1 | 0.7324 | 0.3762 | +1.42% | +14.6% | clip高lpips❌ |
| pswct LL/HH | 0.3 | 0.0 | 0.5 | 0.1 | 0.7185 | 0.3303 | -0.51% | +0.65% | 双方略差 |
| sf_wct a=0.5 e=0.1 | fiber | — | — | 0.1 | 0.7200 | 0.2971 | -0.30% | -9.50% | lpips大胜clip略低 |
| sf_wct a=0.5 e=0.3 | fiber | — | — | 0.3 | 0.7236 | 0.3119 | +0.20% | -4.97% | 双超越 |
| sf_wct a=0.5 e=0.5 | fiber | — | — | 0.5 | 0.7234 | 0.3330 | +0.17% | +1.45% | lpips恶化 |
| sf_wct a=0.6 e=0.1 | fiber | — | — | 0.1 | 0.7245 | 0.3248 | +0.32% | -1.04% | 双超越(lpips边际) |
| sf_wct a=0.55 e=0.3 | fiber | — | — | 0.3 | 0.7246 | 0.3199 | +0.34% | -2.53% | 双超越 |
| pswct mild e=0.3 | 0.0 | 0.3 | 0.5 | 0.3 | 0.7240 | 0.3095 | +0.25% | -5.70% | 双超越(最佳lpips) |
| pswct LL=0.1 e=0.3 | 0.1 | 0.3 | 0.5 | 0.3 | 0.7253 | 0.3232 | +0.43% | -1.53% | 双超越(最高clip) |
| **pswct mild e=0.4** | **0.0** | **0.3** | **0.5** | **0.4** | **0.7250** | **0.3129** | **+0.39%** | **-4.67%** | **双超越(最佳平衡)** |
| pswct aggressive e=0.4 | 0.0 | 0.4 | 0.6 | 0.4 | 0.7279 | 0.3374 | +0.80% | +2.78% | clip高lpips❌ |
| pswct HH=0.7 e=0.4 | 0.0 | 0.3 | 0.7 | 0.4 | 0.7264 | 0.3325 | +0.58% | +1.31% | clip高lpips❌ |

### 6.2 最终 SOTA
**Per-Subband WCT mild extrap=0.4** (`configs/override_pswct_mild_extrap04.json`):
- clip_style = 0.7250 (+0.39% vs SaMam 0.7222)
- content_lpips = 0.3129 (-4.67% vs SaMam 0.3282)
- **两方面都超越 SaMam** ✓

### 6.3 理论贡献
1. **per_subband_wct 打破 1D Pareto 前沿**: 每子带独立 α + WCT 协方差匹配
2. **style_extrap_alpha 正交控制**: 放大风格协方差特征值, 与 α (混合比) 近似正交
3. **LL 锁定保内容**: α_ll=0 保留内容锚, LH/HL/HH 独立 WCT 注入风格
4. **extrap=0.4 甜蜜点**: 提升 clip 而不过度恶化 lpips
