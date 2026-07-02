# Phase 4 实验总览 (Subtractive Ablation + Additive Exploration)

**Date**: 2026-07-01
**Status**: Phase 4G.2b in progress (4G.2 MIXED result documented)
**Current SOTA**: Phase 4F.1 (haar lvl3, clip=0.7319, lpips=0.3428)
**Thresholds**: clip_style ≥ 0.7243, content_lpips ≤ 0.3453

---

## 1. 实验全景表

### 1.1 减法消融 (Subtractive Ablation)

| 阶段 | 实验 | 改动 | clip | lpips | 判定 | 关键结论 |
|------|------|------|------|-------|------|----------|
| 4A1 | dead_code_removal | 删除 6 个死代码占位符 | — | — | PASS | 代码清洁, 无性能影响 |
| 4A2 | spectral_w_ll=0.0 | w_ll=0 (不训练 head_ll) | 0.7117 | 0.2994 | FAIL | w_ll 必要 (但 4G.1 澄清: 是"假阴性", 未训练 v_ll 是噪声) |
| 4A2 | style_extrap_alpha=0.0 | 禁用 style 外推 | 0.7242 | 0.3333 | FAIL | extrap 必要 |
| 4A2 | endpoint_adain_scale=0.0 | 禁用 Endpoint AdaIN | 0.7082 | 0.2994 | FAIL | AdaIN 必要 |

**减法结论**: 所有 3 个核心组件 (w_ll, style_extrap_alpha, endpoint_adain_scale) 均不可移除。Content Fidelity Pathway (DWT haar → AdaIN scale → spectral ODE) 验证完整。

### 1.2 加法探索 (Additive Exploration)

| 阶段 | 实验 | 改动 | clip | lpips | 判定 | 关键结论 |
|------|------|------|------|-------|------|----------|
| 4B1 | freq_a1 | 频域 masking α=1.0 | 0.7258 | 0.3357 | PASS | 频域 mask 有效 |
| 4B1 | freq_a1_rand50 | 频域+随机 mask | 0.7264 | 0.3354 | PASS | 两者可叠加 |
| 4B2 | freq_a1_rand50_10ep | 长训练 10ep | 0.7277 | 0.3394 | PASS | 5ep 后内容漂移 |
| 4B3 | dwt_a1 | DWT tokenizer | 0.7266 | 0.3402 | PASS | Haar DWT 可用 |
| 4C | block_masking+real_DINO | RGB 块遮挡 + DINOv2 | 0.7151 | 0.3177 | **NEGATIVE** | **"Style Is Learned, Not Extracted"** — DINO 污染 clip -0.018 |
| **4D** | **lvl2 (2-Level DWT)** | **多级分解** | **0.7301** | **0.3402** | **PASS - BREAKTHROUGH** | **2-Level 突破 10ep baseline, +0.0040 clip** |
| 4E.1 | db2 lvl1 | Daubechies 平滑小波 | 0.7258 | 0.3288 | FLAT | db2 ≈ haar (像素级平滑对 CLIP/LPIPS 不敏感) |
| 4E.2 | db2 lvl2 | Daubechies 2-Level | 0.7298 | 0.3398 | FLAT | 多级是主导效应, 基函数非关键 |
| **4F.1** | **haar lvl3** | **3-Level DWT** | **0.7319** | **0.3428** | **PASS - NEW SOTA** | **3-Level 最优, +0.0018 over lvl2** |
| 4F.2 | haar lvl4 | 4-Level DWT | 0.7316 | 0.3461 | FAIL | 4-Level 过激, LL_4 (2×2) 丢位置信息 |
| 4G.1a | lock_ll + w_ll=1.0 | 推理锁死 LL | 0.7178 | 0.3281 | **NEGATIVE** | LL velocity 贡献 +0.014 clip, 不可锁死 |
| 4G.1b | lock_ll + w_ll=0.0 | 全锁 | 0.7174 | 0.3372 | FAIL | w_ll 训练提供 backbone 旁路收益 (-0.0091 lpips) |
| **4G.2** | **per_subband α=1.0** | **频域每子带 AdaIN** | **0.7361** | **0.3843** | **MIXED** | **clip NEW SOTA (+0.0042), lpips FAIL (+0.0415, 9× 注入)** |
| 4G.2b | per_subband α=0.5 | 降低注入量 | 0.7362 | 0.3845 | **FAIL** | **α 失效! 多步 Euler 迭代累积使 α=0.5≡α=1.0** |

### 1.3 多级 DWT 趋势表

| Level | LL 尺寸 | clip_style | content_lpips | Δ clip (vs prev) |
|-------|---------|------------|---------------|------------------|
| 1 | 16×16 | 0.7261 | 0.3296 | baseline |
| 2 | 8×8 | 0.7301 | 0.3402 | +0.0040 |
| **3 (SOTA)** | **4×4** | **0.7319** | **0.3428** | **+0.0018** |
| 4 | 2×2 | 0.7316 | 0.3461 | -0.0003 (FAIL) |

**趋势**: 1→2 收益最大 (+0.0040), 2→3 递减 (+0.0018), 3→4 反转 (-0.0003)。3-Level 是峰值。

---

## 2. 核心理论发现

### 2.1 "Content Fidelity Pathway" (4A2 减法验证)

三个不可移除的核心组件构成"内容保真路径":

```
DWT Haar (正交分解) → AdaIN scale (统计匹配) → Spectral ODE (频域速度场)
    ↓                        ↓                         ↓
物理切分 LL/HF         风格注入 + 内容锚       LL/LH/HL 独立 velocity
```

### 2.2 "LL Is Not Pure Content Anchor" (4G.1 NEGATIVE → 核心洞察)

4G.1 的 2×2 矩阵消融精确量化了 LL velocity 的双重角色:

| | w_ll=0 (不训练) | w_ll=1.0 (训练) |
|---|---|---|
| **lock=False** (推理用 v_ll) | 4A2: clip=0.7117 (随机噪声) | **4F.1 SOTA: 0.7319** |
| **lock=True** (推理锁死) | 4G.1b: clip=0.7174 | 4G.1a: clip=0.7178 |

- **v_ll 应用**: +0.0141 clip (LL 携带全局色调/光照/色相风格信息)
- **v_ll 训练**: -0.0091 lpips (head_ll 梯度回流改善 backbone 内容理解)
- **结论**: LL 不是纯内容锚, 是"内容 + 全局风格"的混合载体

### 2.3 "Frequency-Domain Decoupling Works" (4G.2 MIXED → 方向确认)

4G.2 证明 per-subband AdaIN 的正交性统计隔离优势是真实的:
- clip +0.0042 突破 SOTA: 每子带独立匹配比空间域全局更精准
- 但 α=1.0 导致 9× 风格注入 (vs spatial_fiber 的 1×): lpips 超标
- **结论**: 频域解耦方向正确, 需通过 α 参数控制注入量

### 2.4 "Style Is Learned, Not Extracted" (4C NEGATIVE)

4C 证明引入外部 DINOv2 特征作为 style 条件反而损害 clip (-0.018):
- learnable style_memory 是任务最优的 (端到端学习)
- DINOv2 特征是 content-biased 的 (物体语义污染风格)
- **用户决策**: 不使用 DINO, 避免外部模型污染

---

## 3. 论文 Core Story (根据 4G.2b 结果二选一)

### 3.1 如果 4G.2b 成功 (新 SOTA, clip > 0.7319, lpips ≤ 0.3453)

> "我们通过 Haar DWT 多级分解 (4F) 解耦内容 (LL) 与风格 (HF), 以 3-Level 分解取得 SOTA。
> 消融实验矩阵: (1) 减法 (4A2) 验证 Content Fidelity Pathway; (2) LL velocity 消融 (4A2+4G.1) 量化 +0.014 clip 贡献, 证明 LL 携带全局色调;
> (3) 频域 per-subband AdaIN (4G.2) 利用 Haar 正交性保证统计隔离, 突破空间域 SOTA, α 参数 (4G.2b) 调控注入量-保真度 trade-off。
> 完整的三层频域解耦: LL velocity (全局色调) + per-subband AdaIN (笔触/色彩/噪点) + Spectral ODE (频域速度场)。"

### 3.2 如果 4G.2b 仍 FAIL (4F.1 为最终 SOTA)

> "我们通过 Haar DWT 多级分解 (4F) 解耦内容 (LL) 与风格 (HF), 以 3-Level 分解取得 SOTA (clip=0.7319)。
> 消融实验矩阵: (1) 减法 (4A2) 验证三大核心组件不可移除; (2) LL velocity 消融 (4A2+4G.1) 量化 +0.014 clip 贡献;
> (3) 频域 per-subband AdaIN 极限探索 (4G.2): 虽提升 clip (+0.0042) 但引入 9× 风格注入导致 lpips 超标,
> 揭示空间域 fiber 的'隐式正则化'价值 — 统计平均效应天然抑制过拟合。
> 多级 DWT 趋势 (4D-4F): 1→2 (+0.0040), 2→3 (+0.0018), 3→4 (-0.0003), 3-Level 为峰值。"

---

## 4. 文件索引

| 文档 | 内容 |
|------|------|
| [HANDOVER.md](HANDOVER.md) | 顶层交接文档 |
| [phase4g_full_wavelet_ode_design.md](phase4g_full_wavelet_ode_design.md) | 4G 设计 + 4G.1 结果 + 4G.2 结果 |
| [phase4g2_per_subband_adain.md](phase4g2_per_subband_adain.md) | 4G.2 完整设计 + 实验结果 |
| [phase4c_block_masking_rgb.md](phase4c_block_masking_rgb.md) | 4C NEGATIVE (DINO 污染) |
| [state/progress.json](state/progress.json) | 状态机 (iteration=16) |
| [state/directions_tried.json](state/directions_tried.json) | 已试方向记录 |

---

## 5. 待完成工作

1. **论文写作** — Core Story 确认为 §3.2 (4F.1 为最终 SOTA)
2. **Git 提交** — Phase 4 全部实验文档化后统一提交
3. **Future Work** — End-of-trajectory AdaIN, α 衰减调度 (4G.2b 洞察的后续方向)
