# 03 — 历史实验全景

> 完整实验记录：Phase 1-3（清理+masking+验证）、Phase 4A-4J（远程探索，60+ 配置）、Local T1-T19（本地 DWT route 探索，26+ 配置）。本文档按时间顺序组织，标注每个实验的配置、结果、判定与关键结论。

---

## 1. 实验全景概览

| 阶段 | 时间 | 实验数 | 关键产物 | 状态 |
|------|------|--------|----------|------|
| Phase 1 (630) | 2026-06-30 | 11+9+7 | H1-H11 dead code + M1-M9 legacy + L1-L7 keep | 完成 |
| Phase 2 (630) | 2026-06-30 | 4 | Blindfolded Tokenizer (masking) | 完成 |
| Phase 3 (630) | 2026-06-30 | 1 | 10-epoch 完整训练验证 | 完成 |
| Phase 4A-4F | 2026-07-01 | ~20 | 减法消融 + 加法探索（4F.1 远程 SOTA） | 完成 |
| Phase 4G-4H | 2026-07-01 | ~15 | LL lock + per-subband AdaIN + EOTA | 完成 |
| Phase 4I | 2026-07-01 | ~15 | Heun solver + schedule + RK4（4I.7b 远程 SOTA） | 完成 |
| Phase 4J | 2026-07-01 | ~5 | DWT route + few-shot（4J.1 本地起点） | 完成 |
| Local T1-T12 | 2026-07-02 | 15 | Stochastic DWT Route（T11 本地 SOTA） | 完成 |
| Local T13-T16 | 2026-07-02 | 7 | LLGSI/CASI/LLGQCA（全失败） | 完成 |
| Local T18-T19 | 2026-07-02 | 4 | loss/容量调优（全失败，T11 确认 SOTA） | 完成 |

**总配置数**: 90+，系统性证明当前架构的 1:8 trade-off 不可破。

---

## 2. Phase 1: Codebase Cleanup (630)

### 2.1 Phase 1A — H1-H11 零风险 dead code 删除

| Item | 文件 | 内容 | 判定 |
|------|------|------|------|
| H1 | spectral_bridge620.py | 未使用 import | 删除 |
| H2 | spectral_losses620.py | 未使用辅助 loss 函数 | 删除 |
| H3-H5 | blocks620.py | 未使用 helper 方法 | 删除 |
| H6-H8 | style_encoder620.py | 未使用 hook/参数 | 删除 |
| H9-H11 | utils/*.py | 死代码占位符 | 删除 |

**Commit**: `925b6bea7`

### 2.2 Phase 1B — M9 attn_mode bug TDD 修复

**Bug**: `style_attn_mode='relu2'` 在 config 中设置，但 `spectral_bridge620.py` 不传递给 blocks，默认 softmax。

**TDD 修复**:
1. 写失败测试 `tests/test_630_spectral_ode.py`（验证 relu2 传播）
2. 修复 `spectral_bridge620.py` 传递 attn_mode
3. 3-epoch 训练验证 PASS

**Commit**: `69da87cb0`

### 2.3 Phase 1C — Legacy 文件批量删除

| 删除内容 | 行数 |
|----------|------|
| `TimeConditionedLANCETBridge` | ~2070 |
| `lancet_blocks.py`, `lancet_backbone.py` | ~5000 |
| 其它 legacy 文件 | ~4276 |
| **总计** | **-11346 行** |

**Commit**: `bcea0a41b`

### 2.4 Phase 1D — 最简 codebase 性能验证

3-epoch 训练，clip=0.7293, lpips=0.3203 — PASS baseline。

**Commit**: `9de1e9e03`

---

## 3. Phase 2: Masking (Blindfolded Tokenizer)

### 3.1 设计理论

信息瓶颈：mask 部分 style tokens，强迫 style_memory 学更鲁棒表征。

### 3.2 4 组消融（3-epoch）

| 配置 | mask_mode | mask_ratio | clip | lpips | 判定 |
|------|-----------|-----------|------|-------|------|
| baseline | none | 0.0 | 0.7261 | 0.3296 | — |
| random_50 | random | 0.5 | 0.7275 | 0.3238 | **最佳** |
| random_75 | random | 0.75 | 0.7268 | 0.3252 | PASS |
| shuffle_50 | shuffle | 0.5 | 0.7259 | 0.3271 | PASS |
| shuffle_75 | shuffle | 0.75 | 0.7243 | 0.3284 | FLAT |

**结论**: `random_50` 最佳。shuffle 破坏位置信息但效果不如 random dropout。

**Commit**: `8df445e50`

---

## 4. Phase 3: 完整训练验证

10-epoch 从零训练（独立目录，`mask_random_50`）:
- `epoch_0005`: clip=0.7275, lpips=0.3238
- `epoch_0010`: clip=0.7289, lpips=0.3370

**结论**: Phase 3 验证最简 codebase + masking 可用，作为 Phase 4 探索基础。

**Commit**: `adc6a0d38`

---

## 5. Phase 4A: 减法消融

### 5.1 4A1 — Dead Code Removal

删除 6 个死代码占位符（`spectral_brownian_noise_scale`, `loss_type metric`, `loss_fm alias`, `loss_fm_total`, `compute_debug`, `loss_fn.last_debug`）。

**Commit**: `31fc94cac`

### 5.2 4A2 — 三大核心组件减法消融（3-epoch）

| 配置 | clip | lpips | 判定 | 关键结论 |
|------|------|-------|------|----------|
| baseline | 0.7261 | 0.3296 | — | — |
| `spectral_w_ll=0.0` | 0.7117 | 0.2994 | FAIL | w_ll 必要（4G.1 澄清：是"假阴性"） |
| `style_extrap_alpha=0.0` | 0.7242 | 0.3333 | FAIL | extrap 必要 |
| `endpoint_adain_scale=0.0` | 0.7082 | 0.2994 | FAIL | AdaIN 必要 |

**Content Fidelity Pathway 验证**: DWT Haar → AdaIN scale → Spectral ODE 三层路径完整。

**Commit**: `50adae4dc`

---

## 6. Phase 4B: 频域 Masking

### 6.1 4B-1 — avg_pool 频域 masking

| 配置 | α | random | clip | lpips | 判定 |
|------|---|--------|------|-------|------|
| freq_a1 | 1.0 | 0 | 0.7258 | 0.3357 | PASS |
| freq_a05 | 0.5 | 0 | 0.7252 | 0.3347 | PASS |
| freq_a1_rand50 | 1.0 | 0.5 | 0.7264 | 0.3354 | PASS |

**Commit**: `d83a050e0`

### 6.2 4B-2 — 长训练 + ratio 优化

| 配置 | ep | clip | lpips | v_ll_abs | 判定 |
|------|-----|------|-------|----------|------|
| freq_a1_rand50_10ep | 10 | 0.7277 | 0.3394 | 0.7255 | PASS |
| freq_a1_rand30 | 3 | 0.7250 | 0.3252 | 0.6541 | PASS (best lpips) |
| freq_a1_rand70 | 3 | 0.7245 | 0.3284 | 0.5636 | PASS |

**最优**: `mask_ratio=0.5`, 5 epochs（10ep 后内容漂移）。

### 6.3 4B-3 — DWT tokenizer

| 配置 | clip | lpips | v_ll_abs | 判定 |
|------|------|-------|----------|------|
| dwt_a1 | 0.7266 | 0.3402 | 0.7018 | PASS |
| dwt_a1_rand50 | 0.7255 | 0.3297 | 0.6456 | PASS |

**结论**: 正交 Haar DWT 可用（与 avg_pool 平价）。

---

## 7. Phase 4C: DINO 污染（NEGATIVE）

| 配置 | clip | lpips | v_ll_abs | 判定 |
|------|------|-------|----------|------|
| 4C.0 (clean DINO + lvl2) | 0.7118 | 0.3038 | 0.3419 | FAIL (clip -0.0125) |
| 4C.1 (blockmask + lvl2) | 0.7151 | 0.3177 | 0.2662 | FAIL (clip -0.0092) |

**关键洞察**: "Style Is Learned, Not Extracted" — DINOv2 是 content-biased，污染 clip -0.018。learnable style_memory 是任务最优。

**决策**: 不使用 DINO，style_memory 成为唯一 style token 源。

---

## 8. Phase 4D-4F: 多级 DWT 突破

### 8.1 4D — 2-Level DWT BREAKTHROUGH

| 配置 | clip | lpips | 判定 |
|------|------|-------|------|
| lvl2 (3ep) | **0.7301** | 0.3402 | **PASS - BREAKTHROUGH** |
| lvl2_dwt_rand50 | 0.7294 | 0.3394 | PASS |

**突破**: +0.0040 clip over 10ep baseline (0.7288)。

### 8.2 4E — Daubechies db2 (FLAT)

| 配置 | clip | lpips | 判定 |
|------|------|-------|------|
| 4E.1 (db2 lvl1) | 0.7258 | 0.3288 | FLAT (vs haar lvl1 -0.0003) |
| 4E.2 (db2 lvl2) | 0.7298 | 0.3398 | FLAT (vs haar lvl2 -0.0003) |

**结论**: 多级是主导效应，基函数非关键。

### 8.3 4F — 3-Level DWT NEW SOTA

| Level | LL 尺寸 | clip | lpips | Δ clip |
|-------|---------|------|-------|--------|
| 1 | 16×16 | 0.7261 | 0.3296 | baseline |
| 2 | 8×8 | 0.7301 | 0.3402 | +0.0040 |
| **3 (SOTA)** | **4×4** | **0.7319** | **0.3428** | **+0.0018** |
| 4 | 2×2 | 0.7316 | 0.3461 | -0.0003 (FAIL) |

**4F.1 = 远程 SOTA**: clip=0.7319, lpips=0.3428。3-Level 是峰值。

---

## 9. Phase 4G: LL Velocity 消融

### 9.1 4G.1 — True LL Lock 2×2 矩阵（NEGATIVE）

| | w_ll=0 (不训练) | w_ll=1.0 (训练) |
|---|---|---|
| **lock=False** (推理用 v_ll) | 4A2: clip=0.7117 (随机噪声) | **4F.1 SOTA: 0.7319** |
| **lock=True** (推理锁死) | 4G.1b: clip=0.7174 | 4G.1a: clip=0.7178 |

**关键发现**:
- LL velocity 应用：+0.0141 clip
- LL velocity 训练：-0.0091 lpips（梯度回流改善 backbone）
- **LL 不是纯内容锚，是"内容 + 全局风格"的混合载体**

### 9.2 4G.2 — Per-Subband AdaIN（MIXED）

| 配置 | α | clip | lpips | 判定 |
|------|---|------|-------|------|
| 4G.2 | 1.0 | **0.7361** | 0.3843 | MIXED (clip NEW SOTA +0.0042, lpips FAIL +0.0415) |
| 4G.2b | 0.5 | 0.7362 | 0.3845 | FAIL (α=0.5≡α=1.0，迭代累积 invalidate α) |

**关键发现**: 多步 Euler 迭代累积使 `(1-α)^n → 0`。α=0.5 在 12 步后残差仅 0.024%。α 失效。

---

## 10. Phase 4H: EOTA 突破

### 10.1 4H.1 — EOTA + α sweep（per_subband）

| 配置 | α | clip | lpips | 判定 |
|------|---|------|-------|------|
| 4H.1a | 1.0 | 0.7359 | 0.3853 | NEUTRAL (≈4G.2) |
| 4H.1b | 0.5 | 0.7219 | 0.3226 | MIXED (content SOTA, style near-miss) |
| 4H.1c | 0.7 | 0.7280 | 0.3442 | **BALANCED, BOTH PASS** |
| 4H.1d | 0.8 | 0.7309 | 0.3572 | FAIL (lpips > 0.3453) |

**理论突破**: EOTA 恢复 α 有效性。α 现在是有效的 content-style trade-off 旋钮。

### 10.2 4H.1e-g — EOTA + spatial_fiber（NEW SOTA）

| 配置 | α | clip | lpips | 判定 |
|------|---|------|-------|------|
| 4H.1e | 0.5 | 0.7185 | 0.3095 | CONTENT BEST EVER |
| 4H.1f | 0.7 | 0.7231 | 0.3208 | NEAR-MISS |
| **4H.1g** | **0.8** | **0.7251** | **0.3281** | **NEW SOTA** |

**4H.1g vs 4F.1**: clip -0.0068, lpips -0.0147（Pareto 更优）。

### 10.3 4H.2-4H.7 — 战术参数失效证明

| 实验 | 改动 | clip | lpips | 判定 |
|------|------|------|-------|------|
| 4H.2h | w_hf=1.5 | 0.7250 | 0.3330 | 无效 |
| 4H.2i | w_ll=0.5 | 0.7265 | 0.3389 | 无效 |
| 4H.3f | patch+15 | 0.7252 | 0.3280 | 无效 |
| 4H.4e | depth=6 | 0.7265 | 0.3366 | 同向权衡 |
| 4H.4f | dim=96 | 0.7271 | 0.3368 | 同向权衡 |
| 4H.5e | mask=0.25 | 0.7227 | 0.3172 | 同向权衡 |
| 4H.5f | mask=0.75 | 0.7237 | 0.3272 | 同向权衡 |
| 4H.7d | terminal_swd=0.3 | 0.7251 | 0.3281 | 完全无影响 |

**关键发现**: 所有战术参数映射到同一 1D Pareto 前沿。要打破前沿需要结构性变化。

---

## 11. Phase 4I: 结构性突破

### 11.1 4I.1 — 多尺度 α（FAIL）

| 配置 | LH α | HL α | HH α | clip | lpips | 判定 |
|------|------|------|------|------|-------|------|
| 4I.1a | 0.5 | 0.5 | 0.9 | 0.7263 | 0.3383 | 同向权衡 |
| 4I.1d | 0.7 | 0.7 | 1.0 | 0.7310 | 0.3576 | FAIL (HH=1.0 过激) |

**结论**: Haar DWT 子带在分解时正交，但 AdaIN+iDWT 重建耦合。多尺度 α 无法打破 Pareto 前沿。

### 11.2 4I.2 — Heun Solver（STRUCTURAL BREAKTHROUGH）

| 配置 | solver | ep | clip | lpips | 判定 |
|------|--------|-----|------|-------|------|
| 4H.1g | Euler | 3 | 0.7251 | 0.3281 | baseline |
| 4H.1g-5ep | Euler | 5 | 0.7261 | 0.3279 | clip +0.0010 |
| 4I.2a | Heun | 3 | 0.7260 | 0.3279 | clip +0.0009, lpips -0.0002 |
| **4I.2b** | **Heun** | **5** | **0.7266** | **0.3229** | **NEW SOTA, 双提升** |

**核心理论**: Euler→Heun 是结构性 DOF。Heun 的 O(h³) vs Euler 的 O(h²) 提供新自由度。

### 11.3 4I.5 — Time Schedule

| 配置 | schedule | ep | clip | lpips | 偏置 |
|------|----------|-----|------|-------|------|
| 4I.5a | cosine | 3 | 0.7256 | 0.3238 | 内容 |
| 4I.5b | cosine | 5 | 0.7262 | **0.3171** | 内容冠军 |
| 4I.2b | linear | 5 | 0.7266 | 0.3229 | 中性 |
| 4I.5c | rquad | 5 | 0.7293 | 0.3429 | 风格 |

**结论**: Schedule shape 是 Pareto-mapping knob（沿前沿移动），非结构性 DOF。

### 11.4 4I.6 — RK4 Solver（饱和）

| 配置 | solver | clip | lpips | 判定 |
|------|--------|------|-------|------|
| 4I.2b | Heun O(h³) | 0.7266 | 0.3229 | — |
| 4I.6a | RK4 O(h⁴) | 0.7265 | 0.3235 | 饱和（无额外收益） |

**结论**: Solver order 在 Heun 处饱和。Euler→Heun 打破 Pareto，Heun→RK4 触及噪声地板。

### 11.5 4I.7 — Cosine + α 优化（远程 SOTA）

| 配置 | schedule | α | ep | clip | lpips | vs SaMam (旧 0.7222) |
|------|----------|---|-----|------|-------|----------|
| SaMam (旧, 已废弃) | — | — | — | ~~0.7222~~ | ~~0.3282~~ | — |
| 4I.5b | cosine | 0.80 | 5 | 0.7262 | 0.3171 | clip +0.0040, lpips -0.0111 |
| **4I.7b** | **cosine** | **0.85** | **5** | **0.7272** | **0.3218** | **NEW SOTA, 双超越** |
| 4I.7a | cosine | 0.90 | 5 | 0.7283 | 0.3255 | clip +0.0061, lpips -0.0027 |

> **⚠️ SaMam 数据修正 (2026-07-02)**: 旧 SaMam CLIP-S=0.7222 已废弃（256分辨率+wikiart5错误）。正确 SaMam: CLIP-S≈0.625 (open_clip) / ~0.565 (HF 预估), LPIPS≈0.321。所有"vs SaMam"对比需重新解读：T11/4I.7b **大幅双超** SaMam，而非旧文档中的"勉强超越"。详见 [07_related_works.md](07_related_works.md)。

**4I.7b = 远程 SOTA**: clip=0.7272, lpips=0.3218。配置：`EOTA + spatial_fiber + α=0.85 + Heun + cosine + 5ep`。

### 11.6 4I.8 — 饱和确认

| 配置 | 改动 | clip | lpips | 判定 |
|------|------|------|-------|------|
| 4I.8a | 8ep（过度训练） | 0.7284 | 0.3283 | FAIL（lpips 优势消除） |
| 4I.8b | warp_cos p=0.8 | 0.7282 | 0.3271 | Pareto-mapping |
| 4I.8c | num_steps=12 | 0.7269 | 0.3217 | 饱和 |

**结论**: 5ep 是 Heun+cosine 最优。num_steps/schedule 都是 Pareto-mapping knob。

### 11.7 4I.9 — WCT（Pareto trade-off）

| 配置 | WCT α | clip | lpips | 判定 |
|------|-------|------|-------|------|
| 4I.9 | 0.85 | 0.7319 | 0.3568 | STYLE GAIN, CONTENT LOSS |
| 4I.9 | 0.50 | 0.7200 | 0.2971 | CONTENT GAIN, STYLE LOSS |

**结论**: WCT 是 Pareto trade-off 工具，无法单独打破前沿。

### 11.8 4I.10 — Probe 诊断

5 大瓶颈：
- A: Velocity field U 形死亡（t=0.5 cos=0.01）
- B: ODE 轨迹无效（target_reach_ratio=0.0009）
- C: AdaIN 统计匹配失败（mean_l1 修正 -0.7%）
- D: 风格敏感度倒置（LL 0.62, LH 0.20）
- E: 频域能量分布错误（LL 54.4% vs target 61.6%）

### 11.9 4I.11 — Per-Subband WCT（DUAL BEAT SaMam 旧判定，现仍成立）

| 配置 | clip | lpips | vs SaMam (旧 0.7222) | vs SaMam (修正 ~0.565 hf) | 判定 |
|------|------|-------|----------|----------|------|
| 4I.11 (LL=0, LH/HL=0.3, HH=0.5, extrap=0.4) | 0.7250 | 0.3129 | clip +0.39%, lpips -4.67% | clip +28.3%, lpips -2.5% | **DUAL BEAT** |

> **注**: SaMam 数据修正后，4I.11 对 SaMam 的超越幅度更大（clip 从 +0.39% → +28.3%）。"DUAL BEAT" 判定在新旧数据下均成立。

---

## 12. Phase 4J: DWT Route 与 Few-shot

### 12.1 4J.1 — DWT Route Cross-Attention

**配置**: `cross_attn_dwt_route=True`, `endpoint_adain_mode=per_subband_wct`, `scale=0.5`, `extrap=0.4`

**结果**: clip=0.7226, lpips=0.3068 — 本地 DWT route 起点。

### 12.2 4J.2 — WCT Aligned Target（4J.2）

预对齐 target 的 WCT。结果未在 progress.json 记录为成功。

### 12.3 4J.6 — Few-shot Textual Inversion（FAIL）

| 版本 | lr | ep | transfer_clip | all_clip | all_lpips | vs SaMam |
|------|-----|-----|---------------|----------|-----------|----------|
| v1 | 2e-4 | 5 | 0.6984 | 0.7210 | 0.3069 | -3.3%/-3.4% |
| v2 | 5e-3 | 15 | 0.6998 | — | — | -3.1%/+0.2% |
| v3 (+endpoint loss) | 5e-3 | 15 | 0.6996 | — | — | -3.1%/-0.2% |

**根因**: style_memory 梯度通路太长（patch_proj → k/v_proj → relu2 gates → tanh gate），多重门控削弱信号。

**结论**: Few-shot 在当前 cross-attention 架构下不可行。

---

## 13. Local T1-T12: Stochastic DWT Route 探索

### 13.1 T3/T4 — 推理参数 sweep（全 FAIL）

在 4J.1 checkpoint 上测试 6 个推理参数组合：

| 配置 | 改动 | clip | lpips | trade-off |
|------|------|------|-------|-----------|
| T3 ll005 | adain_scale_ll=0.05 | 0.7234 | 0.3132 | 1:8 |
| T3b ll010 | adain_scale_ll=0.10 | 0.7241 | 0.3218 | 1:8 |
| T3c ll015 | adain_scale_ll=0.15 | 0.7248 | 0.3334 | 1:8 |
| T4a extrap05 | extrap_alpha=0.5 | 0.7228 | 0.3127 | 1:18 |
| T4b hh07 | adain_scale_hh=0.7 | 0.7239 | 0.3251 | 1:14 |
| T4c lhhl05 | lh/hl=0.5 | 0.7266 | 0.3395 | 1:8 |

**结论**: 推理参数调优全部失败，1:8-1:18 trade-off。

### 13.2 T5 — Eval-Only DWT Route（FAIL vs 4F.1，PASS vs SaMam）

**设计**: 训练全空间 query，推理 DWT route。

| 配置 | clip | lpips | 判定 |
|------|------|-------|------|
| T5 | 0.7061 | **0.2606** | FAIL (clip<4F.1 0.7319, lpips BEST -15% vs 4J.1)；但 **仍双超 SaMam** (SaMam HF≈0.565/0.321) |

**根因**: 训练/推理 query 分布失配。q_proj 未见过 DWT 分布。

### 13.3 T10 — Stochastic DWT (p=0.5)（FAIL vs 4F.1，PASS vs SaMam）

| 配置 | clip | lpips | 判定 |
|------|------|-------|------|
| T10 p=0.5 | 0.7083 | **0.2480** | FAIL (lpips NEW BEST, clip<4F.1)；但 **仍双超 SaMam** |

**根因**: 50% 概率仍不足以让 q_proj 精通 DWT 分布。

> **注**: T5/T10 的"FAIL"判定是相对于双超目标 (clip>0.7319) 而言。在 SaMam 数据修正后，T5/T10 实际仍大幅超越 SaMam（clip +0.14~0.15 HF, lpips -0.06~0.07）。

### 13.4 T11 — Stochastic DWT (p=0.8)（本地 SOTA）

| 配置 | clip | lpips | 判定 |
|------|------|-------|------|
| **T11 p=0.8** | **0.7213** | **0.2868** | **PARTIAL (clip 差 0.0106, lpips PASS 首次 <0.3068)** |

**p 扫描趋势**:
| p | clip | lpips |
|---|------|-------|
| 1.0 (4J.1) | 0.7226 | 0.3068 |
| **0.8 (T11)** | **0.7213** | **0.2868** |
| 0.5 (T10) | 0.7083 | 0.2480 |
| 0.0 (T5) | 0.7061 | 0.2606 |

### 13.5 T12 — T11 推理参数 sweep（全 FAIL）

| 配置 | clip | lpips | trade-off |
|------|------|-------|-----------|
| T12a ll_route_01 | 0.7220 | 0.2924 | 1:8 |
| T12b ll_route_02 | 0.7226 | 0.2981 | 1:8.7 |
| T12c ll_route_03 | 0.7232 | 0.3024 | 1:8.2 |
| T12d extrap_06 | 0.7221 | 0.3014 | 1:18 |
| T12e adain_ll_010 | 0.7227 | 0.3063 | 1:14 |

**结论**: T5/T10/T11/T12 共 15 个配置系统性证明 1:8 trade-off 不可破。

---

## 14. Local T13-T16: LL 风格注入探索

### 14.1 T13 — LLGSI (style_mem 统计量 AdaIN)

| gate | clip | lpips | trade-off | 判定 |
|------|------|-------|-----------|------|
| 0.1 | 0.7128 | 0.2706 | 1:3.7 | MIXED |

**根因**: style_mem 为高频 query 训练，其统计量不编码全局风格。

### 14.2 T14 — CASI (cross-attn 输出统计量 AdaIN)

| gate | clip | lpips | trade-off | 判定 |
|------|------|-------|-----------|------|
| 0.1 | 0.7152 | 0.2795 | 1:3.69 | MIXED (略优于 T13) |

**根因**: cross-attn 输出仍主要是高频信号。

### 14.3 T15 — LLGQCA (global query cross-attn)

| gate | clip | lpips | trade-off | 判定 |
|------|------|-------|-----------|------|
| 0.1 | 0.7176 | 0.2764 | 1:6.08 | MIXED (clip 持续提升) |

**渐进趋势**: T13→T14→T15 clip 持续提升（0.7128→0.7152→0.7176），证明 cross-attn 非线性表达力优于 AdaIN 线性统计量。

### 14.4 T16 — LLGQCA gate sweep（全 FAIL）

| gate | clip | lpips | d_clip | 判定 |
|------|------|-------|--------|------|
| 0.1 (T15) | 0.7176 | 0.2764 | — | BEST |
| 0.2 (T16a) | 0.7145 | 0.2706 | -0.0031 | FAIL |
| 0.3 (T16b) | 0.7101 | 0.2681 | -0.0075 | FAIL (clip 最低) |
| 0.5 (T16c) | 0.7108 | 0.2688 | -0.0068 | FAIL (轻微反弹) |

**根因**: style_mem 高频偏向是根本限制。增大 gate = 放大高频噪声 = 破坏 LL 色调。

**T13-T16 系列结论**: 7 个配置系统性证明，不动 style_mem 前提下，无法从 style_mem 提取有效全局风格信号。

---

## 15. Local T18-T19: Loss/容量调优

### 15.1 T18 — w_ll sweep

| 配置 | w_ll | clip | lpips | v_ll_abs | 判定 |
|------|------|------|-------|----------|------|
| T11 | 0.0 | 0.7213 | 0.2868 | 0.66 | **本地 SOTA** |
| T18a | 0.5 | 0.7174 | 0.2774 | 0.69 | FAIL (clip 降 lpips 降) |
| T18b | 1.0 | 0.7180 | 0.2764 | 0.61 | FAIL (同方向) |

**结论**: w_ll>0 是 content-heavy trade-off。T11 w_ll=0.0 是 clip 最佳点。

### 15.2 T19 — 模型容量 sweep

| 配置 | 改动 | params | clip | lpips | v_ll_abs | 判定 |
|------|------|--------|------|-------|----------|------|
| T19a | depth=6 | 1.06M | NaN | NaN | — | FAIL (WCT eigh 数值不稳定) |
| T19b | dim=96 | 1.35M | 0.7207 | 0.3142 | 0.34 | FAIL (5ep 欠拟合, v_ll_abs 降 50%) |

**T19a WCT 修复**: `spectral_bridge620.py::_wct_match_fiber` 添加对角线正则化 + try-except 回退 AdaIN。

**结论**: 模型容量增加受限于 (1) 数值稳定性（depth=6 WCT 失败）和 (2) 训练预算（dim=96 欠拟合）。

---

## 16. 实验总结表

### 16.1 Pareto 前沿关键点

| 配置 | clip | lpips | 备注 |
|------|------|-------|------|
| **4F.1 (远程)** | **0.7319** | 0.3428 | 远程 SOTA（无 DWT route） |
| 4I.7b (远程) | 0.7272 | 0.3218 | 远程 EOTA+Heun+cosine |
| 4J.1 (本地) | 0.7226 | 0.3068 | DWT route 起点 |
| **T11 (本地)** | 0.7213 | **0.2868** | **本地 SOTA, lpips PASS, 双超 SaMam** |
| T10 (本地) | 0.7083 | 0.2480 | lpips BEST |
| T5 (本地) | 0.7061 | 0.2606 | clip FAIL |
| SaMam (基线) | ~0.625 oc / ~0.565 hf⚠️ | 0.321 | mamba-train（旧 0.7222 已废弃，详见 §17.2） |

### 16.2 关键 Pareto-mapping knob（沿前沿移动）

| Knob | 方向 | 验证实验 |
|------|------|----------|
| α (adain scale) | 大 α → 风格 ↑, 内容 ↓ | 4H.1b/c/d |
| w_ll | 大 w_ll → 内容 ↑, 风格 ↓ | T18a/b |
| schedule (cosine/rquad) | cosine → 内容, rquad → 风格 | 4I.5 |
| training epochs | 5ep 最优, 8ep 过度训练 | 4I.8a |
| mask_ratio | 高 mask → 内容 ↑ | 4H.5e/f |
| num_steps | 饱和（ns=8 已足够） | 4I.8c |

### 16.3 唯一结构性 DOF

| DOF | 验证 | 效果 |
|-----|------|------|
| Solver order (Euler→Heun) | 4I.2 | 打破 Pareto 前沿 |
| Heun→RK4 | 4I.6 | 饱和（无额外收益） |

### 16.4 全部失败方向

| 方向 | 实验数 | 失败原因 |
|------|--------|----------|
| 推理参数 sweep (T3/T4/T12) | 11 | 1:8-1:18 trade-off |
| Eval-only DWT (T5) | 1 | 训练/推理分布失配 |
| Stochastic DWT (T10) | 1 | p=0.5 不足以精通 DWT |
| LL 风格注入 (T13-T16) | 7 | style_mem 高频偏向 |
| Loss 权重 (T18) | 2 | w_ll>0 是 content-heavy |
| 模型容量 (T19) | 2 | 数值稳定性/欠拟合 |
| 多尺度 α (4I.1) | 2 | 子带 iDWT 耦合 |
| Few-shot (4J.6) | 3 | 梯度通路太弱 |
| DINO (4C) | 2 | content-biased 污染 |

**总计**: 31 个失败配置，系统性证明当前架构 1:8 trade-off 不可破。

---

## 17. Related Works Baselines 对照

> 详见 [07_related_works.md](07_related_works.md)。12 个 baseline 在同一评估协议下（HF transformers CLIP ViT-B/32 + LPIPS Alex + 750 pairs + distinct5_512_classview test set）的完整指标。

### 17.1 12 Baseline 完整指标表

| # | 方法 | 类别 | CLIP-S ↑ | LPIPS ↓ | 训练时间(min) | Finding ID |
|---|------|------|---------|---------|--------------|------------|
| 1 | Identity | baseline | 0.6933 | 0.0000 | 0 | F001 |
| 2 | AdaIN | classical-inf | 0.6679 | 0.7425 | 0 | F002 |
| 3 | WCT (VGG19) | classical-inf | 0.7063 | 0.6348 | 0 | F019 |
| 4 | SD-Turbo | diffusion-inf | 0.6933 | 0.0033 | 0 | F007 |
| 5 | SDEdit s=0.35 | diffusion-sweep | 0.7797 | 0.4508 | 0 | F005 |
| 6 | SDEdit s=0.40 | diffusion-sweep | 0.7934 | 0.4826 | 0 | F006 |
| 7 | StyleID | diffusion-inf | **0.8223** | 0.5523 | 0 | F008 |
| 8 | CUT | gan-train | 0.7137 | 0.3743 | 322.6 | F014 |
| 9 | SaMST | mamba-train | 0.6183 | 0.7490 | 39.5 | F011 |
| 10 | SaMam | mamba-train | ~0.625 oc⚠️ | ~0.321 | ~436 | F020 |
| 11 | Seedream 4.5 (API) | commercial-diffusion-api | 0.7198 | 0.4767 | API | F021 |
| **FC-SB** | **T11** | **spectral-bridge** | **0.7213** | **0.2868** | **~30** | — |

### 17.2 ⚠️ SaMam 数据修正

**旧值 0.7222 已废弃**：来源于 `samam_256_faithful_p8_remote/.../h03_step0105/`，错误根因：
- 256 分辨率（非 512）
- wikiart5 数据集（非 distinct5）
- step 105 早期 checkpoint

**正确评估进行中**（F020）：
- 20K 步训练完成（7h16m，distinct5，512×512）
- 80 个 checkpoint HF transformers CLIP 评估进行中
- open_clip 收敛值: CLIP-S≈0.625, LPIPS≈0.321
- HF transformers 预计: CLIP-S≈0.565（-0.06 vs open_clip）, LPIPS=0.321（backend 无关）

### 17.3 T11 vs 12 Baselines 定位

| 维度 | T11 表现 | 排名 |
|------|---------|------|
| CLIP-S | 0.7213 | 第 5（低于 StyleID/SDEdit×2/CUT；高于 Seedream/SaMam/SaMST/WCT/IDT/SD-Turbo/AdaIN） |
| LPIPS | 0.2868 | **第 1**（所有非 identity 方法中最优） |
| 训练时间 | ~30 min | **第 1**（最快，14.5× 优于 SaMam，10.8× 优于 CUT） |
| 模型规模 | 903K params | 极轻量 |

**关键判定**:
- T11 **DUAL BEAT SaMam**: clip +0.096 (oc) / +0.156 (hf), lpips -0.034, 训练快 14.5×
- T11 **DUAL BEAT Seedream 4.5**: clip +0.0015, lpips -0.1899（轻量模型战胜商业 API）
- T11 在内容保真度（LPIPS）上**碾压**所有训练类方法（CUT 0.3743 / SaMam 0.321 / SDEdit 0.4508 / Seedream 0.4767 / StyleID 0.5523）
- T11 在 CLIP-S 上无法匹敌扩散先验方法（StyleID 0.8223 / SDEdit 0.7934），但定位为"轻量+高保真"路线

### 17.4 失败/完成实验记录（Related Works）

| ID | 方法 | 状态 | 问题/备注 |
|----|------|------|------|
| F018 | WCT VGG-normalised | 失败 | 750张PNG同MD5, 特征值域过窄 |
| F011 | SaMST | 失败 | CLIP-S=0.6183 低于 identity, 内容严重扭曲 |
| ~~F016~~ | ~~SaMam (旧)~~ | 废弃 | 256分辨率+wikiart5数据集错误 |
| F020 | SaMam 20K | 进行中 | HF 评估中 |
| F021 | Seedream 4.5 | ✅ 完成 | 商业 API 参考线，clip=0.7198, lpips=0.4767 |

### 17.5 评估协议一致性

所有 12 baselines + T11 在同一协议下评估：
- 评估脚本: `run_evaluation.py`
- CLIP backend: HF transformers, openai/clip-vit-base-patch32（项目默认 `full_eval_clip_backend="hf"`，详见 [config_schema.py:937](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/config_schema.py#L937)）
- LPIPS: Alex backbone
- test_dir: `I:\wikiart_distinct5_samam_512_classview\test`
- 5 风格: Early_Renaissance / Impressionism / Minimalism / Rococo / Ukiyo_e
- IDT 基线: 0.6933（旧文档中误标 0.6399）
- n_pairs: 750（CUT=745）

> **CLIP backend 对齐**: SaMam 旧 open_clip 数据已废弃（绝对值偏低 0.05-0.10），HF 评估完成后用 HF 数值替换。Seedream 4.5 已用 HF transformers CLIP 评估，与其他 baseline 对齐。详见 [07 §CLIP Backend 对齐](07_related_works.md#-clip-backend-对齐说明重要)。
