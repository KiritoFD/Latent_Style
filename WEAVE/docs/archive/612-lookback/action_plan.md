# 612 回顾 — 突破性能瓶颈的行动计划

> 2026-06-13 后记：本文保留为 612 回顾时的历史行动草案，不再直接作为 Distinct5 正式实验排队依据。当前执行请以 [../612-phase2/README.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/612-phase2/README.md) 为准；其中 `LPIPS >= 0.70` 已被收紧解释为 complete failure，`0.40 <= LPIPS < 0.70` 只保留 archival-only 地位，SDE / exact-I2SB 不再作为正式主线。

> 2026-06-13 implementation correction:
> tokenizer-deepening items listed below are no longer literal TODOs in code.
> The current runtime already has deeper residual query extraction, larger cluster support, 2D position encoding, and pooled global-spatial coupling.
> The remaining question is no longer “has tokenizer depth/PE been implemented?” but “do the current safe-band runs show that the stronger tokenizer is actually creating useful routing on the board?”

> 2026-06-13 appearance follow-on note:
> the first safe topology-gate reentry point already recovered the old all-pairs shelf while still trailing transfer style slightly.
> That weakens the "pure structure failure" reading and raises a narrower hypothesis:
> some of the remaining board gap may be low-order appearance mismatch such as brightness / contrast / exposure statistics.
> The current codebase now has a conservative tokenizer-guided output appearance head so this hypothesis can be tested as a same-family phase2 follow-on instead of jumping immediately to a new structure family.

## 当前状态速览

```
                   style ↑
                   0.74 ┤                    ● SaMST e5 (LPIPS 0.63) 
                   0.73 ┤         ● xpred+kmanifold+pattn+stokes002 (LPIPS 0.62)
                   0.72 ┤    ● xpred+kmanifold (LPIPS 0.68)     ● wikiart2 F_e1 (LPIPS 0.32)
                   0.71 ┤
                   0.70 ┤ ● K_e1  H_e2  F_e1  SaMAM step3k     ● wikiart1 F_e1
                   0.69 ┤
                   0.68 ┤ ○ IDT (no-op anchor)
                         └──┬───────────┬───────────┬───────────┬──→ LPIPS ↓
                           0.0         0.3         0.5         0.7
                          (perfect)  (good)     (acceptable)  (bad)
```

**核心矛盾:**
- LBM baseline 在 distinct5 上无法突破 style≈0.70 + LPIPS≈0.33 的 ceiling
- xpred+pattn 族可以推 style 到 0.73，但 LPIPS 崩溃到 0.60+
- **两者之间有一个巨大的 gap — 没有任何方法同时做到 style>0.72 且 LPIPS<0.40**
- 而 wikiart_stress2 上 F_e1 可以达到 0.72/0.32 — 说明网络 capable，问题在 distinct5 这个 harder dataset 上

---

## 瓶颈诊断

### 瓶颈#1: Endpoint vs Velocity — 模式选择

| 模式 | 目前 best (distinct5) | 优势 | 劣势 |
|------|----------------------|------|------|
| **velocity** (LBM F/H/K) | style 0.70, LPIPS 0.32-0.36 | LPIPS 好 | style 天花板低 |
| **endpoint** (xpred+pattn) | style 0.73, LPIPS 0.60 | style 天花板高 | LPIPS 爆炸 |

**根因**: endpoint 模式直接预测 x_1，不预测 delta。训练时目标 `matched_target` 是 OT 后的风格化 latent（已大幅偏离源内容）。网络学会的是"重绘"而非"编辑"。

**行动**: 回归 velocity 模式 + xpred 的 proximal 修复能力。velocity 预测 delta（编辑），本质上是保内容的；但之前 velocity 模式的 style 天花板低是因为 kinetic 限制了移动。能否:
- 用 velocity 模式
- 加 xpred+kmanifold 的 manifold-adaptive kinetic（而非 global L2）
- 再加 pattn 的 cross-attn proximal refinement
- 这就是 **velocity + k-manifold + pattn** 组合

### 瓶颈#2: PureLatentSpatialTokenizer 能力评估

**现状**: Round2 正在 3060 远程训练 pure_latent_spatial tokenizer 各组。
**关键缺失**: 目前没有看到 round2 results — 需要紧急获取。

| 组 | solver | sigma | batch | 预期 |
|----|--------|-------|-------|------|
| tok_pure_latent_spatial | euler_legacy | 0.0 | 16 | 纯 tokenizer 有效性 |
| sde_i2sb_sigma_0p25 | solver_i2sb | 0.25 | ~24-28 | 轻度噪声 |
| sde_i2sb_sigma_0p5 | solver_i2sb | 0.5 | ~28-34 | 推荐级噪声 |
| sde_i2sb_sigma_1p0 | solver_i2sb | 1.0 | ? | 极端噪声 |

**若 round2 results 显示 pure_latent_spatial tokenizer 在 LPIPS/style tradeoff 上持平或弱于 legacy_factorized:**
→ tokenizer 增强是首要任务（见下）

**若 round2 results 显示 SDE (solver_i2sb) 不优于 ODE (euler_legacy):**
→ SDE 路线暂停，转 PC solver

### 瓶颈#3: 结构保持机制缺失

当前代码中没有任何显式的内容保真机制:
- ❌ 无 content loss
- ❌ 无 self-attention injection (PnP style)
- ❌ 无 cycle consistency
- ✅ 仅有 residual gain + latent scale factor（硬编码的）

**这就是为什么 style 推到 0.73 时 LPIPS 崩到 0.60** — 没有一个主动的力把生成结果拉回内容原点。

**solver_pc (Predictor-Corrector) 是最被低估的 solver**:
- 先走 ODE (predictor)，再通过内容校正步 (corrector) 拉回源图附近
- 已实现但 round1 结果在 remote，需要拉取分析
- 这是最"便宜"的结构保持方案 — 不改训练代码，不改架构

---

## 行动计划（按优先级）

### 🔴 优先级 1: 紧急拉取 Round2 results

**行动**: 从远程 I:\GitHub\Latent_Style\SchrodingerBridge\exp\inmortal-exp\ 拉取:
1. `pure_latent_spatial/` 组的 full_eval summary
2. `sde_i2sb_sigma_0p5/` 组的 full_eval summary
3. `tok_baseline_global/` 组的 full_eval summary

**判断准则**:
- 如果 pure_latent_spatial 在 LPIPS < 0.40 时 style > 0.70 → tokenizer 有效，继续 SDE 路线
- 如果 pure_latent_spatial style < 0.68 → tokenizer 不足，需要增强
- 如果 sde_i2sb 不优于 tok_pure_latent_spatial (ODE) → SDE 无益，停 SDE 转 PC

### 🟡 优先级 2: 回归 velocity + k-manifold + pattn

无论 round2 results 如何，这是 **最可能同时提升 style 和保持 LPIPS** 的组合:

```
model:
  tokenizer_family: legacy_factorized (当前最好用的)
  transport_prediction_mode: velocity  ← 关键切换
  solver_family: euler_legacy
  backend_attention_family: attn_pnp_selfinject  ← 结构保持
  proximal_mode: crossattn_texture  (pattn)
  kinetic_penalty_mode: manifold_adaptive_split  (k-manifold)
  kinetic_lambda_low: 1.0
  kinetic_lambda_high: 0.02

bridge:
  objective_mode: omf
  bridge_sigma: 0.0
  w_flow: 1.0
  w_kinetic: 1.0  ← 需要扫描 0.5-2.0
  semantic_swd_num_projections: 64
```

**为什么这个组合最可能工作**:
- velocity 模式保内容（预测 delta 而非 x_1）
- k-manifold kinetic 允许高频区域自由移动，只限制低频结构
- pattn proximal 在 endpoint 预测后做 cross-attn texture refinement
- attn_pnp_selfinject 在 attention 层注入结构

### 🟡 优先级 3: 启动 solver_pc 评估

**行动**: 拉取 round1 remote solver_pc 的 eval 数据。

solver_pc 是 ODE + content correction:
- 不需要改变训练
- 不需要新架构
- 在每一步 inference 时自动纠正结构偏差

**配置**:
```
model:
  solver_family: solver_pc
  solver_corrector_steps: 2  (扫描 1/2/4)
  solver_corrector_step_size: 0.1
```

如果 PC solver 能在保持 style 的同时显著改善 LPIPS → 这是最快路径。

### 🟢 优先级 4: Tokenizer 增强（多步实验）

如果 round2 显示 pure_latent_spatial tokenizer 不足:

1. **加深 query_extractor**: 2 层 Conv → 4-5 层 ResBlock
2. **增加 clusters**: 16 → 32 或 64
3. **添加位置编码**: sinusoidal 或 learnable position embedding
4. **增强 global_code**: 把 global_code 和 spatial_map 通过 cross-attn 关联

---

## 关键决策树

```
Round2 pure_latent_spatial results 怎么样?
├─ style>0.70 且 LPIPS 可接受 → ✅ 继续 SDE 路线，启动 i2sb sigma sweep
│   └─ sde_i2sb sigma=0.5 优于 ODE? 
│       ├─ 是 → ✅ 推进到论文
│       └─ 否 → ⚠️ 可能是训练-推理分布不匹配问题 → 修复 _bridge 加噪逻辑
│
└─ style<0.68 或 LPIPS 崩 → ❌ tokenizer 不够
    ├─ 选项A: 增强 tokenizer (3-5天开发)
    ├─ 选项B: 回归 legacy_factorized + velocity + k-manifold + pattn
    └─ 选项C: 直接切 solver_pc，不改架构和训练，仅在推理改善

当前最快的推进路径:
  velocity + k-manifold + pattn  (3天训练) + solver_pc 评估 (1天)
```

---

## 实验队列建议

| # | 实验名 | 配置 | 预期训练时间 | 目标 |
|---|--------|------|-------------|------|
| 1 | vel_kman_pattn | velocity+kmanifold+pattn, b16 | ~20min/epoch ×8 | 突破 LBM 0.70 ceiling |
| 2 | vel_kman_pattn_kin_sweep | 同上, scan w_kinetic 0.5/1.0/2.0 | ~60min total | 找到最佳 kinetic 权重 |
| 3 | pc_solver_eval | 用现有 K_e1 或 F_e1 ckpt + solver_pc | 仅eval | 立即验证 PC 有效性 |
| 4 | pure_latent_spatial_enhanced | deepened query_extractor + 32 clusters | ~25min/epoch ×8 | 验证 tokenizer 增强 |
