# 612 Lookback — SchrodingerBridge Experiment Analysis & Redesign Plan
## 2026-06-12 全面回顾

> 2026-06-13 后记：本文是回顾诊断，不是当前 Distinct5 正式执行权威。当前正式计划请以 [../612-phase2/README.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/612-phase2/README.md) 为准；其中 `LPIPS >= 0.70` 已定义为 complete failure，`0.40 <= LPIPS < 0.70` 只保留 archival-only 解释，exact-I2SB / SDE 主线建议已退役。

> 2026-06-13 implementation correction:
> the tokenizer implementation has already moved beyond the early weak sketch described in parts of this note.
> Current code now uses:
> - configurable residual query blocks
> - configurable cluster count
> - 2D sine/cosine position encoding
> - pooled global-spatial coupling
> - expanded runtime observability including tokenizer effective-count / gate / mask / spatial-energy reads
> Therefore any remaining tokenizer diagnosis in this file should be read as a hypothesis about board behavior, not as a statement that the runtime is still on the old `2-layer conv + 16 clusters + no PE` path.

> 2026-06-13 appearance-side hypothesis:
> the first safe semantic-topology-gate recovery point has now shown a cleaner LPIPS band and an all-pairs shelf break without yet clearing the transfer shelf.
> That suggests at least part of the remaining gap may be low-order appearance mismatch rather than missing structure routing alone.
> A conservative tokenizer-guided output appearance head is now available in code so this hypothesis can be tested directly in phase2 without reverting to endpoint-style style amplification.
>
> 2026-06-13 exact-I2SB follow-on note:
> the next exact-I2SB diagnostic should not add another heuristic noise window.
> the cleaner follow-on is to keep the exact posterior coefficients unchanged and only floor the predictor time on the earliest step, so the `x_1` estimator is no longer queried at exact `t=0` even though training samples `t` inside `(eps, 1-eps)`.

---

## 〇、远程机器环境 (100.115.18.62)

### Windows Host
| 项 | 值 |
|----|-----|
| GPU | NVIDIA 3060 12GB (8.6GB used / 3.4GB free) |
| Driver | 581.08 |
| SSH | port 2222, user administrator |

### WSL2
| 项 | 值 |
|----|-----|
| Distro | Ubuntu-26.04 (Running, WSL2) |
| Python | 3.x |
| Torch | 2.11.0+cu128, CUDA available |
| Diffusers | 0.38.0 |
| Transformers | 5.9.0 |
| lpips | 0.1.4 |

### 实验目录
| 路径 | 说明 |
|------|------|
| `I:\GitHub\Latent_Style\SchrodingerBridge\exp\` | 主实验目录 (~200+ experiments) |
| `I:\GitHub\Latent_Style\SchrodingerBridge\exp\inmortal-exp\` | immortal实验系列 (45+ runs) |
| `I:\GitHub\Latent_Style\Related_Works\baseline_pipeline\` | SaMAM/SaMST 基线 |
| `I:\GitHub\Latent_Style_TokenizerClean\SchrodingerBridge\` | Tokenizer 独立工作区 |
| `I:\latent_style_remote_curated\` | 整理归档 |

### ⚠️ Round2 Pure-Latent SDE 实验状态
**未找到 tok_pure_latent_spatial、sde_i2sb_sigma_* 目录** — 这些 experiment 的目录在远程 I: 盘上不存在，可能已被清理或从未完整训练。需要确认是否需要重新启动 round2。

### WikiArt512 参考数据（远程 archives 中）
| Experiment | clip_style | LPIPS | 说明 |
|------------|-----------|-------|------|
| spectral_stat_full_adapt_e2 (trueint) | 0.791 | 0.307 | 最佳平衡点 |
| spectral_stat_full_adapt_e2 (gain4) | 0.791 | 0.309 | 高 gain |
| lowfreq_moment_hfguard_tsw8 | 0.797 | 0.308 | 高频保护 |
| truegrad_tokenbudget_full_e1 | 0.791 | 0.316 | budget token |
| direct_atom_residual_e8 | 0.789 | 0.360 | baseline reference |
| IDT no-op | 0.795 | 0.000 | 恒等变换 |

💡 **WikiArt512 上 LBM 可以达到 style~0.79 + LPIPS~0.31** — 而 Distinct5 只有 style~0.70 + LPIPS~0.32。这说明:
- 模型能力本身是足够的，瓶颈不在架构
- **Distinct5 的挑战是目标风格更难区分**（5个风格中 Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e 特征差异不如 WikiArt 的 5 类大）
- 提升 Distinct5 需要更强的 style-specific 特征提取

### plan-612.md — Pure-Latent Tokenizer + I2SB SDE 路线

核心理念: 抛弃 DINO 等外部先验, 构建纯内生 (Endogenous) 潜空间架构:
- **Tokenizer**: 从 z_0 本身提取路由特征 → 全局 Keys → 风格 Values → spatial map
- **Solver**: I2SB 精确后验求解器, 用布朗噪声打破 ODE 的方差坍缩
- **Loss**: 极简化, 只有 Flow Loss + Terminal SWD, 删除所有 heuristic losses

设计波次:
- Wave1: Tokenizer 架构验证 (global vs spatial)
- Wave2: SDE vs ODE & noise sweep (0.25/0.5/1.0)
- Wave3: Heuristic Loss Ablation
- Wave4: NFE 效率测试 (4/8 steps)

### bridge.md — I2SB 精确后验数学原理

三个方案:
1. **I2SB** (endpoint prediction, exact posterior) — 推荐首选
2. **Stochastic Flow Matching** (velocity prediction, Euler-Maruyama)
3. **Langevin Predictor-Corrector** (不改变训练, 仅在推理注入 SDE)

训练: `x_t = (1-t) * x_0 + t * x_1 + sigma * sqrt(t*(1-t)) * ε`
推理: `mu = c_curr * x_t + c_target * x_1_pred`
      `var = sigma^2 * (t_next - t_curr)(1 - t_next) / (1 - t_curr)`

### attn.md — Backbone Attention 改造

三种空间保持方案:
1. **Spatial-Modulated Self-Attention (SA-Mod)**: Content Q×K 锚定结构
2. **Gromov-Wasserstein OT Attention**: 距离-距离匹配
3. **SPADE-Attention**: 空间门控 + Window Attention

---

## 二、当前代码实现状态

### ✅ 已实现

| 组件 | 文件 | 状态 |
|------|------|------|
| PureLatentSpatialTokenizer | semantic_tokenizer.py:83-160 | 已实现 |
| I2SB 训练加噪 | losses.py:417-438 | 已实现 |
| I2SB 推理求解器 | model.py:516-542 | 已实现 |
| I2SB 合约验证 | style_families.py:156-170 | 已实现 |
| Round2 实验注册 | round2_registry.py | 已实现 |
| 多 Solver 框架 (PC, RK, UNSB) | model.py:855-918 | 已实现 |

### ❌ 未实现 / 已禁用

| 组件 | 原因 |
|------|------|
| DINO Tokenizer (tok_a/b/c/d) | 依赖外部 ViT, 已有实现但不在主线 |
| attn_gw_ot | 计算开销大, 仅归档 |
| attn_gated_spade | 在 round1 中测试过 |
| SA-Mod Attention | 在 aaai2027/round1_attn_sa_mod_fast_local/ 中测试过 |

### 💡 实际主线的 Tokenizer 流程

```
content_latent (z_0) [B, 4, 64, 64]
    ↓
query_extractor (configurable residual query stack)
    ↓
2D position encoding + normalized queries
    ↓
attention with universal_keys [1, K, query_dim] / temperature
    ↓
attn_weights [B, HW, K] × style_values [B, K, spatial_dim]
    ↓
spatial_map [B, spatial_dim, H, W] + gate_map + mask_map
    ↓
global_code = base_style_code + raw_style_global + pooled_spatial_global_gate
```

---

## 三、关键实验结果汇总

### 3.1 Local Best Points (Distinct5-512)

| Experiment | clip_style | content_lpips | delta_idt | Notes |
|------------|-----------|---------------|-----------|-------|
| **LBM H_e2** | 0.6994 | 0.3484 | +0.0285 | 当前 baseline |
| **LBM K_e1** | 0.7010 | 0.3623 | +0.0312 | 最高 style |
| **LBM F_e1** | 0.6969 | 0.3186 | +0.0244 | 最好 LPIPS |
| **inmortal xpred+kmanifold+pattn** | 0.7338 | 0.6278 | — | 风格天花板, LPIPS 崩溃 |
| **inmortal xpred+kmanifold+pattn+stokes002** | 0.7307 | 0.6183 | — | 当前 xpred 最优 |
| **SaMAM step3000** | 0.6978 | 0.3221 | +0.0247 | 强 LPIPS baseline |
| **SaMST e5** | 0.7276 | 0.6271 | +0.0590 | 极端风格, LPIPS 极高 |
| **idt no_op** | 0.6801 | 0.0000 | 0.0000 | 恒等变换参考 |

### 3.2 Local Best Points (WikiArt Stress1)

| Experiment | clip_style | content_lpips | delta_idt |
|------------|-----------|---------------|-----------|
| **LBM F_e1** | 0.7221 | 0.3244 | +0.0074 |

### 3.3 Local Best Points (WikiArt Stress2)

| Experiment | clip_style | content_lpips | delta_idt |
|------------|-----------|---------------|-----------|
| **LBM F_e1** | 0.7406 | 0.3188 | +0.0013 |

### 3.4 SaMAM Latent Baseline (failures)

| Experiment | clip_style | content_lpips |
|------------|-----------|---------------|
| distinct5 step5000 fast | 0.6509 | 0.8092 |
| legacy256 step5000 fast | 0.6509 | 0.8092 |

### 3.5 SaMST Latent Baseline (failures)

| Experiment | clip_style | content_lpips |
|------------|-----------|---------------|
| b50_fast | 0.6104 | 0.7296 |
| b300_fast | 0.6104 | 0.7296 |

### 3.6 Round1 Attention / Solver Experiments

| Experiment | Status |
|------------|--------|
| round1_attn_sa_mod_fast_local | 24 epochs trained, checkpoints available |
| round1_attn_gated_spade_remote | Full eval pulled |
| round1_attn_pnp_selfinject_remote | Full eval pulled |
| round1_solver_pc_fast_local | 36 epochs trained |
| round1_solver_tangent_rk_remote | 32 epochs trained |
| round1_solver_unsb_cycle_remote | 30 epochs trained |
| round1_tok_b_cross_image_remote | Full eval pulled |
| round1_tok_c_residual_adapter_remote | Full eval pulled |

### 3.7 Remote Round2 Pure SDE Experiments

| Experiment | sigma | Status |
|------------|-------|--------|
| tok_baseline_global | 0.0 | Running |
| tok_pure_latent_spatial (c1-c11) | 0.0 | Running |
| sde_i2sb_sigma_0p25 (b24c3-b28c1) | 0.25 | Running |
| sde_i2sb_sigma_0p5 (b28c4-b34c1) | 0.5 | Running |
| sde_i2sb_sigma_1p0 | 1.0 | Running |
| optimal_clean | 0.5 | Running |
| optimal_with_heuristics | 0.5 | Running |

---

## 四、当前架构诊断

### 4.1 Tokenizer 层 — PureLatentSpatialTokenizer

**现状**: 已实现, 在 round2 中运行。

**设计优点**:
- 完全内生, 零外部依赖
- 路由机制 (Query→Key→Value) 理论上可以从 z_0 的自组织聚类中学习空间语义
- gate_map 和 mask_map 提供自适应注入强度

**当前真正还需要回答的问题**:

1. **实现层面的旧弱点已经大多补上, 现在要看 board 证据**:
   - deeper residual query extractor: 已实现
   - larger cluster count: 已实现并可配置
   - 2D position encoding: 已实现
   - pooled global-spatial coupling: 已实现
   - 因而 tokenizer 的核心问题已经从 “有没有这些能力” 变成 “这些能力在 Distinct5 safe-band 上有没有转化成可见收益”

2. **当前更重要的是路由是否真的被用到**:
   - 现在日志里已经可以看:
     - `attn_entropy`
     - `attn_effective_count`
     - `gate_mean`
     - `mask_mean`
     - `spatial_map_abs`
     - `global_gate_abs`
   - 真正的 next question 是:
     - safe-band 运行里这些量是否稳定活跃
     - 它们的变化是否和 style / LPIPS 走向一致

3. **style global branch 是否真的参与了有效控制**:
   - pooled global-spatial coupling 虽然已实现
   - 但仍需要看 `global_gate_abs` 和最终 board 行为是否证明它提供了有效的 global prior, 而不是只在局部 spatial path 上工作

4. **真正剩下的 tokenizer 风险更偏向“能力没有转化成收益”而非“模块缺失”**:
   - 如果 safe-band 上 tokenizer 观测一直活跃, 但 board 仍停在 `0.70x / 0.37x-0.38x`
   - 那么问题更可能在训练-side structure control / solver / loss geometry, 而不是 tokenizer 基础能力没有写出来

### 4.2 Bridge / SDE 层

**现状**: I2SB 精确后验求解器已实现且数学正确。

**训练阶段** (losses.py):
- `x_t = (1-t) * content + t * matched_target + sigma * sqrt(t*(1-t)) * noise`
- Loss: `MSE(pred_x1, matched_target)`

**推理阶段** (model.py):
- 正确实现了 I2SB 公式: `mu = c_curr * h + c_target * x_1_pred`, `var = sigma^2 * (t_next - t_curr)(1 - t_next) / (1 - t_curr)`

**为什么 SDE 可能没有带来好的结果**:

1. **训练-推理分布不匹配**: 
   - 训练时 x_t 带有布朗噪声, 网络学习从噪声状态预测 x_1
   - 但在推理时, 初始状态 h=x (源图) 是干净的, 没有噪声!
   - 直到中间步骤注入噪声, 网络才遇到带噪输入
   - 这意味着网络在推理过程中的第一个预测 (t=0 附近) 已经是 out-of-distribution 的
   - **核心矛盾**: 训练时 t 越接近 0 噪声越小, 推理时第一步就要求大噪声预测

2. **噪声幅度与 latent 范围的适配问题**:
   - VAE latent 通常归一化到 ~[-3, +3] 范围
   - sigma=0.5 的方差在 t=0.5 时: sqrt(t*(1-t)) = 0.5, 所以实际噪声 std = 0.5 * 0.5 = 0.25
   - 这对于 latent 的尺度来说可能刚好合适, 也可能太大
   - sigma=1.0 时 std=0.5, 几乎占 latent 范围的 1/6, 可能太强

3. **网络架构可能不适合去噪**:
   - LANCETBridge 是为 Flow Matching (ODE) 设计的, 不是为去噪 (Denoising) 设计的
   - 它没有像 U-Net + Time Embedding 那样的去噪架构
   - predict_transport_base 接收 t 和 x_t, 但时间条件方式可能不够强用于去噪

4. **多步推理中的误差累积**:
   - 每一步注入随机噪声, 如果 x_1_pred 有误差, 噪声会加剧误差
   - 多步后误差累积可能导致最终质量下降而非提升

5. **对比 ODE (bridge_sigma = 0) 的优势不明显**:
   - ODE 在 xpred 系列中已经能达到 0.73+ clip_style (但 LPIPS 高)
   - SDE 声称能打破方差坍缩, 但实际效果取决于网络能否在噪声下预测准确的 x_1
   - 如果 x_1_pred 不准, SDE 只是加噪声, 不会帮助

### 4.3 架构层面的根本问题

**核心矛盾: Style-Up vs Structure-Down (风格-结构 tradeoff)**

| 方法 | Style | LPIPS | 特征 |
|------|-------|-------|------|
| LBM baseline (H/F/K) | 0.70 | 0.32-0.36 | 平衡点 |
| xpred + pattn + stokes | 0.73 | 0.59 | 风格强但结构崩 |
| SaMAM (RGB 256) | 0.70 | 0.32 | 纯 RGB 的方法 |
| SaMST (latent 512) | 0.73 | 0.63 | 风格强但破坏大 |

**问题**: 
1. 当前 LBM 在 Distinct5 上的天花板大约是 clip_style=0.70, content_lpips=0.33
2. 要突破这个天花板, 引入 xpred + pattn 会把 style 提升到 0.73, 但 LPIPS 崩溃到 0.59-0.63
3. SaMAM (SSM, RGB 256) 可以达到 0.70/0.32, 与我们持平但不在 latent space
4. **关键差距**: 我们缺少一个能同时提升 style 又保持结构的方法

**可能的原因**:

1. **没有内容保真度约束**: 
   - 当前只有 Terminal SWD (风格分布匹配) + Flow Loss (预测终点)
   - 没有显式的内容保持机制 (如 content loss, cycle consistency, 或 self-attention injection)
   - xpred 模式直接预测 x_1 而不是 delta, 更容易丢失内容

2. **PureLatentSpatialTokenizer 的 spatial map 可能不够强**:
   - spatial map 是 style-specific 的, 不包含 content 的结构信息
   - 它是额外加的偏置, 但不能保证内容不丢失
   - 需要更强的内容先验注入

3. **Skip connections 可能是瓶颈**:
   - 当前 skip 路径通过 `skip_router` 融合, 有 `skip_disabled` 和 `skip_fusion_mode` 
   - 如果 skip 太强, style 上不去; 如果 skip 太弱, 结构保不住
   - 需要自适应 skip 门控

### 4.4 Solver 层面

| Solver | 特点 | 适用 |
|--------|------|------|
| euler_legacy | ODE, 单步或少数步 | 快速, 流畅 |
| solver_i2sb | SDE, 多步 | 打破方差坍缩 |
| solver_tangent_rk | RK4 ODE, 多步 | 高精度路径 |
| solver_pc | ODE + 内容校正 | 结构保持 |
| solver_unsb_cycle | SDE + 校正 + 周期一致性 | 理论上最佳 |

**solver_pc** 可能是最被低估的 solver:
- 先走 ODE, 再通过内容校正步把结果拉回源图附近
- 这在理论上直接解决了结构保持问题
- 在 round1 中运行了 36 epochs, 但结果未仔细分析

---

## 五、下一步建议

### 优先级 1: Tokenizer 增强 (低风险, 高回报)

1. **增加 query_extractor 深度**: 2层 → 4-5层, 带 skip connections
2. **增加 num_clusters**: 16 → 32 或 64
3. **添加位置编码**: 使 queries 包含空间坐标信息
4. **实验 tokenizer 内部状态**: 检查 attention entropy 是否足够高 (cluster 是否充分使用)

### 优先级 2: Content 保真度修复

1. **PC Solver + Content Correction**: 用 solver_pc 在推理时自动校正
2. **Content Loss (可选)**: 添加轻量级 content preservation loss
3. **自适应 Skip Gate**: 根据 content_latent 自动调节 skip connection 强度

### 优先级 3: SDE 诊断

1. **先验证 I2SB 训练有效**: 检查 bridge_sigma=0.5 训练时的 x_t 噪声水平, 确认网络能 predict x_1
2. **对比 ODE vs SDE 在少步数 (NFE=4) 下的质量**
3. **检查 SDE 推理时的噪声注入时机** — 也许从 t=0.2 开始加噪声更好

### 优先级 4: 回归更稳健的 baseline

1. **评估 solver_pc round1 结果**
2. **回归 velocity 模式** (不是 endpoint) — velocity 预测 delta, 更容易保持内容
3. **考虑简化的 Cycle Consistency** — 这是 UNSB 论文的核心卖点

---

## 六、清理建议

### 本地可删除的 checkpoint

以下 checkpoint 可安全删除 (按大小从大到小):

```
# round1_attn_sa_mod_fast_local 中间 checkpoint (保留 epoch_0024 最后)
epoch_0001.pt .. epoch_0023.pt (保留 0024)

# S-add 实验中的中间 epoch
S-add__K-1_C-0_W-20_Col-0/full_eval/epoch_0001..0007 (保留 0008)

# aaai2027 根目录下的陈旧 checkpoint
carriergate_fresh_epoch_0002.pt
hold4twostage_epoch_0002.pt
knee_carriergate_fresh_epoch_0002.pt

# 大文件 archive
archives/exp_archive_20260526_051536/
archives/old_experiment_dirs/
archives/old_paper_workspaces/
```

### 远程可删除的内容

参考 remote_i_curated 中的 cleanup list。
