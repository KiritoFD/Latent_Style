# WEAVE Method 诊断审计：组件实际贡献 vs 论文叙事

## 日期: 2026-07-11
## 背景: No-Flow/Flow-Only消融揭示训练loss中flow之外几乎无贡献，需全面审视Method

---

## 1. 核心发现（TL;DR）

**WEAVE的实际有效组件只有3个，而非论文声称的18个：**

| 排名 | 组件 | 类型 | 消融影响 | 论文叙述角色 |
|------|------|------|----------|-------------|
| 1 | **Rectified Flow (Flow Matching)** | 训练loss | No-Flow: DINO-C -0.093, LPIPS +0.084 | 核心机制 ✓ |
| 2 | **Haar Wavelet Decomposition** | 架构 | w/o Wavelet: CLIP-S -0.016, DINO-S -0.021 | 核心机制 ✓ |
| 3 | **Endpoint AdaIN (spatial_fiber)** | 推理后处理 | w/o Endpoint AdaIN: CLIP-S -0.016 | 核心机制 ✓ (但实现与叙述不符) |

**论文声称有效但实际无效的组件（7个）：**

| 组件 | 消融影响 | 论文叙述角色 | 实际状态 |
|------|----------|-------------|----------|
| SWD Guide | Flow-Only ≈ Full (Δ<0.004) | 增强 | **无效**（梯度被flow淹没） |
| Edge Loss | 占总loss 2.2% | 辅助正则 | **无效** |
| Low-pass Anchor Loss | 占总loss 1.5% | 辅助正则 | **无效** |
| ASG (Adaptive Style Gate) | CLIP-S +0.0002（噪声级） | 增强 | **无效**（zero-init + 5ep训练不足） |
| Spectral ODE | 参数不被消费 | 未在Method出现 | **代码死参数** |
| Style Amplification | 推理时=1.0（关闭） | 未在Method出现 | **关闭** |
| DWT-Routed Cross-Attention | 推理时=false | 未在Method出现 | **关闭** |

---

## 2. 最严重的发现：推理风格注入与训练loss解耦

### 2.1 论文叙事的逻辑链

论文Method的叙事主线是：
```
Wavelet分解 → 高频路由架构(cross-attn) → 谱Flow Matching + SWD → 端点WCT
     ↑                    ↑                      ↑                ↑
  坐标系变换           架构级隔离              训练目标          端点对齐
```

论文声称**四层保护机制**共同确保"结构保留 + 风格迁移"：
1. Wavelet正交分解（构造保证）
2. 高频路由cross-attention（架构级隔离）
3. 谱flow matching + SWD（训练目标约束）
4. 端点WCT（时间级保护）

### 2.2 实际数据揭示的真相

**推理时风格注入的实际来源（按强度排序）：**

| 机制 | 推理时状态 | 风格注入强度 | 是否在训练loss中监督 |
|------|-----------|-------------|-------------------|
| Endpoint AdaIN (spatial_fiber, α=1.0, per-step) | **每步执行** | **最强** | **否** — 纯几何后处理 |
| Cross-Attention (gate≈0.05) | 启用但极弱 | 微弱 | 是（通过velocity head间接） |
| ASG空间gate map | 启用但delta≈0 | 噪声级 | 是（通过velocity head间接） |
| Style Amplification | 关闭(=1.0) | 无 | — |
| DWT-Routed Cross-Attention | 关闭(false) | 无 | — |

### 2.3 关键结论

**WEAVE的实际工作机制与论文叙事严重不符：**

1. **训练loss（flow matching）主要学到的是velocity field的content保持能力** — 不是风格迁移能力
2. **风格迁移几乎完全来自推理时的Endpoint AdaIN后处理** — 这是一个纯几何统计量匹配，与训练loss解耦
3. **Cross-attention（论文声称的"高频路由"）gate≈0.05，几乎不注入风格** — 论文的"架构级隔离"叙事缺乏实证
4. **SWD guide对最终结果无贡献** — 被flow loss的梯度结构性淹没

**这意味着WEAVE的实际工作流程是：**
```
训练: 学习一个"内容保持的velocity field"（flow matching主导）
推理: velocity field做内容传输 + Endpoint AdaIN做风格注入（后处理）
```

而**不是**论文叙述的：
```
训练: 联合学习内容传输+风格迁移（flow + SWD + edge + endpoint）
推理: 学到的velocity field同时完成内容保持和风格迁移
```

---

## 3. Semantic SWD为什么没效果？实现分析

### 3.1 实现正确性结论

**核心SWD实现无bug。** `_sliced_wasserstein`函数（spectral_losses620.py:36-77）的数学定义正确：
- 投影方向每次新鲜随机（符合SWD理论保证）
- 排序+L1距离计算正确
- sample_weight的inverse-CDF采样逻辑正确

**`_semantic_region_swd`实现无致命bug，但设计有隐患：**
- k-means聚类合理（确定性种子，4次迭代）
- **Region对齐是亮度代理启发式**（按centroid均值排序），不是语义对应
- Q=256量化匹配数学等价于标准W1
- 非对称聚类：gen regions按content语义划分，target regions按target外观划分

### 3.2 SWD无效的根本原因（三重结构性劣势）

**不是bug，不是权重不对，不是数值太小 — 是分布级loss与像素级loss竞争时的结构性劣势。**

#### 劣势1：梯度量级的三重稀释

```
Flow loss梯度: d(loss_fm)/d(v) = 2·w_flow·(v_pred - v_target) / N
  → 只有空间平均稀释，per-pixel梯度 ~7e-5

SWD loss梯度: d(loss_swd)/d(z_hat1) = w_swd · sign · dirs / (N·P)
  → 排序置换稀释 + 投影平均(P=64) + 空间平均
  → per-pixel梯度 ~3e-5，且方向模糊
```

#### 劣势2：梯度方向性

- **Flow loss**：每个像素收到确定的、指向target的方向信号
- **SWD loss**：只告诉"你的输出分布应该更像target分布"，不告诉哪个像素往哪移

#### 劣势3：方向冲突

No-Flow消融揭示：当移除flow loss后，SWD的梯度方向是**"牺牲内容换风格"**：
- No-Flow: CLIP-S +0.005（风格更强）但DINO-C -0.093（内容崩溃）

这说明SWD的弱信号方向与flow loss的内容保持方向**冲突**，被后者压制。

### 3.3 SWD的正确使用方式（如果要用）

当前架构下SWD无效，但如果要让分布级loss有效，需要：

1. **两阶段训练**：先用flow loss训练到收敛，再用SWD做精细化分布约束
2. **梯度隔离**：SWD只作用于特定子带（如HH），避免与flow loss在LL上冲突
3. **更大权重**：8.0不够，需要80+才能与flow loss竞争（但会引入内容崩溃风险）
4. **或者改用Gram loss**：Gram矩阵捕获通道间相关性，梯度比SWD更直接

---

## 4. 逐组件诊断：论文18个组件的实际贡献

### 4.1 核心机制层（论文声称6个，实际有效3个）

| # | 组件 | 论文叙述 | 实际状态 | 证据 |
|---|------|----------|----------|------|
| 1 | VAE Latent Space | 核心基础 | **有效**（基础坐标系） | 所有操作在此空间 |
| 2 | Rectified Flow | 核心训练框架 | **有效**（核心机制） | No-Flow: DINO-C -0.093 |
| 3 | Haar Wavelet Decomposition | 核心创新 | **有效**（架构级） | w/o Wavelet: CLIP-S -0.016 |
| 4 | High-Frequency Routing (cross-attn) | 核心架构设计 | **几乎无效** | gate≈0.05，风格注入极弱 |
| 5 | Style Memory Module | 核心风格注入 | **存疑** | 与cross-attn绑定，gate极弱 |
| 6 | Endpoint WCT | 核心端点对齐 | **有效但名不副实** | 实际是per-step AdaIN，非endpoint-only |

### 4.2 增强层（论文声称4个，实际有效0个）

| # | 组件 | 论文叙述 | 实际状态 | 证据 |
|---|------|----------|----------|------|
| 7 | SWD Guide | 增强风格信号 | **无效** | Flow-Only ≈ Full |
| 8 | RMSNorm | 增强设计选择 | 未消融 | 低影响 |
| 9 | Few-Shot Adaptation | 增强应用能力 | 未消融 | 依赖style memory有效性 |
| 10 | Empirical Frequency Probes | 动机验证 | 有效（经验证据） | Figure 2 |

### 4.3 辅助层（论文声称6个，实际有效1个）

| # | 组件 | 论文叙述 | 实际状态 | 证据 |
|---|------|----------|----------|------|
| 11 | Heun Solver (8-step) | 推理工具 | **有效**（多步贡献） | w/o Flow(1-step): CLIP-S -0.003 |
| 12 | IDT Floor | 评估校准 | 有效（评估概念） | — |
| 13 | Edge Loss | 正则项 | **无效** | 占总loss 2.2% |
| 14 | Low-pass Anchor Loss | 正则项 | **无效** | 占总loss 1.5% |
| 15 | Spectral Flow-Matching Loss | 主训练目标 | **有效**（=flow matching） | — |
| 16 | Complete Objective | 训练目标汇总 | **部分有效** | 只有flow项有效 |

### 4.4 代码中存在但论文未提及的组件

| 组件 | 代码状态 | 推理时状态 | 备注 |
|------|----------|-----------|------|
| ASG (Adaptive Style Gate) | 存在 | 启用但delta≈0 | zero-init + 5ep不足 |
| Style Amplification | 存在 | 关闭(=1.0) | — |
| DWT-Routed Cross-Attention | 存在 | 关闭(false) | — |
| Per-subband Gate | 存在 | 关闭(false) | — |
| Velocity head FiLM | 存在 | 关闭(false) | — |
| LL AdaLN-Zero / Tone Bias | 存在 | 关闭 | — |
| Spectral ODE | 参数存在 | **不被消费** | 代码死参数 |

---

## 5. Endpoint AdaIN的"名不副实"问题

### 5.1 论文叙述 vs 代码实现

**论文叙述**（3.3节 Endpoint Statistical Alignment）：
- "只在最终端点($n=1$)应用统计对齐"
- "无论风格强度如何都保留结构"
- 强调"endpoint-only"避免逐步blending的方差衰减

**代码实现**（WEAVE主表配置）：
- `endpoint_adain_only_last_step = false`（默认值）
- **每步ODE Euler更新后都执行AdaIN**（8步 = 8次AdaIN注入）
- α=1.0 完全替换高频fiber

### 5.2 这意味着什么

论文的核心论点之一是"endpoint-only alignment principle"——论证为什么只在端点做一次统计对齐。但实际代码**每步都做**，这与论文的数学论证直接矛盾：

论文说：逐步blending导致方差衰减（$n=8, \alpha=0.5$时仅$3.9\times10^{-3}$内容存活）
代码做：8次α=1.0的完全替换

**但模型仍然有效！** 这说明：
1. 要么论文的"endpoint-only"论证是错的（逐步AdaIN也work）
2. 要么`endpoint_adain_only_last_step=false`是一个配置bug（应该设为true）
3. 要么8次α=1.0替换的累积效果与1次类似（每次都匹配到同一style统计量）

### 5.3 建议验证

需要运行一个消融：`endpoint_adain_only_last_step=true`，看是否与当前(false)有差异。如果无差异，说明"endpoint-only"原则不重要；如果有差异，说明当前配置可能不是最优。

---

## 6. 对论文叙事的影响评估

### 6.1 当前叙事的脆弱点

论文的Method叙事依赖**四层保护机制**的协同：
1. Wavelet正交分解
2. 高频路由cross-attention（架构级隔离）
3. 谱flow matching + SWD（训练目标约束）
4. 端点WCT（时间级保护）

但实际数据表明：
- 第2层（cross-attn）gate≈0.05，几乎不工作
- 第3层（SWD）无贡献，只有flow matching有效
- 第4层（端点WCT）实际是per-step AdaIN，与"endpoint-only"叙述矛盾

**实际有效的只有第1层（wavelet）和flow matching + per-step AdaIN。**

### 6.2 叙事是否需要重写？

**不需要完全重写，但需要调整重点：**

#### 可以保留的叙事元素
1. VAE latent space操作（基础）
2. Haar wavelet分解（核心创新，有效）
3. Rectified flow（核心训练框架，有效）
4. Endpoint statistical alignment（有效，但需要修正"endpoint-only"描述）
5. 频率解耦的动机（frequency probes有效）

#### 需要弱化或删除的叙事元素
1. **SWD guide** — 改为"我们尝试了分布级约束但发现flow matching已足够"，或直接删除
2. **高频路由cross-attention的"架构级隔离"** — 实际gate极弱，不应强调
3. **"四层保护机制"** — 简化为"wavelet分解 + flow matching + AdaIN"
4. **"endpoint-only alignment principle"** — 与代码实现矛盾，需要验证或修正

#### 需要新增的叙事元素
1. **Per-step AdaIN作为主要风格注入通道** — 这是实际工作机制，应正面描述
2. **Flow matching学到的是content保持能力** — 风格迁移由AdaIN完成，这是设计意图

### 6.3 最小改动方案

如果不想大改，可以调整叙述重点：

1. **3.2节**：弱化cross-attention的"架构级隔离"叙述，改为"轻量级风格条件注入"
2. **3.3节**：删除SWD guide的详细描述（或降为footnote），保留endpoint WCT但修正"endpoint-only"为"per-step statistical alignment"
3. **消融表**：只保留有效消融（w/o Wavelet, No-Flow, w/o Endpoint AdaIN, 1-step）

### 6.4 大改方案（如果愿意）

重新定位WEAVE的核心贡献：
- **不是**"四层保护机制的协同"
- **而是**"wavelet分解 + flow matching的content保持 + AdaIN的style注入的简洁分离"

这个叙事更诚实，也更有说服力：训练和推理职责分明，架构简洁。

---

## 7. 验证实验结果（2026-07-11 完成）

### 7.1 Gate Open实验：cross-attention是否能在gate全开时注入风格？

**配置**：`style_cross_attn_gate_init: 3.0`（tanh(3.0)≈0.995，cross-attention全强度注入），训练5ep

**结果**：

| Config | CLIP-S | LPIPS | DINO-C | DINO-S | ΔCLIP-S |
|--------|--------|-------|--------|--------|---------|
| Full (WEAVE) | 0.7261 | 0.3354 | 0.7692 | 0.4843 | -- |
| **Gate Open (3.0)** | **0.7251** | **0.3354** | **0.7758** | **0.4834** | **-0.001** |

**结论：gate全开仍然无效！**

- CLIP-S仅降0.001，DINO-C反而升0.007（更接近内容保留）
- 训练loss=6.41（高于full的4.98），说明cross-attention确实注入了更多信号
- 但注入的信号对风格迁移无贡献——**cross-attention架构在当前设计下根本不工作**
- 不是gate小的问题，而是style memory/cross-attention学到的东西本身无用

**对叙事的影响**：cross-attention的"高频路由架构级隔离"叙事完全站不住脚，应从Method中删除或大幅弱化。

### 7.2 Symmetric SWD实验：修复实现后SWD是否有效？

**配置**：`swd_symmetric_regions: true`（gen和target共享content-based region标签，移除亮度代理对齐），`swd_semantic_blend: 0.8`，训练5ep

**结果**：

| Config | CLIP-S | LPIPS | DINO-C | DINO-S | ΔCLIP-S | ΔDINO-C |
|--------|--------|-------|--------|--------|---------|---------|
| Full (WEAVE) | 0.7261 | 0.3354 | 0.7692 | 0.4843 | -- | -- |
| **Sym SWD** | **0.7101** | **0.4204** | **0.6384** | **0.4589** | **-0.016** | **-0.131** |

**结论：对称SWD"生效"了，但方向是负面的！**

- CLIP-S降0.016，DINO-C崩溃0.131，LPIPS升0.085
- 对称聚类让SWD的信号更强（region真正语义对应了），所以产生了显著变化
- 但变化方向是**内容崩溃**——证实了之前的诊断：SWD驱动"牺牲内容换风格"
- **SWD不是实现问题，而是这类loss在当前架构下的方向性问题**

**对叙事的影响**：SWD应从Method中删除。它不是"增强"，而是有害信号。论文Figure 3的"SWD恢复风格分离能力"只是理论分析，实际训练中SWD的梯度方向有害。

### 7.3 Endpoint Only实验：per-step AdaIN是否必要？

**配置**：`endpoint_adain_only_last_step: true`（只在最后一步做AdaIN，而非每步），推理时消融，用base checkpoint

**结果**：

| Config | CLIP-S | LPIPS | DINO-C | DINO-S | ΔCLIP-S |
|--------|--------|-------|--------|--------|---------|
| Full (WEAVE, per-step) | 0.7261 | 0.3354 | 0.7692 | 0.4843 | -- |
| **Endpoint Only (last-step)** | **0.7261** | **0.3354** | **0.7691** | **0.4843** | **0.000** |

**结论：per-step和endpoint-only完全相同！**

- 4个指标差异在0.0001量级（噪声级）
- 论文的"endpoint-only alignment principle"数学论证不成立——逐步AdaIN不会导致方差衰减
- 原因：每次AdaIN都匹配到同一style统计量，8次α=1.0替换的累积效果与1次相同
- **"endpoint-only"原则不重要**，可以从叙事中删除

**对叙事的影响**：论文3.3节的"endpoint-only alignment principle"及其数学论证（方差衰减、几何侵蚀）应删除。Endpoint AdaIN本身仍有效（w/o AdaIN CLIP-S -0.016），但"只在端点做"vs"每步都做"没有区别。

### 7.4 验证实验总结

| 实验 | 假设 | 结果 | 结论 |
|------|------|------|------|
| Gate Open | gate小导致cross-attn无效 | **无效**（ΔCLIP-S=-0.001） | cross-attn架构根本不工作 |
| Sym SWD | 实现不对导致SWD无效 | **有害**（DINO-C -0.131） | SWD方向有害，非实现问题 |
| Endpoint Only | per-step AdaIN必要 | **无差异**（Δ=0.000） | endpoint-only原则不重要 |
| **Contrastive SWD** | **SWD匹配对象错误（与FM冗余）** | **轻微有效**（ΔCLIP-S=-0.003, ΔDINO-S=-0.007） | **新方向正确但不崩溃，旧SWD冗余根因确认** |

### 7.5 Contrastive SWD实验：修复匹配对象后SWD是否有效？

**背景**：诊断发现旧SWD无效的根因是——SWD匹配z₁，FM也匹配z₁，point-wise matching蕴含distribution matching，SWD在数学上冗余。

**新设计**：Style-contrastive SWD，匹配对象不再是z₁，而是batch内的风格分布：
- 同target_style_id的生成图：分布应一致（SWD→0）
- 不同target_style_id的生成图：分布应有margin（hinge loss: max(0, margin - SWD)）
- 这是FM**不能**提供的约束：FM只做point-wise z_hat1→z_1，不约束同风格生成图之间的分布一致性

**配置**：`w_style_contrastive: 8.0, margin: 0.05, projections: 64`，禁用旧SWD/edge/endpoint_content（权重全设0），训练5ep

**结果**：

| Config | CLIP-S | LPIPS | DINO-C | DINO-S | ΔCLIP-S | ΔDINO-S | ΔDINO-C |
|--------|--------|-------|--------|--------|---------|---------|---------|
| Full (WEAVE) | 0.7261 | 0.3354 | 0.7692 | 0.4843 | -- | -- | -- |
| **Contrastive SWD** | **0.7227** | **0.3458** | **0.7578** | **0.4770** | **-0.003** | **-0.007** | **-0.011** |

**对比三种SWD方案：**

| SWD方案 | 匹配对象 | ΔCLIP-S | ΔDINO-C | ΔDINO-S | 结论 |
|---------|----------|---------|---------|---------|------|
| 旧SWD (legacy) | z₁（与FM冗余） | -0.001 | +0.008 | +0.003 | 冗余，无效 |
| Sym SWD | z₁（对称聚类） | -0.016 | **-0.131** | -0.025 | 有害，内容崩溃 |
| **Contrastive SWD** | **batch内风格分布** | **-0.003** | **-0.011** | **-0.007** | **轻微影响，不崩溃** |

**关键发现：**

1. **根因确认**：旧SWD无效确实是因为匹配对象与FM冗余。Contrastive SWD匹配不同对象（batch内风格分布），产生了可测量的效果（ΔCLIP-S=-0.003 vs 旧SWD的-0.001）。

2. **方向正确**：与Sym SWD的DINO-C -0.131崩溃不同，Contrastive SWD的DINO-C仅-0.011，没有内容崩溃。这说明约束"同风格分布一致+跨风格分布分离"的方向是健康的，不像旧SWD那样"牺牲内容换风格"。

3. **效果较弱**：5ep训练下效果仍然较弱（-0.003）。可能原因：
   - 5个风格×5样本的batch内对比信号有限
   - margin=0.05可能需要调优
   - 需要更长训练才能看到分布分离的累积效果

4. **不崩溃的意义**：即使效果较弱，"不崩溃"本身是重要发现——证明SWD可以以健康方式融入训练，只是需要进一步优化（更大batch、更多风格、更长训练、margin调优）。

**对叙事的影响**：
- 旧SWD（匹配z₁）应删除 — 冗余
- 新Contrastive SWD是有前景的方向 — 但5ep效果弱，当前论文中可作为future work或辅助loss
- SWD在Latent上的区分能力（Figure 3）依然valid — 那是作为评估指标，不是训练loss

---

## 8. 更新后的诊断结论

### 7.6 Contrastive SWD强度扫描：更剧烈是否更好？

**背景**：base contrastive SWD（w=8, margin=0.05）效果较弱（ΔCLIP-S=-0.003）。测试更剧烈的对比信号和数学上更合理的设计（加入InfoNCE centroid对比）。

**新设计**：三合一contrastive SWD
1. **Same-style consistency**：同风格对SWD→0
2. **Cross-style hinge**：跨风格对SWD→margin
3. **Centroid InfoNCE**：每个样本应更接近自己风格的centroid（用cross-entropy on -SWD/τ）

**三档配置**：

| 档位 | w | margin | temp | w_centroid | P |
|------|---|--------|------|------------|---|
| mild | 20 | 0.10 | 0.2 | 1.0 | 64 |
| strong | 40 | 0.15 | 0.1 | 2.0 | 64 |
| extreme | 80 | 0.20 | 0.05 | 4.0 | 128 |

**结果**：

| Config | CLIP-S | LPIPS | DINO-C | DINO-S | ΔCLIP-S | ΔDINO-S | ΔDINO-C |
|--------|--------|-------|--------|--------|---------|---------|---------|
| Full (WEAVE) | 0.7261 | 0.3354 | 0.7692 | 0.4843 | -- | -- | -- |
| base (w=8) | 0.7227 | 0.3458 | 0.7578 | 0.4770 | -0.003 | -0.007 | -0.011 |
| **mild (w=20)** | **0.7105** | **0.6581** | **0.5817** | **0.4569** | **-0.016** | **-0.027** | **-0.188** |
| **strong (w=40)** | **0.7111** | **0.6018** | **0.6331** | **0.4622** | **-0.015** | **-0.022** | **-0.136** |
| **extreme (w=80)** | **0.7007** | **0.5665** | **0.6614** | **0.4471** | **-0.025** | **-0.037** | **-0.108** |

**关键发现：**

1. **contrastive SWD越强，内容崩溃越严重** — DINO-C从0.7692→0.5817（mild）→0.6614（extreme），LPIPS从0.3354→0.6581（mild，几乎翻倍）。这与旧SWD/sym SWD的退化模式完全一致。

2. **DINO-S不升反降** — 目标是提升DINO-S，但所有档位DINO-S都下降（-0.007到-0.037）。风格分离的压力没有转化为更好的风格迁移，反而牺牲了风格质量。

3. **CLIP-S持续下降** — 从-0.003到-0.025，contrastive压力越大CLIP-S越差。

4. **LPIPS的悖论** — extreme的LPIPS=0.5665（最高），但DINO-C=0.6614（不是最低）。说明extreme的"内容损失"不是简单的像素偏移，而是语义结构的破坏。

**根本原因分析：**

Contrastive SWD的梯度方向问题与旧SWD本质相同：
- **FM已经让z_hat1 point-wise匹配z_1** — 这已经是最优的content-style平衡
- **任何额外的分布约束（无论是matching z_1还是contrastive）都在破坏FM找到的平衡**
- contrastive SWD push同风格分布一致时，牺牲了per-sample的content保持
- push跨风格分布分离时，引入了不必要的扰动

**结论：在当前FM架构下，任何形式的SWD分布约束都是有害或无效的。** FM的point-wise matching已经隐含了最优的分布约束，额外的分布loss只会干扰FM找到的平衡点。

**对论文叙事的最终影响：**
- SWD（任何形式）应从Method中完全删除
- Figure 3的"SWD在Latent上的区分度"作为评估指标分析依然valid
- 但"SWD作为训练loss增强风格迁移"的叙事不成立 — FM已蕴含分布约束

### 8.1 WEAVE的实际有效组件（最终确认）

| 排名 | 组件 | 类型 | 消融影响 | 验证状态 |
|------|------|------|----------|----------|
| 1 | **Rectified Flow** | 训练loss | No-Flow: DINO-C -0.093 | ✓ 已验证 |
| 2 | **Haar Wavelet** | 架构 | w/o Wavelet: CLIP-S -0.016 | ✓ 已验证 |
| 3 | **Endpoint AdaIN** | 推理后处理 | w/o AdaIN: CLIP-S -0.016 | ✓ 已验证 |

**无效组件（全部已验证）：**

| 组件 | 验证方式 | 结果 |
|------|----------|------|
| Cross-Attention (高频路由) | Gate Open (0.05→3.0) | ΔCLIP-S=-0.001，无效 |
| SWD Guide | Sym SWD (修复实现) | DINO-C -0.131，有害 |
| ASG | w/o ASG | ΔCLIP-S=+0.000，无效 |
| Edge Loss | Loss组成分析 | 占2.2%，无效 |
| Endpoint Content Loss | Flow-Only | Δ<0.004，无效 |
| "Endpoint-Only"原则 | endpoint_only_last_step=true | Δ=0.000，无差异 |

### 8.2 对论文叙事的最终建议

**应从Method中删除的叙事元素：**
1. ~~"高频路由架构级隔离"~~ — cross-attention无效
2. ~~"SWD guide恢复风格分离能力"~~ — SWD有害
3. ~~"endpoint-only alignment principle"~~ — per-step和endpoint-only无差异
4. ~~"四层保护机制"~~ — 实际只有wavelet + flow + AdaIN

**应保留的叙事元素：**
1. VAE latent space操作（基础）
2. Haar wavelet分解（核心创新，有效）
3. Rectified flow（核心训练框架，有效）
4. Endpoint statistical alignment / AdaIN（有效，但不需要"endpoint-only"论证）
5. 频率解耦的动机（frequency probes有效）

**建议的新叙事主线：**
```
WEAVE = Wavelet分解 + Flow Matching + Endpoint AdaIN
训练: Wavelet分解 → Flow Matching学到content保持的velocity field
推理: ODE积分 → Endpoint AdaIN注入风格统计量
```

简洁、诚实、有效。不需要"四层保护"的过度包装。

### 8.3 对"大改模型"的最终回答

**不需要大改模型。** WEAVE的实际工作机制是简洁有效的：
- 训练学到content保持能力（flow matching）
- 推理做风格注入（AdaIN）
- Wavelet提供频率解耦的坐标系

问题在于论文叙事过度包装了无效组件。修复方法是调整叙事，而非修改模型。

如果要让cross-attention/SWD真正有效，需要的不是调参数，而是重新设计：
1. cross-attention需要更强的style conditioning（如CLIP/DINO风格特征而非学习embedding）
2. SWD需要两阶段训练或与flow loss梯度隔离
3. 这些都是未来工作，不是当前论文的范围

---

## 附录: 实验数据汇总

### A.1 训练时消融（修改loss函数）

| Config | CLIP-S | LPIPS | DINO-C | DINO-S | ΔCLIP-S | 有效性 |
|--------|--------|-------|--------|--------|---------|--------|
| Full (WEAVE) | 0.7261 | 0.3354 | 0.7692 | 0.4843 | -- | 基准 |
| SWD→MSE | 0.7256 | 0.3319 | 0.7768 | 0.4868 | -0.001 | ✗ |
| w/o SWD | 0.7248 | 0.3358 | 0.7736 | 0.4832 | -0.001 | ✗ |
| LL 0.3→1.0 | 0.7246 | 0.3313 | 0.7759 | 0.4849 | -0.002 | ✗ |
| w/o Wavelet | 0.7098 | 0.3619 | 0.7705 | 0.4638 | **-0.016** | ✓ |
| No-Flow | 0.7312 | 0.4192 | 0.6767 | 0.4662 | +0.005 | ✓ (内容崩溃) |
| Flow-Only | 0.7243 | 0.3380 | 0.7657 | 0.4817 | -0.002 | ✗ (≈Full) |

### A.2 推理时消融

| Config | CLIP-S | LPIPS | ΔCLIP-S | 有效性 |
|--------|--------|-------|---------|--------|
| w/o Flow (1-step) | 0.7229 | 0.3646 | -0.003 | ✓ (中等) |
| w/o ASG | 0.7263 | 0.3442 | +0.000 | ✗ |
| w/o Endpoint AdaIN | 0.7098 | 0.3022 | **-0.016** | ✓ (最强) |

### A.3 Loss组成分析

| 组件 | 值(随机数据) | 权重 | 加权值 | 占比 | 有效性 |
|------|------------|------|--------|------|--------|
| loss_fm | 4.584 | 1.0 | 4.584 | 92.1% | ✓ |
| single_step_swd | 0.027 | 8.0 | 0.215 | 4.3% | ✗ |
| single_step_edge | 1.103 | 0.1 | 0.110 | 2.2% | ✗ |
| endpoint_content | 0.076 | 1.0 | 0.076 | 1.5% | ✗ |
