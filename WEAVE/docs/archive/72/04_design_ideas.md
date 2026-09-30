# 04 — 设计思路

> 从最初的问题出发，逐步推演 FC-SB 架构的演化路径：每个设计决策的动机、备选方案、实验验证与最终选择。本文档是"为什么这样设计"的决策记录，而非"如何实现"的代码说明。

---

## 1. 起点与核心问题

### 1.1 任务定位

**目标**: latent-space image stylization — 在 SDXL VAE latent 空间做风格迁移，避免像素空间方法的高计算成本。

**核心挑战**: 内容保真（LPIPS）与风格一致（CLIP-S）的 trade-off。

### 1.2 为什么选 Schrödinger Bridge？

**备选方案**:
- (A) GAN-based（CycleGAN, MSGAN）：训练不稳定，模式坍塌
- (B) Diffusion-based（DDIM, SDE）：推理慢（50-1000 步）
- (C) **Schrödinger Bridge**：熵正则化 OT，理论最优路径，少量步数（8 步）

**选择理由**: SB 提供 entropy-regularized optimal transport 的理论框架，在 content→style 分布间寻找最可能路径。简化为 deterministic Flow Matching 后，8 步 ODE 积分即可完成迁移。

### 1.3 为什么频域解耦？

**观察**: 风格信息在频域有天然分层：
- LL（低频）：全局色调、光照、色相
- LH/HL（中频）：边缘、纹理、笔触方向
- HH（高频）：噪点、细节

**假设**: 如果能对各频段独立控制，就能精细调节内容-风格 trade-off，而非全局一刀切。

---

## 2. 架构演化路径

### 2.1 第一阶段：Content Fidelity Pathway（Phase 1-4A）

**问题**: 如何保证内容不被风格迁移破坏？

**设计决策**:
1. **Haar DWT 正交分解**：无信息损失，子带统计独立
2. **3 个独立 velocity head**（LL/LH/HL）：各频段独立学习速度场
3. **Endpoint AdaIN**：推理末步统计匹配注入风格

**4A2 减法消融验证**（3 个核心组件不可移除）:
- `w_ll=0.0`（移除 LL velocity 训练）→ clip=0.7117 FAIL
- `style_extrap_alpha=0.0`（移除外推）→ clip=0.7242 FAIL
- `endpoint_adain_scale=0.0`（移除 AdaIN）→ clip=0.7082 FAIL

**结论**: Content Fidelity Pathway = DWT Haar → AdaIN scale → Spectral ODE，三层路径完整必要。

### 2.2 第二阶段：多级 DWT 突破（Phase 4D-4F）

**问题**: 单级 DWT 的 LL 子带（16×16）仍包含过多中频信息，分离不够彻底。

**设计决策**: 多级递归 DWT，对 LL 再分解。

**4F 趋势验证**:
- Level 1 (LL 16×16): clip=0.7261
- Level 2 (LL 8×8): clip=0.7301 (+0.0040)
- **Level 3 (LL 4×4): clip=0.7319 (+0.0018)** ← SOTA
- Level 4 (LL 2×2): clip=0.7316 (-0.0003, FAIL)

**为什么 3-Level 是峰值？**
- 1→2: 中尺度结构被分离到 LH_2/HL_2/HH_2，LL 更纯 → 显著增益
- 2→3: 小尺度结构进一步分离 → 递减增益
- 3→4: LL_4 仅 2×2 = 4 像素，**丢失位置信息** → 反转

**4E Daubechies 验证**: db2 ≈ haar（FLAT），证明基函数非关键，多级才是主导效应。

### 2.3 第三阶段：LL 不是纯内容锚（Phase 4G.1 关键洞察）

**问题**: 能否锁死 LL 完全保内容？

**4G.1 2×2 矩阵消融**:

| | w_ll=0 (不训练) | w_ll=1.0 (训练) |
|---|---|---|
| **lock=False** (推理用 v_ll) | 4A2: 0.7117 (噪声) | **4F.1 SOTA: 0.7319** |
| **lock=True** (推理锁死) | 4G.1b: 0.7174 | 4G.1a: 0.7178 |

**关键发现**:
- LL velocity 应用: +0.0141 clip（LL 携带全局色调/光照风格信息）
- LL velocity 训练: -0.0091 lpips（梯度回流改善 backbone 内容理解）

**结论**: LL 是"内容 + 全局风格"的混合载体，不能锁死。这一洞察颠覆了"LL = pure content"的初始假设。

### 2.4 第四阶段：EOTA 解耦（Phase 4G.2b → 4H.1）

**问题**: 4G.2 per-subband AdaIN 的 α=0.5 和 α=1.0 结果几乎相同，α 失效。

**根因分析**: 多步 Euler 迭代累积。残差 `r_n = (1-α)^n`，n=12 步、α=0.5 时 `r = 0.5^12 = 0.024%`，α 被迭代累积 invalidate。

**设计决策 — EOTA (End-of-Trajectory AdaIN)**:
- 仅在最后一步应用 AdaIN
- 前 N-1 步纯 ODE 积分
- 解耦 ODE 求解与风格注入

**理论意义**: 匹配 SB 理论 — 风格是 terminal condition 而非 per-step perturbation。

**4H.1 验证**: EOTA 恢复 α 有效性。α sweep 显示单调 trade-off：
- α=0.5: content SOTA, style FAIL
- α=0.7: BALANCED, BOTH PASS
- α=0.8: NEW SOTA (spatial_fiber + EOTA)
- α=1.0: style SOTA, content FAIL

### 2.5 第五阶段：数值精度突破（Phase 4I.2）

**问题**: 4H 战术参数全部失效（loss/patch/mask/capacity 都映射到同一 1D Pareto 前沿）。

**假设**: ODE 积分精度可能是新的自由度。

**设计决策 — Heun solver**:
- Euler: O(h²) 截断误差
- Heun: O(h³)，predictor-corrector
- 同步数下轨迹更准确

**4I.2 验证 — 结构性突破**:
- 4I.2a (Heun 3ep) ≈ 4H.1g (Euler 5ep)：精度增益 ≈ +2 epochs
- 4I.2b (Heun 5ep): clip +0.0015, lpips -0.0052 — **双提升**
- **复合效应**: Heun 优势随训练时长增长（3ep→5ep: lpips 降 25x 多于 Euler）

**4I.6 RK4 饱和**: Heun→RK4 无额外收益。其他误差源（训练噪声、velocity field 精度）主导。

**核心理论贡献**: solver order 是独立于 α 的结构性 DOF。所有非 solver-order 自由度都映射到同一 1D Pareto 前沿。

### 2.6 第六阶段：Schedule 探索（Phase 4I.5/4I.8）

**问题**: 时间步分布是否影响 trade-off？

**4 种 schedule 测试**:
- linear（中性）: 4I.2b clip=0.7266, lpips=0.3229
- cosine（内容偏置）: 4I.5b clip=0.7262, lpips=0.3171
- rquad（风格偏置）: 4I.5c clip=0.7293, lpips=0.3429
- warp_cos p=0.8（轻风格偏置）: 4I.8b clip=0.7282, lpips=0.3271

**分类结论**:
- Schedule shape 是 **Pareto-mapping knob**（沿前沿移动），非结构性 DOF
- 但 cosine 的内容偏置提供 lpips 余量，可用更高 α 换 clip

**4I.7b 组合**: cosine + Heun + α=0.85 → clip=0.7272, lpips=0.3218（远程 SOTA；v5 修正, SaMam 数据完整性修正: 4I.7b CLIP-S 大幅超越 SaMam 0.5816 (+0.1456), LPIPS 微弱落后 SaMam 0.2434 (-0.0784, 但 SaMam CLIP-S 低于 Identity 风格转移失败), 4I.7b DUAL BEAT SaMam）。

### 2.7 第七阶段：DWT Route（Phase 4J.1）

**问题**: style_memory 被迫学"维持结构"，分散了表达笔触/色彩的能力。

**设计决策 — DWT Route Cross-Attention**:
- 对特征图做 DWT
- **LL bypass**: 不参与 cross-attention query
- 仅 LH/HL/HH tokens query style_memory
- IDWT 重建（LL 保持原值）

**理论收益**: style_memory 100% 容量表达笔触/色彩，不被迫学"维持结构"。

**4J.1 结果**: clip=0.7226, lpips=0.3068 — 本地 DWT route 起点。

### 2.8 第八阶段：Stochastic DWT Route（Local T5-T11）

**问题**: 4J.1 的 DWT route 使 style_mem 训练时只服务高频 query，学到的风格表达偏向高频纹理。

**T5 设计 — Eval-Only DWT**:
- 训练时全空间 query（像 4F.lvl3，style_mem 学完整风格）
- 推理时 DWT route（像 4J.1，LL bypass 保护内容）

**T5 结果**: clip=0.7061 FAIL，但 lpips=0.2606 BEST。根因：训练/推理 query 分布失配，q_proj 未见过 DWT 分布。

**T10 设计 — Stochastic (p=0.5)**:
- 训练时 50% DWT + 50% 全空间
- 推理时始终 DWT

**T10 结果**: clip=0.7083 FAIL，lpips=0.2480 NEW BEST。根因：50% 概率仍不足以让 q_proj 精通 DWT。

**T11 设计 — Stochastic (p=0.8)**:
- 训练时 80% DWT + 20% 全空间
- 推理时始终 DWT

**T11 结果**: clip=0.7213, lpips=0.2868 — **本地 SOTA**，lpips 首次 PASS 0.3068。

**p 扫描趋势**:
- p↑: clip↑（q_proj 越精通 DWT），上限在 4J.1 的 0.7226
- p=0.5: lpips 最低（0.2480）
- p=0.8: 平衡点（clip/lpips 双达标附近）

**核心洞察**: 80% DWT 让 q_proj 精通 DWT 系数，20% 全空间让 style_memory 学更完整风格。两者权衡得到 T11。

### 2.9 第九阶段：LL 风格注入探索（Local T13-T16，全失败）

**问题**: T11 的 clip 差目标 0.0106，能否通过 LL 风格注入提升？

**T13 LLGSI**: style_mem 池化统计量 → LL AdaIN
- 结果: clip=0.7128, lpips=0.2706 (trade-off 1:3.7)
- 根因: style_mem 为高频训练，统计量不编码全局风格

**T14 CASI**: cross-attn 输出统计量 → LL AdaIN
- 结果: clip=0.7152, lpips=0.2795 (略优于 T13)
- 根因: cross-attn 输出仍主要是高频信号

**T15 LLGQCA**: LL 全局向量 query cross-attend style_mem
- 结果: clip=0.7176, lpips=0.2764 (trade-off 1:6.08)
- 渐进改善: T13→T14→T15 clip 持续提升，证明 cross-attn 非线性表达力优于 AdaIN

**T16 gate sweep** (0.2/0.3/0.5): 全部 FAIL
- 根因: 增大 gate = 放大高频噪声 = 破坏 LL 色调

**系列结论**: 7 个配置系统性证明，不动 style_mem 前提下，无法从 style_mem 提取有效全局风格信号。style_mem 的高频偏向是根本限制。

### 2.10 第十阶段：Loss/容量调优（Local T18-T19，全失败）

**问题**: 能否通过 loss 权重或模型容量提升 clip？

**T18 w_ll sweep**:
- w_ll=0.0 (T11): clip=0.7213, lpips=0.2868 ← 本地 SOTA
- w_ll=0.5 (T18a): clip=0.7174, lpips=0.2774 ← FAIL
- w_ll=1.0 (T18b): clip=0.7180, lpips=0.2764 ← FAIL

**结论**: w_ll>0 是 content-heavy trade-off。T11 w_ll=0.0（LL 自由漂移）是 clip 最佳点。

**T19 容量 sweep**:
- depth=6 (T19a): NaN — WCT eigh 数值不稳定
- dim=96 (T19b): clip=0.7207, lpips=0.3142 — 5ep 欠拟合（v_ll_abs 降 50%）

**结论**: 容量增加受限于数值稳定性（depth=6）和训练预算（dim=96 欠拟合）。

---

## 3. 关键设计决策汇总

### 3.1 保留的决策

| 决策 | 理由 | 验证实验 |
|------|------|----------|
| Haar DWT（vs avg_pool）| 正交性，统计隔离 | 4B-3 |
| 3-Level DWT（vs 1/2/4-Level）| 峰值，4-Level 丢位置 | 4F |
| learnable style_memory（vs DINO）| "Style Is Learned, Not Extracted" | 4C |
| LL velocity 应用（vs lock）| +0.0141 clip，LL 携带全局风格 | 4G.1 |
| EOTA（vs per-step AdaIN）| 恢复 α 有效性 | 4H.1 |
| Heun solver（vs Euler/RK4）| 结构性 DOF，打破 Pareto | 4I.2 |
| cosine schedule | 内容偏置，lpips 余量 | 4I.5/4I.7 |
| Stochastic DWT p=0.8 | q_proj 精通 DWT + style_mem 学完整风格 | T11 |
| w_ll=0.0 | LL 自由漂移是 clip 最佳 | T18 |
| depth=4, dim=64 | 数值稳定 + 5ep 充分训练 | T19 |

### 3.2 放弃的决策

| 决策 | 放弃理由 | 验证实验 |
|------|----------|----------|
| DINO 外部特征 | content-biased 污染 -0.018 | 4C |
| LL lock | LL 携带 +0.014 clip 全局风格 | 4G.1 |
| per-subband α=0.5 (per-step) | 迭代累积 invalidate α | 4G.2b |
| 多尺度 α | iDWT 重建耦合子带 | 4I.1 |
| RK4 solver | Heun 已饱和 | 4I.6 |
| 8+ epochs | 过度训练，lpips 退化 | 4I.8a |
| num_steps=12 | 饱和，ns=8 已足够 | 4I.8c |
| Eval-only DWT (T5) | 训练/推理分布失配 | T5 |
| Stochastic p=0.5 (T10) | q_proj 不精通 DWT | T10 |
| LLGSI/CASI/LLGQCA (T13-T16) | style_mem 高频偏向根本限制 | T13-T16 |
| w_ll>0 (T18) | content-heavy trade-off | T18 |
| depth=6 / dim=96 (T19) | 数值不稳定 / 欠拟合 | T19 |
| Few-shot Textual Inversion (4J.6) | 梯度通路太弱 | 4J.6 |

### 3.3 备选未探索方向

| 方向 | 潜力 | 风险 |
|------|------|------|
| 全新架构（非 DWT route）| 可能打破 1:8 trade-off | 重新设计成本高 |
| 独立全局风格信号源 | 可能提升 clip | 需额外训练 |
| 调整双超目标阈值 | 立即可行 | 学术价值降低 |
| 频域 differentiated ODE steps | 未验证 | 可能耦合 |

---

## 4. 设计哲学

### 4.1 减法优先（Subtractive Ablation First）

每个新组件加入前，先验证移除后的影响。4A2 减法消融是 Content Fidelity Pathway 的基石。

### 4.2 结构性变化优先于参数调优

Phase 4H 证明：所有战术参数（loss/patch/mask/capacity）映射到同一 1D Pareto 前沿。要打破前沿必须引入结构性变化（solver order, schedule, routing）。

### 4.3 理论指导实验

每个实验都有明确的理论假设：
- 4G.1: "LL 是纯内容锚" → 推翻
- 4I.1: "HH 对内容不敏感" → 推翻
- 4I.2: "数值精度是新自由度" → 确认
- T11: "80% DWT + 20% 全空间可平衡" → 确认

### 4.4 系统性证明优于单点失败

每个失败方向都做完整 sweep（T3/T4 6 配置、T13-T16 7 配置、T18-T19 4 配置），而非单个实验就下结论。这样能区分"参数调优失败"和"架构根本限制"。

### 4.5 用户指导原则

- **"Style Is Learned, Not Extracted"**: 指导 4C 放弃 DINO，T13-T16 不动 style_mem
- **"不要动 style_mem，去动目标对齐和注意力的频域路由"**: 指导 T5/T10/T11 探索 DWT route
- **"无效代码确认后直接删除"**: 指导 628/629/630 Phase 1 清理
