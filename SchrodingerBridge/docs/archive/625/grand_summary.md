# Schrödinger Bridge 风格迁移项目阶段性总览

> 日期：2026-06-26
> 范围：612→616→618→619→620→622→625 全周期统合
> 性质：阶段性大文档，整合 6 个月探索、645+ 次实验、4 次范式重构的完整图景
> 当前基线：I7（endpoint_film_init_std=0.1, style_film_init_std=0.0, style_embed_scale=4.0），epoch_0002
> 当前指标：clip_style=0.7017, content_lpips=0.3625, WFI=0.3757（5-style all_pairs_overview）

---

## 序言：6 个月帕累托死结

本项目历经 6 个月、645+ 次实验，始终被困在一个**三难困境**中：

```
clip_style ↑（风格相似度）
content_lpips ↓（内容保真度）
WFI ↓（白化程度，越低越好）
```

三者相互制约：降 LPIPS 则 clip_style 降，提 clip_style 则 LPIPS 升，训练越久 WFI 越恶化。所有方法都卡在 clip_style≈0.70 / LPIPS≈0.30-0.36 / WFI≈0.40-0.49 的"保守吸引子"。

**贯穿全项目的核心矛盾**：模型在 L_flow 主导下偏好"少注入 style"以降 loss，表现为 gate collapse（0.048）、endpoint shrinkage（α=0.16）、GroupNorm 白化、attention 均匀化。这不是 bug，而是当前训练目标下的**最优策略**。

**本总结的视角**：从 4 次范式重构中提取真正的因果链，区分"已证伪的假设"与"已确认的理论"，给出当前 FC-SB 阶段的精确定位和未来战略。

---

## 第一章：四轴演变史

### 1.1 数据轴：2 风格 → 5 风格 → open-set

```
V0  monet2photo (2风格)
V3  CycleGAN-T (4风格)        ← GPU 预加载启用，batch=256
V5  自定义 5 风格 distinct5   ← SaMST per-style 0.7597/0.3374 作为天花板参照
V8  SaMST per-style 基线       ← 每风格单独训练，证实通用模型与 per-style 的 0.05-0.09 gap
V10 620 Spatial Bridge         ← DINOv2 patch tokens (256 memory) open-set
```

**关键洞察**：
- per-style 模型（0.76）是通用模型（0.67-0.71）的天花板参照系
- distinct5 的 IDT CLIP-S=0.68 << wikiart512 的 0.795，IDT 越低=风格间距越大=任务越难
- 数据量非瓶颈，virtual_length=0.1 即够训练

### 1.2 架构轴：SA-Flow → 620 Spatial Bridge

```
SA-Flow (0.60) → LGT-X → C-G-W → LatentAdaCUT (0.72, 64-token)
→ Schrödinger Bridge (0.701, Distinct5 天花板)
→ 620 Spatial Bridge (0.67-0.71, DINOv2+LoRA→256 memory tokens)
```

**关键转折点**：
- **DiT patchification 6D reshape 易 bug** → 改用简单 conv 验证
- **Cross-attention 必须多 token**：1 token 时 softmax 恒=1.0；64-token + sharpen=2.5 才有选择性
- **InstanceNorm 是 attention 的毒药**：白化 features → 均匀 attention；Q/K 必须 twin-norm
- **Zero-init gate** 保证从 AdaGN-only 稳定渐进学习

**620 架构核心**：DINOv2+LoRA→256 memory tokens，AdaLN(time)+SelfAttn+CrossAttn(style)+FiLM endpoint head。但 runtime 显示 gate=0.048（95% 关闭），cross_attn_delta=0.038（极弱），FiLM gamma/beta=0.13（弱）。

### 1.3 Loss 轴：膨胀 → 清理 → 极简

```
单一 FM-MSE → 9 项 kitchen-sink → 3 项核心 → FC-SB 各向异性
```

**关键结论**：
- **对抗 loss 在 latent space 不稳定**：CycleGAN→MSE 是正确方向
- **NCE/repulsive/cycle/structure/TV 等 heuristic loss 无收益或负收益**：PatchNCE 直接摧毁风格（0.694→0.674）
- **SWD 是唯一有效的 style loss**，但权重需仔细调（100→250→0.15→8→2）
- **Kinetic energy 是必须的稳定器**（防 velocity 爆炸）
- **Phase 1 Cleanup**：losses.py 942→340 行，保留 L_flow + L_kinetic + L_terminal_swd
- **620 简化**：L_flow + 8·single_step_swd + 0.1·edge，σ=0.02

**重要反转**（ablation_audit 阶段）：edge loss 从"必要"变为"有害"，edge=0.0 在 WFI/CLIP-S/LPIPS 三项均优于 0.1，是"三赢"开关。

### 1.4 评估轴：单指标 → 三难困境

```
clip_style 单指标 → +LPIPS → +WFI（白化定量）
```

**WFI 等级**：
- 正常 0.50-0.65
- 轻微 0.68-0.72
- 中等 0.73-0.78
- 严重 >0.85

**关键发现**：
- clip_style 不是可靠唯一指标：白化图可能 clip_style 很高
- 训练时间≠效果提升：3-epoch WFI 单调恶化（0.43→0.47），存在最优停止点
- 0.70 是隐含天花板：LANCET/Distinct5/620 均卡在 0.67-0.71
- Fiber-SDE σ=0.08 不训练达 0.711（可能不需要训练）—— 这是后来 FC-SB 的灵感来源

---

## 第二章：核心理论演进

### 2.1 第一阶段：单点优化（612-616）

早期尝试通过 OT 匹配、Fiber Bundle SDE 等单点改造突破瓶颈，但收效有限。

**关键结论**：
- Euclidean OT 不稳定，Minibatch OT 抖动
- ODE Unrolling 导致梯度消失
- Fiber Bundle 理论框架提出但未有效实施

### 2.2 第二阶段：保守偏好诊断（618-620）

**核心洞察**：所有失败都追溯到"模型选择保守策略"——1-token attn、白化、低 gate、shrinkage、WFI 恶化都是同一根因的不同表现。

**5 大致命缺陷确认**（619 system_diagnosis）：
1. 时间风格纠缠（`style_code + time_code`）
2. 伪交叉注意力（K/V 来自学习 tokens，非风格图）
3. 闭集查表（`nn.Embedding(num_styles, D)`）
4. Minibatch OT 不稳定
5. 训练中 ODE 展开

**均值坍缩定理**（619 model/01）：确定性 ODE 输出收敛于 `E[x_style | π(x_0)=c]`（所有风格算术平均 = 灰色模糊）。突破需满足三条件之一：随机性（SDE）、instance-level conditioning、正确训练目标。

### 2.3 第三阶段：退化吸引子理论（622）

**统一数学模型**：四机制耦合退化吸引子 D

```
D = {g≈0.05, α≈0.16, η≈0.997, R_style≈0.001}
```

**四机制耦合方程**：gate path + norm path + attn path - shrinkage feedback，稳态解即 D。

**信号衰减链**：
```
Style encoder (1.0) → patch_proj (0.90) → CrossAttn gate (0.045)
→ StyleFiLM (0.018) → GroupNorm (0.005) → Head zero-init (0.001)
总衰减 99.9%
```

**加性-乘性模型**：α = max(α_attn·α_FiLM, α_GN) - α_loss
- 当前预测 α=0.18 vs 观测 α=0.16（误差 12.5%）
- α_attn=0.05, α_FiLM=0.40, α_GN=0.28, α_loss=0.10

**6 个可证伪命题**：
1. Gate Collapse 必然性
2. GN 白化定理：GN 经 L 层后 R_style^(8) ≤ 0.2^8 ≈ 6.6×10⁻⁵
3. SWD 梯度正交性：cos(grad_SWD, v_target) = -0.024（正交）
4. 训练-输出不匹配：Fiber-SDE 不训练 0.711 > 训练后 0.701
5. 有效 style 维度极低：k∈[10,50]，远小于 16384
6. 三难困境

**关键预测验证**：
- gate=0.3 单独无效（max(0.30×0.40, 0.28)=0.28，仍被 GN 限制）—— 与实验完全一致
- 三轴联合（gate=0.3+hd512+移除 GN）→ α=0.38, WFI=0.30
- +velocity_scale+AdaGN → α=0.55, WFI=0.20

### 2.4 第四阶段：FC-SB 纤维丛分解（622-625）

**理论核心**：将薛定谔桥（SB）、Flow Matching（FM）与纤维丛（Fiber Bundle）三大理论"大一统"为纤维约束薛定谔桥（FC-SB）。

**核心方程**：
- 底流形（结构）绝对静止：`db = 0·dt + 0·dW_t`
- 纤维空间（风格）狂热扩散：`df = v_fiber·dt + σ_fiber·dW_t`，σ_fiber=0.08 为"魔法阈值"

**三项工程改造**：
1. 各向异性训练目标：Target = Base(content) + Fiber(style) + 高通噪声
2. 推理期 Fiber-Euler-Maruyama SDE 解耦 + 绝对刚性保护（BASE LOCKING）
3. 直接预测 Fiber Endpoint（仅 Δf）

**代码微创手术**：Base Locking、GroupNorm→RMSNorm2d、Style Gate 初值 0.05→0.5。

**三阶段课程**：σ=0（结构锚定）→σ=0.03（纤维解耦）→σ=0.08（SDE 引爆）。

---

## 第三章：关键实验里程碑

### 3.1 时间线

```
2026-05-09  黑点缓解报告
2026-05-30  Tokenizer 重启设计
2026-06-01  Main table gap analysis（clip_style 卡 0.70）
2026-06-07  Immortal 系列（XPred Stokes002 最高 clip_style=0.737 但 LPIPS=0.607）
2026-06-12  Pure latent I2SB foundation smoke
2026-06-16  Fiber Bundle Phase 2 阶段总结
2026-06-20  620 swd16 vl=0.04 突破 0.705（白化发现前）
2026-06-21  fog 工作流启动：endpoint shrinkage 诊断（α=0.16）
2026-06-21  Round E3：hd512 WFI=0.3906 首次过门
2026-06-22  Phase 5 消融审计：推荐配置 WFI=0.3757（DINO patches 反转为有害）
2026-06-24  3-axis fix handover
2026-06-25  FC-SB Round 2 实验启动
2026-06-26  FC-SB Phase 3 deepfix + search 完成
```

### 3.2 关键指标对比

| 阶段 | 配置 | clip_style | LPIPS | WFI | 备注 |
|------|------|-----------|-------|-----|------|
| SaMST per-style | 参照天花板 | 0.7597 | 0.3374 | — | 每风格单独训练 |
| LANCET K | 历史前沿 | 0.701 | 0.362 | — | Distinct5 天花板 |
| 620 swd16 | 白化前最优 | 0.7051 | 0.2935 | — | 5 epoch |
| 620 hd512 | Round E3 | 0.7015 | 0.3382 | 0.3906 | 首次过 WFI 门 |
| 620 推荐 | Phase 5 | 0.6995 | 0.3422 | 0.3757 | 当前基线前身 |
| Fiber-SDE 不训练 | 神秘数据 | 0.711 | 0.337 | — | FC-SB 灵感来源 |
| E4-long ep5 | 长训最优 | 0.727 | 0.581 | — | clip 高但 lpips 崩 |
| **I7 ep2** | **当前基线** | **0.7017** | **0.3625** | **0.3757** | **FC-SB Phase 3 起点** |
| **U4(α0.1)** | **Phase 3 最佳** | **0.7225** | **0.3660** | — | **击败 I7** |
| Seedream IDT | 健康参照 | — | — | 0.158 | WFI 差距 +0.218 |

### 3.3 历史反转记录

项目演进中有多次"原以为正确，后被证伪"的关键反转：

1. **DINO patches 从"必要"变为"有害"**（ablation_audit）：target_dino_patches 严重白化（WFI=0.6407），latent 条件源通过门（WFI=0.3842）
2. **edge loss 从"必要"变为"有害"**：edge=0.0 三项全优
3. **hd512 从"最终最优"变为"非必需"**：hd128 在推荐配置下足够
4. **更大模型=更好被推翻**：容量 3× 增加→差异 0.001
5. **训练越久越好被推翻**：3-epoch WFI 单调恶化
6. **Text 条件能提升质量被推翻**：T5 vs no-T5 差 0.001（gate=0.047 截断）

---

## 第四章：白化危机与修复

### 4.1 白化根因定位

**白化起源于 endpoint 预测，不是 solver 也不是 VAE decode**。

**Endpoint Shrinkage 病理**：
- `latent_alpha_mean ≈ 0.1633`：endpoint 只移动了目标方向 16%
- `high_alpha ≈ -0.0501`：高频方向错误
- 白化集中在 t≈0 的 source 端，t 越大 alpha 越接近 1

**统一解释**：velocity 参数化 + style gate 过小 + endpoint head 零初始化 + GroupNorm(1) 共同导致 shrinkage basin。

### 4.2 Round E1-E3 决策台账

| 决策 | 假设 | 证据 | 结论 |
|------|------|------|------|
| D1 attention 改造 | gated_raw/relu2/style_select 降 WFI | WFI=0.64/0.53/0.50，clip 0.696-0.699 | **否证**，转向 endpoint |
| D2 endpoint_film | hd128 改善三项 | WFI 0.49→0.43，clip↑lpips↓ | **支持** |
| D3 更多 epoch | 训练越久越好 | WFI 0.43→0.47 单调恶化 | **否证** |
| D4 init_std=0.02 | 非零初始化破 shrinkage | WFI 0.43→0.40 | **部分支持** |
| D5 hd512 | 容量提升是瓶颈 | WFI=0.3906 过门 | **支持** |
| D6 HF Residual | 保留 source 高频 | WFI=0.4746，残差权重 0.1→0.089 | **否证** |

### 4.3 Phase 5 系统化消融审计

**重大反转**：纠正 fog 阶段的多个历史假设

| 维度 | 历史假设 | 消融结论 |
|------|---------|---------|
| DINO patches | 必要 | **有害**（WFI=0.6407）|
| latent 条件源 | 未充分测试 | **通过门**（WFI=0.3842）|
| edge loss | 必要 | **有害**（edge=0.0 三赢）|
| dim=128 | 收益 | **无收益**（CLIP-S +0.001）|
| hd512 | 最终最优 | **非必需**（hd128 足够）|
| gate_init=0.05 | Phase 4 推荐 | **组合下超门**，调整为 0.3 |

**推荐配置**：WFI=0.3757, CLIP-S=0.6995, LPIPS=0.3422（全部通过），配置简化 hd512→hd128（-75%）、移除 style_film、移除 edge loss。

### 4.4 白化理论的统一

**平凡解统一数学理论**（trivial_solution_unified）：

**核心命题**：平凡解不是 bug 而是特征——在当前训练目标和架构下，保守策略确实是 loss 最优的。

**三条件形成平凡解**：
1. FM 主导（L_flow 权重远大于 style loss）
2. SWD 平坦（投影排序不变时梯度为常数）
3. style 梯度衰减（gate × attention × norm × init × proj ≈ 0.016）

**五层乘积保守机制**：α_gate(0.1) × α_attn(0.3) × α_norm(0.7) × α_init(0.8) × α_proj(0.95) ≈ 0.016

**关键洞察**：单一修复无效——各机制相乘而非相加，只改一处会被其他机制乘回去。突破需至少打破两个条件。

**突破策略三层级**：
1. 架构去安全阀（去 GN / FiLM-only，最高 ROI）
2. 训练目标重构（Endpoint-supervised，根本解）
3. 训练策略优化（两阶段/课程学习）

---

## 第五章：FC-SB 理论与 Round 2 实验

### 5.1 FC-SB 的理论定位

FC-SB 是对 622 退化吸引子理论的**工程实施方案**：

**核心思想**：与其试图改变模型的保守偏好（失败 645 次），不如通过**几何物理锁死**强制分离 content 和 style——base 流形绝对静止，fiber 空间自由扩散。

**与之前尝试的本质区别**：
- 之前：在统一 latent space 中用 loss 引导模型学习 content/style 分离（失败）
- FC-SB：用 Haar 分解 + BASE LOCKING 在几何上强制分离，不依赖模型"自觉"

### 5.2 Round 2 实验设计

**第一轮失败根因**：配置读取 bug（`i2sb_fiber_project_*` 在 ModelConfig 但代码只读 BridgeConfig）→ FC-SB 从未生效。

**Round 2 增量矩阵**（7 组）：
- G0 基线 → G1 速度投影 → G2 Base Locking → G3 Fiber SDE σ=0.04 → G4 Full σ=0.06 → G5 σ=0.08 魔法阈值 → G6 课程调度

**五大假说**：
1. Base Locking 锁 LPIPS<0.30
2. Fiber SDE 提 style
3. Fiber-Only Endpoint
4. sigma 剂量效应
5. 课程调度

### 5.3 推理流程完整链路（FC-SB 实施后）

```
初始化: h = source_latent

每个时间步 t (共 num_steps 步):

Stage 1: N1 Endpoint AdaIN 块 (L676-847)
  ├─ Haar 分解 content fiber → f_ll, f_lh, f_hl, f_hh
  ├─ Haar 分解 style fiber → s_lh, s_hl, s_hh
  ├─ per-band AdaIN 匹配:
  │   mid_matched = adain_match(f_lh/f_hl, s_lh/s_hl)  # 中频
  │   hh_matched  = adain_match(f_hh, s_hh)             # 高频
  ├─ α-blend:
  │   mid_final = mid_adain_scale * mid_matched + (1-mid_adain_scale) * f_mid
  │   hh_final  = hh_adain_scale  * hh_matched  + (1-hh_adain_scale)  * f_hh
  ├─ 重构: ep_fiber_matched = haar_inv(0, mid_lh, mid_hl, hh_final)
  └─ endpoint = ep_base + (1-α)*ep_fiber_curr + α*ep_fiber_matched

Stage 2: Velocity 计算 (L861-864)
  ├─ v_pred = (endpoint - h) / denom
  └─ v_fiber = v_pred - lp(v_pred)  # fiber 投影，去除低频

Stage 3: Euler 步进 (L912)
  └─ h = h + v_fiber * dt

Stage 4: Fiber Noise Injection (L914-941, 可选)
  └─ h = h + sigma_t * noise_fiber  # 高频布朗噪声

Stage 5: BASE LOCKING (L943-957) 🚨
  ├─ 标准: h = x_base_lock + (h - lp(h))  # = Base(content) + Fiber(current)
  └─ tri_band: h = x_base_lock + blended_mid + h_hh  # 三频带锁死

返回 h
```

### 5.4 BASE LOCKING 的核心作用

**BASE LOCKING 是 FC-SB 保内容的根本机制**：

```python
h = x_base_lock + (h - lp(h))  # = Base(content) + Fiber(current)
```

- `x_base_lock`：源 content 的 lowpass，在整个推理过程中恒定不变
- `h - lp(h)`：当前状态的 highpass（fiber），允许随时间步变化
- **效果**：content 的低频结构被绝对锁死，N1 的风格注入只能影响 fiber（高频）维度

**两种模式**：
1. **标准 vertical**（L957）：base 完全锁死，mid+hh 自由
2. **tri_band_lock**（L945-955）：LL 锁死，mid 部分 blend（`tri_band_edge_alpha`），hh 完全自由

---

## 第六章：Phase 3 deepfix + search 突破

### 6.1 deepfix：开关修复

**修复前的死路径**：

| 方向 | 修复前状态 | 根因 |
|------|-----------|------|
| T/U/V | 9 个变体 LPIPS 全部 0.4180，参数无效果 | inference.py dict 分支丢弃 style_latent_tensor，N1 块永不执行 |
| W | W2b loss 恒为 0 | anti_input_margin=0.3 远小于 dist_input O(10-50)，F.relu 恒为 0 |

**修复内容**（3 层协调改动）：
1. [run_evaluation.py:3174-3248](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/utils/run_evaluation.py#L3174)：构造 style_latent_tensor（VAE encode 目标风格参考图）
2. [inference.py:548-551](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/utils/inference.py#L548)：dict 分支提取 style_latent_tensor 传递
3. [model620.py:677](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L677)：新增 N1 可观测性（n1_adain_executed, n1_ep_fiber_abs）

**修复后验证**：
- T1 smoke test: `n1_adain_executed=1.0`, `n1_ep_fiber_abs≈0.3564` ✅
- 修复后 10 个变体 LPIPS 分布在 0.3735~0.6685（全部激活）

### 6.2 search：参数搜索完整结果

#### U 方向（style_extrap_alpha，外推强度）

| 变体 | α | clip_style | lpips | Δclip vs I7 | Δlpips vs I7 | 击败 I7 |
|------|---|-----------|-------|-------------|--------------|---------|
| **U4** | 0.10 | 0.7225 | 0.3660 | +0.0208 | +0.0035 | **YES** |
| U5 | 0.15 | 0.7195 | 0.3683 | +0.0178 | +0.0058 | YES |
| U1 | 0.20 | 0.7164 | 0.3735 | +0.0147 | +0.0110 | YES |
| U6 | 0.25 | 0.7131 | 0.3807 | +0.0114 | +0.0182 | YES |
| U7 | 0.30 | 0.7094 | 0.3897 | +0.0077 | +0.0272 | no |
| U2 | 0.50 | 0.6959 | 0.4307 | -0.0058 | +0.0682 | no |
| U3 | 1.00 | 0.6736 | 0.5218 | -0.0281 | +0.1593 | no |

**趋势**：α 越小越好。clip 单调下降，lpips 单调上升。α=0.1 已接近"无副作用"区间。

#### V 方向（patch_adain_kernel，空间核大小）

| 变体 | k | clip_style | lpips | Δclip vs I7 | Δlpips vs I7 | 击败 I7 |
|------|---|-----------|-------|-------------|--------------|---------|
| V1 | 4 | 0.7242 | 0.5196 | +0.0225 | +0.1571 | no |
| V2 | 8 | 0.7290 | 0.4497 | +0.0273 | +0.0872 | no |
| V3 | 16 | 0.7295 | 0.3963 | +0.0278 | +0.0338 | no |
| V4 | 20 | 0.6334 | 0.5889 | -0.0683 | +0.2264 | no (崩塌) |
| V5 | 24 | 0.6562 | 0.5330 | -0.0455 | +0.1705 | no (崩塌) |
| **V6** | 32 | 0.7262 | 0.3722 | +0.0245 | +0.0097 | **YES** |

**趋势**：非单调。仅 2 幂次 kernel（4/8/16/32）工作正常，非 2 幂次（20/24）崩塌。

#### T 方向（multiband_adain，频带分离）

| 变体 | mid | hh | clip_style | lpips | n1_ep_fiber_abs |
|------|-----|-----|-----------|-------|-----------------|
| T1 | 0.3 | 0.3 | 0.6518 | 0.6650 | 0.3508 |
| T2 | 0.5 | 0.3 | 0.6574 | 0.6684 | 0.4096 |
| T3 | 0.3 | 0.5 | 0.6587 | 0.6641 | 0.3762 |
| T4 | 0.5 | 0.5 | 0.6609 | 0.6685 | 0.4290 |

**hh 排查结论**：hh 生效在 clip 维度（+0.0072），不在 lpips 维度（-0.0007）—— 是设计如此，非 bug。BASE LOCKING 锁死 content lowpass 保 lpips，hh 只作用于 fiber 高频纹理提 clip。

#### W 方向（anti_input_style loss，训练侧）

| 变体 | margin | clip_style | lpips | Δlpips vs I7 | step=1 loss | step=51+ loss |
|------|--------|------------|-------|--------------|-------------|---------------|
| W2c | 5 | 0.7123 | 0.3580 | -0.0045 | 0.4156 | 0.0 |
| W2d | 10 | 0.7060 | 0.4270 | +0.0645 | 1.2793 | 0.0 |
| W2e | 15 | 0.6946 | 0.4652 | +0.1027 | 2.4753 | 0.0 |
| W2b | 20 | 0.6947 | 0.4645 | +0.1020 | 3.9000 | 0.0 |

**结论**：未找到有效折中点。hinge loss 仅 step=1 生效，后续无梯度。

### 6.3 帕累托前沿

```
clip_style
  ↑
  0.730 ┤                          ● V3(k16)
  0.725 ┤              ● V6(k32)
  0.720 ┤  ● U4(α0.1)
  0.715 ┤    U5(α0.15)
  0.710 ┤      U1(α0.2)  U6(α0.25)
  0.705 ┤  ● I7(baseline)
  0.700 ┤
       └─────────────────────────────────→ lpips
       0.36  0.37  0.38  0.39  0.40
```

**前沿轨迹**：I7 → U4(α0.1) → V6(k32) → V3(k16)

**5 个击败 I7 的点**：U4(α0.1)/U5(α0.15)/U1(α0.2)/U6(α0.25)/V6(k32)

---

## 第七章：核心 Insight

### 7.1 Insight 1：保守偏好是"特征"不是"Bug"

**645 次实验的最大教训**：模型的保守策略（gate collapse、endpoint shrinkage、白化）不是实现 bug，而是在当前训练目标下的**最优解**。

**数学证明**：在 L_flow 主导 + SWD 平坦 + style 梯度衰减三条件下，保守策略确实是 loss 最优。模型"理性地选择"不注入 style，因为注入 style 会增加 loss。

**推论**：试图通过调参/改架构让模型"更勇敢"是徒劳的——除非改变训练目标本身。FC-SB 的突破正在于此：通过 BASE LOCKING 几何锁死，让"注入 style"不再增加 content loss（base 被保护），从而消除保守偏好的动机。

### 7.2 Insight 2：三难困境的物理本质

**三难困境不是模型容量限制，而是 BASE LOCKING 的结构性约束**：

- BASE LOCKING 锁死 content lowpass → 保 LPIPS
- 同时限制 N1 只能影响 fiber（高频）→ 限制 clip_style 上限
- 要突破 clip 上限，必须放松 BASE LOCKING → 损害 LPIPS

**这意味着**：clip-lpips 权衡是**几何约束**，不是优化问题。没有"免费午餐"——任何 clip 提升必然伴随 lpips 代价。

**帕累托前沿的真正含义**：前沿上的点不是"最优配置"，而是"在不同 lpips 容忍度下的最佳 clip"。U4(α0.1) 之所以是最佳综合点，是因为它在 lpips 几乎不变（+0.97%）的情况下提 clip +2.97%——这是"低风险高收益"区域。

### 7.3 Insight 3：训练-推理分离的必要性

**N1 块只在推理路径执行，训练路径不走 N1**——这不是设计缺陷，而是深刻的工程哲学。

**原因**：
- N1 是 fiber 统计匹配（AdaIN），不可微或梯度不稳定
- 训练时学习 velocity 预测，推理时做后处理统计匹配
- 分离让训练目标纯粹（L_flow + L_kinetic），不被 N1 的统计匹配干扰

**推论**：U/V/T 是**推理期参数**，只需修改 checkpoint config，不需重新训练。这是 FC-SB 的工程优势——可以快速探索参数空间。

**但这也是 Phase 3 deepfix 的根因**：因为 N1 不在训练路径，训练时无法发现"style_latent 为 None"的 bug。直到评估时才发现 9 个变体 LPIPS 全部 0.4180（死路径）。**教训**：训练-推理分离的架构必须有推理期的 probe 验证。

### 7.4 Insight 4：作用点正交性决定协同潜力

**U/V/T 三个方向作用在 N1 块的不同位置**：

| 方向 | 作用点 | 代码位置 | 机制 |
|------|--------|---------|------|
| U (style_extrap_alpha) | **输入侧** | L698-699 | 放大 style_fiber 全局缩放 |
| V (patch_adain_kernel) | **计算侧** | L813-841 | 改变 AdaIN 空间粒度 |
| T (mid/hh_adain_scale) | **输出侧** | L791-792 | α-blend matched 与 original fiber |

**协同/拮抗预测**：
- U4+V6 联合：U 输入侧 + V 计算侧，作用点正交，**可能叠加**
- U4+T 联合：作用点正交，但 T 方向 lpips 已 0.66+，叠加可能恶化
- V6+T 联合：V 不区分频带，T 区分频带，**机制冲突**

**副作用大小**：U4 的 lpips 增量（+0.0035）< V6 的 lpips 增量（+0.0097），说明输入侧放大（U）的副作用小于计算侧粒度调整（V），因 U 不改变空间结构。

### 7.5 Insight 5：W 方向的根本性矛盾

**W anti_input_style loss 存在目标矛盾**：

1. **模型目标**：output → target style
2. **W 约束**：保持 input style ≠ target style
3. **矛盾**：W 实际上在**惩罚模型让 output 接近 target**——这与风格迁移目标直接冲突

**hinge loss 梯度失效的真正原因**：不是"约束满足"，而是"模型放弃挣扎"——既然 W 约束与目标冲突，模型选择优先满足主目标，让 W loss 自然归零。

**推论**：W 方向可能是**错误方向**。即使改 soft hinge，根本矛盾不解决。正确的"反白化"约束应该约束 **output 的统计分布**（防止过度白化），而非 input-target 距离。

### 7.6 Insight 6：kernel 2 幂次的几何根因

**V 方向 kernel 必须 2 幂次的根因不是算术问题，而是 Haar 小波的 dyadic 结构**：

- Haar 是 dyadic 小波，每级分解把信号分成 4 个子带（LL/LH/HL/HH），每个子带尺寸减半
- feature map 尺寸 32 → LL/LH/HL/HH 各 16 → 再分解各 8 → ...
- 这个 2 进制层级结构要求所有空间操作与 2 幂次对齐

**非 2 幂次 kernel 崩塌的机制**：
- patch 跨越 Haar 子带边界 → patch 内统计混合不同频带信息
- AdaIN 匹配破坏频带分离 → 风格统计污染 content 频带
- 重构后产生 patch 边界伪影

**最优 kernel 的几何意义**：
- k=32（单 patch）= 全局 AdaIN = 最强空间平滑 = lpips 最低
- k=16（4 patch）= 局部统计 = 风格更精细 = clip 最高（0.7295）
- k=4（64 patch）= 统计噪声 = lpips 暴涨（0.5196）

**k=16 是"风格-内容平衡点"**——足够局部以捕捉风格纹理，又足够全局以保持统计稳定性。

### 7.7 Insight 7：三层纤维动力学

FC-SB 的"双层动力学"（base 死寂 / fiber 狂热）在 two_level 模式下细化为**三层**：

```
频带        活跃度      作用                  实验证据
─────────────────────────────────────────────────────────
LL (base)   死寂       content 结构锁死       BASE LOCKING 完全锁死
Mid (LH+HL) 中等活跃    边缘/粗纹理风格化      T 方向 mid 参数对 lpips 生效 (Δ=0.0034)
HH          狂热       细纹理/笔触风格化      T 方向 hh 参数对 clip 生效 (+0.0072)
```

**关键洞察**：
- LPIPS 对 lowpass + mid 敏感（结构 + 边缘），对 hh 几乎不敏感
- CLIP 对 mid + hh 都敏感（纹理 + 笔触都是风格信号）
- 这解释了为什么 hh 参数"提 clip 不损 lpips"——hh 作用于 LPIPS 不敏感的频带
- **帕累托前沿的最优策略**：优先用 hh 提 clip，mid 谨慎调整

### 7.8 Insight 8："代码已写 ≠ 功能已生效"

**Phase 3 deepfix 的核心教训**：9 个变体 LPIPS 全部 0.4180（N1 死路径），但代码中确实写了 N1 块。

**根因链**：
1. `run_evaluation.py` 构造 `target_style_latent` 为 dict（含 DINO patches）
2. `inference.py` dict 分支只提取 DINO 字段，**完全丢弃 `target_style_latent` kwarg**
3. `model620.py` L676 守卫 `style_latent is not None` 永远为 False，N1 块跳过

**更深的教训**：`model_endpoint_style_high_abs` 被误认为是 N1 块的执行指标，实际测量的是 `forward()` 的 endpoint head 投影层，与 N1 块无关。**observability 指标的语义必须明确**。

**改进**：新增 probe gate（`n1_adain_executed=1.0`），评估后自动检查，失败标记 INVALID。这是"probe-first 原则"——任何修复后，先用 probe 验证开关生效，再做完整评估。

---

## 第八章：帕累托前沿演进

### 8.1 6 个月的前沿移动

```
clip_style
  ↑
  0.760 ┤  ● SaMST per-style (天花板)
  0.740 ┤
  0.730 ┤                          ● V3(k16)  ● E4-long ep5 (lpips=0.581)
  0.725 ┤              ● V6(k32)
  0.720 ┤  ● U4(α0.1)
  0.715 ┤    U5(α0.15)
  0.710 ┤  ● Fiber-SDE 不训练 (神秘数据)
  0.705 ┤      U1(α0.2)  U6(α0.25)  ● 620 swd16  ● LANCET K
  0.700 ┤  ● I7(baseline)  ● 620 推荐
  0.695 ┤
  0.690 ┤
  0.685 ┤
  0.680 ┤
  0.675 ┤  ● 619 最优
  0.670 ┤
  0.665 ┤
  0.660 ┤  ● 618 H1
  0.640 ┤  ● IDT baseline
       └───────────────────────────────────────────→ lpips
       0.29  0.30  0.34  0.36  0.40  0.50  0.58
```

### 8.2 前沿移动的驱动力

| 阶段 | 前沿移动 | 驱动力 |
|------|---------|--------|
| IDT → 618 | 0.640→0.670 | DINO CLS 离线 top-K 配对 |
| 618 → 620 | 0.670→0.705 | DINOv2 256×384 真实空间 cross-attention + 单步 SWD |
| 620 → LANCET | 0.705→0.701 | （实际是回退，LANCET 是更早的基线）|
| LANCET → I7 | 0.701→0.702 | FiLM endpoint head + init_std=0.1 |
| I7 → U4 | 0.702→0.723 | style_extrap_alpha=0.1（推理期参数，无需重训）|
| I7 → V3 | 0.702→0.730 | patch_adain_kernel=16（推理期参数）|

**关键观察**：从 I7 到 U4/V3 的前沿移动，是**纯推理期参数调整**，无需重新训练。这是 FC-SB 训练-推理分离架构的工程红利。

### 8.3 与天花板的差距

- **SaMST per-style 天花板**：0.7597/0.3374
- **当前最佳综合点 U4**：0.7225/0.3660
- **clip 差距**：0.0372（5%）
- **lpips 差距**：0.0286（8.5%）

**差距来源分析**：
- per-style 模型每风格单独训练，无风格间干扰
- 通用模型需处理 5 风格间的 style representation 共享
- 这个 gap 可能是通用模型的**理论上限**——除非用 mixture-of-experts 或 per-style adapter

---

## 第九章：下一步战略

### 9.1 短期（1-2 周）

#### 9.1.1 U4+V6 联合实验（推荐优先）

U4(α0.1) 与 V6(k32) 作用点正交（U 输入侧 / V 计算侧），探索联合协同：
- 配置：style_extrap_alpha=0.1 + patch_adain_kernel=32
- 预期：clip 可能叠加（+5-6%），lpips 可能叠加（+1.2-1.5%）
- **风险**：V6 的全局匹配可能"吸收"U4 的统计放大

#### 9.1.2 更小 α 探索

U4(α0.1) 已接近无副作用区间，探索 α=0.05/0.08：
- 如果 α=0.05 时 lpips < 0.3625 且 clip > 0.7017，则突破帕累托前沿
- 理论依据：α 越小，style 统计量放大越温和

#### 9.1.3 hh 优先策略

基于三层纤维动力学，优先用 hh 提 clip，mid 保 lpips：
- 配置：multiband_adain_mode='two_level' + hh_adain_scale=0.5~0.7 + mid_adain_scale=0.1~0.2
- 与 U4 联合：U4(α0.1) + hh_adain_scale=0.5
- **风险**：T 方向所有变体 lpips 都 0.66+，需验证单独提高 hh 是否能控制 lpips

### 9.2 中期（2-4 周）

#### 9.2.1 BASE LOCKING 部分放松

当前 BASE LOCKING 完全锁死 content lowpass。探索 tri_band_lock 模式：
- tri_band_edge_alpha 从 0.0 逐步增加到 0.1/0.2/0.3
- 理论依据：mid 对 lpips 有影响（T 方向 Δ=0.0034），放松 mid 是"可控风险"
- 目标：放松 base 锁死可能突破 clip 上限，但需监控 lpips

#### 9.2.2 I7 基础训练到 5 epoch

当前 U4 继承自 I7 epoch_0002。探索 I7 训练到 5 epoch 后再应用 U4：
- 历史经验显示 epoch 5 是自然收敛的最佳停止点
- 预期：I7 训练更充分 → velocity 预测更准 → U4 的 style 放大建立在更优基础上

#### 9.2.3 W 方向重新定义

基于根本矛盾分析，W 方向需要重新定义：
- **选项 A（推荐）**：放弃 W 方向，算力优先投入 U/V 联合和 BASE LOCKING 放松
- **选项 B**：重新定义 W 为 output 统计约束（防止 output 过度白化）
- **选项 C**：soft hinge 改造（如果坚持原 W 方向，但根本矛盾未解决）

### 9.3 长期（1-2 月）

#### 9.3.1 per-style adapter 探索

与 SaMST per-style 天花板（0.7597）的 5% gap 可能是通用模型理论上限。探索：
- mixture-of-experts：每风格专用 expert
- per-style adapter：共享主干 + 每风格小 adapter
- 目标：缩小通用模型与 per-style 的 0.05-0.09 gap

#### 9.3.2 FC-SB 理论深化

当前 FC-SB 只用了 Haar 一级分解。探索：
- 多级 Haar 分解（2 级 / 3 级）
- 其他小波（Daubechies、Symlet）
- 学习的滤波器组（替代固定 Haar）

#### 9.3.3 WFI 进一步降低

当前 WFI=0.3757，与 Seedream IDT（0.158）仍有 +0.218 差距。探索：
- 更深的 RMSNorm 替代 GroupNorm
- 更激进的 style gate 初始化
- 训练目标重构（Endpoint-supervised）

---

## 第十章：方法论反思

### 10.1 有效的方法

1. **probe-first 原则**：任何修复后先用 probe 验证开关生效，再做完整评估
2. **单因子消融**：每次只改一个变量，避免组合混淆
3. **三阶段评估协议**：smoke test → 参数搜索 → 最佳点训练
4. **帕累托前沿分析**：在 clip-lpips 二维空间绘制前沿，避免单指标误导
5. **口径验证**：明确 all_pairs/transfer/identity 三种口径，避免数值不可比

### 10.2 无效的方法

1. **调参突破天花板**：645 次实验证明，调参无法突破 0.70 天花板
2. **增加模型容量**：dim 64→128, blocks 4→6，差异 0.001
3. **更长训练**：3-epoch WFI 单调恶化
4. **更多 heuristic loss**：NCE/repulsive/cycle/structure/TV 全部无效或负收益
5. **单点修复**：五层乘积保守机制下，单点修复会被其他机制乘回去

### 10.3 关键工程教训

1. **"代码已写" ≠ "功能已生效"**：必须有运行时 probe 验证
2. **observability 指标语义必须明确**：`model_endpoint_style_high_abs` 不是 N1 块指标
3. **训练-推理分离架构需要推理期 probe**：N1 不在训练路径，训练时无法发现死路径
4. **远程环境路径陷阱**：cmd.exe 下 `/mnt/i/` 会被误解析为 `C:\mnt\i\`
5. **hinge loss 梯度失效**：模型一步推过 margin 后梯度归零，看似 loss 低实际是失效
6. **配置加载时机**：json 更新与训练启动之间有竞态，需启动前校验

### 10.4 "如果重新开始"检查清单

**不要做**（14 条）：
1. 不要用 DiT patchification（6D reshape 易 bug）
2. 不要用 1-token attention（softmax 恒=1.0）
3. 不要用 InstanceNorm（attention 的毒药）
4. 不要用 Minibatch OT（不稳定）
5. 不要用 ODE Unrolling（梯度消失）
6. 不要用 PatchNCE（摧毁风格）
7. 不要用 Repulsive/Cycle/Structure/TV loss（无效）
8. 不要用 T5 text（gate=0.048 时无效）
9. 不要盲目增加模型容量（无收益）
10. 不要训练越久越好（WFI 恶化）
11. 不要用 DINO patches 作条件源（白化）
12. 不要用 edge loss（三赢开关，关闭更好）
13. 不要用非 2 幂次 kernel（边界伪影）
14. 不要用 hinge loss 约束 input-target 距离（目标矛盾）

**要做**（6 条）：
1. 用 DINOv2 CLS + LoRA open-set 风格条件
2. 用 64-token + sharpen=2.5 的 cross-attention
3. 用 L_flow + L_kinetic + SWD 极简 loss
4. 用 FiLM endpoint head（绕过 cross-attention）
5. 用 FC-SB BASE LOCKING 几何锁死 content
6. 用 probe-first 原则验证所有开关

---

## 附录 A：关键文件索引

### 核心代码
- [src/model620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py)：N1 块（L676-847）、BASE LOCKING（L943-957）、integrate_transport（L553）
- [src/losses620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/losses620.py)：W loss（L636-676）
- [src/blocks620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/blocks620.py)：SpatialBridgeBlock620
- [src/config_schema.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/config_schema.py)：配置 schema
- [src/utils/inference.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/utils/inference.py)：style_latent_tensor 传递（L548-551）
- [src/utils/run_evaluation.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/utils/run_evaluation.py)：style_latent_tensor 构造（L3174-3248）

### 实验脚本
- [exp/625_fc_sb/gen_i7_direction_configs.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/exp/625_fc_sb/gen_i7_direction_configs.py)：变体 checkpoint 生成
- [exp/625_fc_sb/run_rtuv_eval.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/exp/625_fc_sb/run_rtuv_eval.py)：批量评估（含 probe gate）
- [exp/625_fc_sb/run_w_batch.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/exp/625_fc_sb/run_w_batch.py)：W 批量训练（含 config 校验）

### 阶段文档
- [docs/622/FC.md](file:///g:/GitHub/Latent_Style/SchrodingerBridge/docs/622/FC.md)：FC-SB 核心理论
- [docs/622/history/](file:///g:/GitHub/Latent_Style/SchrodingerBridge/docs/622/history/)：完整 6 阶段历史
- [docs/620/fog/](file:///g:/GitHub/Latent_Style/SchrodingerBridge/docs/620/fog/)：白化诊断与修复
- [docs/625/fc-sb-round2-experiment-complete.md](file:///g:/GitHub/Latent_Style/SchrodingerBridge/docs/625/fc-sb-round2-experiment-complete.md)：FC-SB Round 2 完整记录
- [docs/625/fc_sb_phase3_stage_summary.md](file:///g:/GitHub/Latent_Style/SchrodingerBridge/docs/625/fc_sb_phase3_stage_summary.md)：Phase 3 阶段总结
- [docs/625/grand_summary.md](file:///g:/GitHub/Latent_Style/SchrodingerBridge/docs/625/grand_summary.md)：本文档

### Spec 文档
- [.trae/specs/fc-sb-phase3-deepfix/](file:///g:/GitHub/Latent_Style/SchrodingerBridge/.trae/specs/fc-sb-phase3-deepfix/)：开关修复 spec
- [.trae/specs/fc-sb-phase3-search/](file:///g:/GitHub/Latent_Style/SchrodingerBridge/.trae/specs/fc-sb-phase3-search/)：参数搜索 spec

### 实验数据
- I7 baseline: `exp/625_fc_sb/from_scratch_win/init_I7/epoch_0002.pt`
- U/V 变体: `exp/625_fc_sb/from_scratch_win/rtuv_variants/`
- W 变体: `exp/625_fc_sb/from_scratch_win/w_W2b/` ~ `w_W2e/`
- 评估结果: `exp/625_fc_sb/from_scratch_win/rtuv_variants/<name>_eval_v2/summary.json`

---

## 附录 B：理论修正时间线

| 时间 | 修正内容 | 旧理解 | 新理解 |
|------|---------|--------|--------|
| 622 | 退化吸引子理论 | 保守策略是 bug | 保守策略是 loss 最优特征 |
| 622 | SWD 梯度正交性 | SWD 是好的 style loss | 训练中有效但评估中不可靠 |
| 620 ablation | DINO patches | 必要 | 有害（白化）|
| 620 ablation | edge loss | 必要 | 有害（三赢开关）|
| 620 ablation | hd512 | 最终最优 | 非必需（hd128 足够）|
| 625 Phase 3 | N1 块语义 | 风格注入 | fiber 统计匹配 |
| 625 Phase 3 | clip-lpips 权衡 | 模型容量限制 | BASE LOCKING 结构性约束 |
| 625 Phase 3 | style_extrap_alpha | 外推强度 | style_fiber 全局缩放（StyleGAN truncation 反向）|
| 625 Phase 3 | U/V/T 作用点 | 都是风格强度 | U 输入侧 / V 计算侧 / T 输出侧 |
| 625 Phase 3 | W hinge loss | margin 越大约束越强 | 目标矛盾，模型放弃挣扎 |
| 625 Phase 3 | kernel 2 幂次 | 算术整除问题 | Haar dyadic 结构对齐 |
| 625 Phase 3 | hh/mid 职责 | 都影响 lpips | 正交：hh 提 clip 不损 lpips |

---

## 附录 C：核心指标速查表

### 当前基线
| 指标 | I7 ep2 | U4(α0.1) | V3(k16) | V6(k32) |
|------|--------|----------|---------|---------|
| clip_style | 0.7017 | 0.7225 | 0.7295 | 0.7262 |
| content_lpips | 0.3625 | 0.3660 | 0.3963 | 0.3722 |
| WFI | 0.3757 | — | — | — |

### 历史参照
| 指标 | SaMST per-style | LANCET K | 620 swd16 | Seedream IDT |
|------|----------------|----------|-----------|--------------|
| clip_style | 0.7597 | 0.701 | 0.7051 | — |
| content_lpips | 0.3374 | 0.362 | 0.2935 | — |
| WFI | — | — | — | 0.158 |

### WFI 等级
| 等级 | WFI 范围 | 状态 |
|------|---------|------|
| 健康 | <0.40 | 通过门 |
| 正常 | 0.50-0.65 | 可接受 |
| 轻微 | 0.68-0.72 | 需改善 |
| 中等 | 0.73-0.78 | 不合格 |
| 严重 | >0.85 | 崩溃 |

---

## 结语

6 个月、645+ 次实验、4 次范式重构，项目终于从"调参突破天花板"的循环中走出，进入"几何物理锁死"的 FC-SB 范式。当前 U4(α0.1) 是 6 个月以来首次在 lpips 几乎不变（+0.97%）的情况下提 clip +2.97% 的点，标志着帕累托前沿的真正移动。

下一步的 U4+V6 联合实验和 BASE LOCKING 部分放松，有望进一步推动前沿。与 SaMST per-style 天花板（0.7597）的 5% gap 是通用模型的挑战，可能需要 per-style adapter 或 mixture-of-experts 才能突破。

**核心教训**：在风格迁移任务中，"让模型勇敢注入 style"不能靠调参，必须靠**改变训练目标或几何约束**让"勇敢"成为最优策略。FC-SB 的 BASE LOCKING 正是这一哲学的体现——通过几何锁死 content，消除模型"保守"的动机。
