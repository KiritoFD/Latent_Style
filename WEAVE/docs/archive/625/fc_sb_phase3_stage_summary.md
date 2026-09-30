# FC-SB Phase 3 阶段性总结 — 模型图景、实验结果与理论修正

> 日期：2026-06-26
> 范围：FC-SB Phase 3 deepfix（开关修复）+ search（参数搜索与 hh 排查）
> baseline：I7 初始化（endpoint_film_init_std=0.1, style_film_init_std=0.0, style_embed_scale=4.0），epoch_0002
> I7 baseline 指标：clip_style=0.7017, content_lpips=0.3625（5-style all_pairs_overview, 25 对含 identity）

---

## 一、模型图景

### 1.1 FC-SB 核心范式

FC-SB（Fiber-Conditioned Schrödinger Bridge）的核心思想是在 latent space 中通过 Schrödinger Bridge 框架学习源到目标的传输，同时利用 fiber（highpass）分解实现 content 保真。

**关键分解**：latent = base（lowpass）+ fiber（highpass）
- base 锁死保 content 结构
- fiber 匹配 style 统计实现风格迁移

### 1.2 推理流程完整链路

推理走 `integrate_transport()`（[model620.py:553](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L553)，`@torch.no_grad()`），每个时间步 t 执行 5 个 stage：

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

### 1.3 BASE LOCKING 的核心作用

**BASE LOCKING 是 FC-SB 保内容的根本机制**（[model620.py:957](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L957)）：

```python
h = x_base_lock + (h - lp(h))  # = Base(content) + Fiber(current)
```

- `x_base_lock`：源 content 的 lowpass，在整个推理过程中恒定不变
- `h - lp(h)`：当前状态的 highpass（fiber），允许随时间步变化
- **效果**：content 的低频结构被绝对锁死，N1 的风格注入只能影响 fiber（高频）维度

**两种模式**：
1. **标准 vertical**（L957）：base 完全锁死，mid+hh 自由
2. **tri_band_lock**（L945-955）：LL 锁死，mid 部分 blend（`tri_band_edge_alpha`），hh 完全自由

### 1.4 N1 块的真正语义

**修正理解**：N1 不是"风格注入"块，而是"fiber 统计匹配"块。

- N1 匹配的是 fiber 的 per-channel 统计（μ/σ），通过 AdaIN 调整现有 fiber 的分布
- 不是注入新的风格信息，而是将 content fiber 的统计对齐到 style fiber
- 受 BASE LOCKING 限制，N1 只能影响 fiber（高频）维度，无法改变 content 的低频结构

**N1 的频带分离**（multiband_adain_mode='two_level'）：
- **mid（LH+HL）**：中频边缘信息，对 lpips 有一定影响
- **hh（HH）**：对角高频纹理，对 clip_style 有影响但对 lpips 几乎无影响
- 两者职责正交，由 BASE LOCKING 保证

### 1.5 训练 vs 推理路径分离

**关键设计**：N1 块只在推理路径（`integrate_transport`）执行，训练路径（`forward`）不走 N1。

- 训练时：`forward()` 学习 velocity 预测，不涉及 N1 的 fiber 统计匹配
- 推理时：`integrate_transport()` 执行 N1 的 fiber 统计匹配作为后处理

这意味着：
- U/V/T 参数是**推理期参数**，只需修改 checkpoint config 字段，不需重新训练
- W 参数是**训练期参数**（loss 项），需要重新训练

### 1.6 三层纤维动力学（精细图景）

FC-SB 的"双层动力学"（base 死寂 / fiber 狂热）在 two_level 模式下进一步细化为**三层**：

```
频带        尺寸      活跃度      作用                  实验证据
─────────────────────────────────────────────────────────────────
LL (base)   H/2×W/2   死寂       content 结构锁死       BASE LOCKING 完全锁死
Mid (LH+HL) H/2×W/2   中等活跃    边缘/粗纹理风格化      T 方向 mid 参数对 lpips 生效 (Δ=0.0034)
HH          H/2×W/2   狂热       细纹理/笔触风格化      T 方向 hh 参数对 clip 生效 (+0.0072)
```

**关键洞察**：
- LPIPS 对 lowpass + mid 敏感（结构 + 边缘），对 hh 几乎不敏感
- CLIP 对 mid + hh 都敏感（纹理 + 笔触都是风格信号）
- 这解释了为什么 hh 参数"提 clip 不损 lpips"——hh 作用于 LPIPS 不敏感的频带
- 帕累托前沿的最优策略：**优先用 hh 提 clip，mid 谨慎调整**

**tri_band_lock 模式**（L945-955）的三层控制：
- LL：完全锁死（`x_base_lock`）
- Mid：部分 blend（`tri_band_edge_alpha` 控制 content edge 与 current edge 的混合）
- HH：完全自由（`h_hh = h - h_mid_full`）

这是放松 BASE LOCKING 的安全阀：通过 mid 部分放松，允许风格影响边缘，同时保留 LL 锁死保 content 主体结构。

---

## 二、实验结果总结

### 2.1 Phase 3 deepfix：开关修复

#### 修复前的死路径

| 方向 | 修复前状态 | 根因 |
|------|-----------|------|
| T/U/V | 9 个变体 LPIPS 全部 0.4180，参数无效果 | inference.py dict 分支丢弃 style_latent_tensor，N1 块永不执行 |
| W | W2b loss 恒为 0 | anti_input_margin=0.3 远小于 dist_input O(10-50)，F.relu 恒为 0 |

#### 修复内容

1. **T/U/V 修复**（3 层协调改动）：
   - [run_evaluation.py:3174-3248](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/utils/run_evaluation.py#L3174)：构造 style_latent_tensor（VAE encode 目标风格参考图）
   - [inference.py:548-551](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/utils/inference.py#L548)：dict 分支提取 style_latent_tensor 传递
   - [model620.py:677](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L677)：新增 N1 可观测性（n1_adain_executed, n1_ep_fiber_abs）

2. **W 修复**：
   - anti_input_margin 从 0.3 增大到 20.0（匹配 dist_input 量级）
   - [losses620.py:636-676](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/losses620.py#L636)：加 W loss debug print

#### 修复后验证

- T1 smoke test: `n1_adain_executed=1.0`, `n1_ep_fiber_abs≈0.3564` ✅
- W2b: dist_input mean=41-57, loss step1=3.90（非零）✅
- 修复后 10 个变体 LPIPS 分布在 0.3735~0.6685（全部激活）

### 2.2 Phase 3 search：参数搜索

#### U 方向（style_extrap_alpha，外推强度）完整结果

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

#### V 方向（patch_adain_kernel，空间核大小）完整结果

| 变体 | k | clip_style | lpips | Δclip vs I7 | Δlpips vs I7 | 击败 I7 |
|------|---|-----------|-------|-------------|--------------|---------|
| V1 | 4 | 0.7242 | 0.5196 | +0.0225 | +0.1571 | no |
| V2 | 8 | 0.7290 | 0.4497 | +0.0273 | +0.0872 | no |
| V3 | 16 | 0.7295 | 0.3963 | +0.0278 | +0.0338 | no |
| V4 | 20 | 0.6334 | 0.5889 | -0.0683 | +0.2264 | no (崩塌) |
| V5 | 24 | 0.6562 | 0.5330 | -0.0455 | +0.1705 | no (崩塌) |
| **V6** | 32 | 0.7262 | 0.3722 | +0.0245 | +0.0097 | **YES** |

**趋势**：非单调。仅 2 幂次 kernel（4/8/16/32）工作正常，非 2 幂次（20/24）崩塌（patch 边界伪影）。k 越大 lpips 越低（空间平滑越强）。

#### T 方向（multiband_adain，频带分离）

| 变体 | mid | hh | clip_style | lpips | n1_ep_fiber_abs |
|------|-----|-----|-----------|-------|-----------------|
| T1 | 0.3 | 0.3 | 0.6518 | 0.6650 | 0.3508 |
| T2 | 0.5 | 0.3 | 0.6574 | 0.6684 | 0.4096 |
| T3 | 0.3 | 0.5 | 0.6587 | 0.6641 | 0.3762 |
| T4 | 0.5 | 0.5 | 0.6609 | 0.6685 | 0.4290 |

**hh 排查结论**（smoke test 验证）：
- hh 0.3→0.5: `n1_hh_final_abs` +38.8%, `clip_style` +0.0072, `content_lpips` -0.0007
- **hh 生效在 clip 维度，不在 lpips 维度** — 是设计如此，非 bug
- 原因：BASE LOCKING 锁死 content lowpass 保 lpips，hh 只作用于 fiber 高频纹理提 clip
- T 方向所有变体 lpips 都很高（0.66+），multiband_adain 对内容损害大

#### W 方向（anti_input_style loss，训练侧）调参结果

| 变体 | margin | clip_style | lpips | Δlpips vs I7 | step=1 loss | step=51+ loss |
|------|--------|------------|-------|--------------|-------------|---------------|
| W2c | 5 | 0.7123 | 0.3580 | -0.0045 | 0.4156 | 0.0 |
| W2d | 10 | 0.7060 | 0.4270 | +0.0645 | 1.2793 | 0.0 |
| W2e | 15 | 0.6946 | 0.4652 | +0.1027 | 2.4753 | 0.0 |
| W2b | 20 | 0.6947 | 0.4645 | +0.1020 | 3.9000 | 0.0 |

**结论**：未找到有效折中点。hinge loss 仅 step=1 生效（模型一步就把 dist_input 推过 margin），后续无梯度。margin=5 是平凡解（等同无正则化），margin≥10 lpips 恶化 +0.06~0.10。

### 2.3 口径验证

**0.7295 确认是 5-style all_pairs_overview（25 对含 identity），非 transfer 子集**。从 25 格 matrix_breakdown 逐格手算均值与 stored 字段对齐到 10 位小数。

| 变体 | all_pairs clip (25对) | transfer clip (20对) | identity clip (5对) |
|------|----------------------|---------------------|-------------------|
| I7 | 0.7017 | 0.6730 | 0.8166 |
| V3(k16) | 0.7295 | 0.7042 | 0.8304 |
| W2b | 0.6947 | 0.6801 | 0.7534 |

---

## 三、理论修正

### 3.1 修正 1：N1 是"fiber 统计匹配"非"风格注入"

**旧理解**：N1 endpoint AdaIN 块是"风格注入"机制，将风格信息注入 content。

**修正理解**：N1 是"fiber 统计匹配"机制，通过 AdaIN 将 content fiber 的 per-channel 统计（μ/σ）对齐到 style fiber。不是注入新信息，而是调整现有 fiber 的分布。

**证据**：
- N1 的核心操作是 `ep_fiber_norm = (ep_fiber_curr - pred_mean) / pred_std` 然后 `ep_fiber_matched = ep_fiber_norm * target_std + target_mean`（[model620.py:865-869](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L865)）
- 这是标准 AdaIN：normalize content，再用 style 统计 denormalize
- N1 不引入新的 style token 或 style embedding，只调整 fiber 的统计分布

### 3.2 修正 2：clip-lpips 权衡是 BASE LOCKING 的必然结果

**旧理解**：clip-lpips 权衡是模型容量的限制，可以通过增加参数或改进训练突破。

**修正理解**：clip-lpips 权衡是 BASE LOCKING 机制的结构性约束。BASE LOCKING 锁死 content lowpass 保 lpips，同时限制 N1 只能影响 fiber（高频）维度。要突破 clip 上限，必须放松 BASE LOCKING，但这会损害 content 保真。

**证据**：
- 所有 U/V/T 变体的 lpips 都 ≥ I7 baseline（0.3625），无一能降低 lpips
- clip 提升伴随 lpips 上升，呈帕累托前沿
- T 方向 hh 参数对 lpips 几乎无影响（Δ=-0.001），因 hh 作用于高频，LPIPS 对高频不敏感

### 3.3 修正 3：U 方向"温和放大"优于"强放大"（style_extrap_alpha 的真实机制）

**旧理解**：style_extrap_alpha 是"外推强度"，α 越大风格迁移越强，效果越好。

**修正理解**：α 的真实机制是 **style_fiber 的全局缩放倍数**（[model620.py:698-699](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L698)）：

```python
if style_extrap_alpha > 0.0:
    style_fiber = style_fiber * (1.0 + style_extrap_alpha)
```

- α=0.1 → style_fiber 放大 1.1 倍（style 的 μ/σ 同比放大）
- α=1.0 → style_fiber 放大 2.0 倍（统计量翻倍）
- 放大后的 style_fiber 进入 AdaIN，让匹配后的 content fiber 统计更"极端"

**关键区分**：U 的 α **不是** α-blend 比例（那是 `endpoint_adain_scale`，L873），而是**输入侧的 style 信号放大**。这是 StyleGAN truncation trick 的反向应用——推向更极端的风格分布。

**机制**：
- 小 α（0.1）：style 统计量温和放大，AdaIN 匹配后的 fiber 略微强化风格方向，content 结构基本不动
- 大 α（1.0）：style 统计量翻倍，AdaIN 强制 content fiber 拉向极端分布，扰动 content 结构
- lpips 恶化速率远快于 clip 下降，说明大 α 的副作用是非线性的（统计量放大引发 fiber 分布偏离自然流形）

**证据**：
- α=0.1→0.5: clip 从 0.7225 降到 0.6959（-2.9%），lpips 从 0.3660 升到 0.4307（+17.7%）
- α=0.1→1.0: clip 从 0.7225 降到 0.6736（-6.8%），lpips 从 0.3660 升到 0.5218（+42.6%）
- lpips/clip 恶化比 ≈ 6:1，说明统计量放大的副作用远大于风格增益

### 3.4 修正 4：V 方向 kernel 必须 2 幂次

**旧理解**：patch_adain_kernel 是连续参数，可以任意设置。

**修正理解**：kernel 必须是 2 幂次（4/8/16/32），非 2 幂次（20/24）导致边界伪影和性能崩塌。

**机制**：
- patch AdaIN 需要将 feature map 分成不重叠的 patch
- feature map 尺寸通常是 2 幂次（32/64/128）
- 非 2 幂次 kernel 不能整除 feature map 尺寸，导致 patch 边界不对齐
- 边界处的统计匹配产生伪影，污染整个 feature map

**证据**：
- k=20: clip 暴跌到 0.6334（-9.8%），lpips 暴涨到 0.5889（+62.3%）
- k=24: clip 0.6562，lpips 0.5330（同样崩塌）
- k=32: 完全恢复，clip=0.7262, lpips=0.3722

### 3.5 修正 5：W hinge loss 的梯度失效问题

**旧理解**：W anti_input_style loss 通过 hinge loss 约束 input style 和 target style 的距离，margin 越大约束越强。

**修正理解**：hinge loss `F.relu(margin - dist_input)` 的梯度在 `dist_input > margin` 时为 0。模型一步就把 dist_input 推过 margin，后续训练步无梯度，loss 失效。

**证据**：
- step=1: loss 非零（0.42/1.28/2.48 对应 margin=5/10/15）
- step=51+: loss 全部归 0（dist_input min 已超过 margin）
- margin=5 的 lpips=0.3580（低于 I7），但这是平凡解（loss 几乎不生效，等同无正则化）

**改进方向**：
1. 降低 `w_anti_input_style` 权重，减弱初始梯度冲击
2. 改用 soft hinge（如 `softplus(margin - dist_input)`）或 KL 散度，使 loss 在 dist_input 接近 margin 时仍有梯度
3. 动态 margin 退火（训练初期大 margin，后期小 margin）

### 3.6 修正 6：hh 和 mid 职责正交

**旧理解**：T 方向的 mid 和 hh 都影响 lpips，hh 参数变化应导致 lpips 变化。

**修正理解**：hh 和 mid 职责正交。hh 作用于对角高频纹理，影响 clip_style（风格纹理迁移）但不影响 content_lpips（content 结构保留）。mid 作用于中频边缘，对 lpips 有一定影响。

**机制**：
- BASE LOCKING 锁死 content lowpass，LPIPS 主要受 lowpass/中频影响
- HH 是 Haar 对角高频分量，LPIPS 对高频不敏感
- N1 设计目标正是"base 锁死保 LPIPS, fiber 获得风格统计提 clip"

**证据**（T1 vs T3 smoke test）：
- hh 0.3→0.5: `n1_hh_final_abs` +38.8%, `n1_hh_contribution_ratio` 30%→36%
- `clip_style` +0.0072（风格增强）
- `content_lpips` -0.0007（几乎不变）

### 3.7 修正 7：U/V 作用点差异（输入侧 vs 计算侧）

**旧理解**：U 和 V 都是 N1 块的"风格强度"参数，作用类似。

**修正理解**：U 和 V 作用在 N1 块的不同位置，机制完全不同：

| 方向 | 作用点 | 代码位置 | 机制 |
|------|--------|---------|------|
| U (style_extrap_alpha) | **输入侧** | L698-699 | 放大 style_fiber 全局缩放，强化 style 统计量 |
| V (patch_adain_kernel) | **计算侧** | L813-841 | 改变 AdaIN 的空间粒度，控制统计匹配的局部性 |
| T (mid/hh_adain_scale) | **输出侧** | L791-792 | α-blend matched fiber 与 original fiber 的混合比例 |

**三层作用点的物理含义**：
- **U（输入侧）**：改变 style 信号本身的强度——"给 style 加增益"
- **V（计算侧）**：改变统计匹配的空间粒度——"全局匹配 vs 局部匹配"
- **T（输出侧）**：改变 matched fiber 的混合比例——"用多少 style 替换 content"

**协同/拮抗预期**：
- U4+V6 联合：U 放大 style 信号（输入侧），V 用大核平滑（计算侧）—— 两者作用点正交，可能叠加
- U4+T 联合：U 放大 style 信号，T 控制 mid/hh 混合 —— 两者作用点正交，但 T 方向 lpips 已 0.66+，叠加可能恶化
- V6+T 联合：V 用大核，T 控制 mid/hh 混合 —— V 作用于整体 patch，T 作用于频带分离，机制冲突（V 不区分频带，T 区分频带）

**证据**：
- U/V 都在帕累托前沿上，但 U4(α0.1) 的 lpips 增量（+0.0035）远小于 V6(k32) 的 lpips 增量（+0.0097）
- 说明输入侧放大（U）的副作用小于计算侧粒度调整（V），因 U 不改变空间结构

### 3.8 修正 8：W 方向的根本性矛盾

**旧理解**：W anti_input_style loss 通过约束 input style 与 target style 的距离，防止模型"偷懒"直接复制 input style。

**修正理解**：W 方向存在**根本性的目标矛盾**：

1. **模型目标**：output → target style（让输出匹配目标风格）
2. **W 约束**：保持 input style ≠ target style（防止 input 被拉向 target）
3. **矛盾**：如果 input style 本身就接近 target style（同风格迁移），W 约束会阻止模型学习正确的迁移方向

**更深层问题**：W 约束的是 input 和 target 的距离，但 input style 是数据集给定的，模型无法改变 input。W 实际上是在**惩罚模型让 output 接近 target**——这与风格迁移的目标直接冲突。

**hinge loss 梯度失效的根因**：
- 模型一步就把 dist_input 推过 margin（step=1 loss 非零，step=51+ loss=0）
- 这不是"约束满足"，而是"模型放弃挣扎"——既然 W 约束与目标冲突，模型选择优先满足主目标（output → target），让 W loss 自然归零
- margin≥10 时 lpips 恶化 +0.06~0.10，说明 W 在 step=1 的强梯度冲击破坏了 content 保真

**改进方向**（基于根本矛盾分析）：
1. **放弃 W 方向**：W 的目标与风格迁移冲突，可能是错误方向
2. **重新定义 W**：约束 output 的统计分布（而非 input-target 距离），让 output 不要过度白化
3. **改用 soft hinge / KL**：如果坚持 W 方向，至少改用连续梯度 loss

**证据**：
- W2c(margin=5): lpips=0.3580（低于 I7），但这是 W loss 几乎不生效的平凡解
- W2d(margin=10): lpips=0.4270（+0.0645），W 在 step=1 的冲击破坏 content
- W2b/W2e(margin=20/15): lpips=0.4645/0.4652（+0.10），冲击更强

### 3.9 修正 9：kernel 2 幂次的几何解释

**旧理解**：V 方向 kernel 必须 2 幂次是"不能整除 feature map 尺寸"的算术问题。

**修正理解**：根因是 **Haar 小波分解的 2 进制层级结构与 patch 分割不对齐**：

**Haar 分解的几何结构**：
- Haar 是 dyadic 小波，每级分解把信号分成 4 个子带（LL/LH/HL/HH），每个子带尺寸减半
- feature map 尺寸 32 → LL/LH/HL/HH 各 16 → 再分解各 8 → ...
- 这个 2 进制层级结构要求所有空间操作与 2 幂次对齐

**patch_adain_kernel 的几何约束**：
- `F.unfold(kernel_size=k, stride=k)` 把 feature map 分成不重叠的 patch（L826-827）
- 每个 patch 独立做 AdaIN（per-patch μ/σ 匹配）
- patch 边界必须与 Haar 子带边界对齐，否则：
  - patch 跨越 Haar 子带边界 → patch 内统计混合不同频带信息
  - AdaIN 匹配破坏频带分离 → 风格统计污染 content 频带
  - 重构后产生 patch 边界伪影

**为什么 k=20/24 崩塌而 k=32 正常**：
- feature map 尺寸 32（或 64/128）
- k=20: 32/20=1.6，不能整除，最后 12 像素是残留 patch → 边界伪影
- k=24: 32/24=1.33，不能整除，最后 8 像素是残留 patch → 边界伪影
- k=32: 32/32=1，整除，单 patch 覆盖全部 → 等同全局 AdaIN（最平滑）
- k=16: 32/16=2，整除，4 个 patch → 局部统计但边界对齐

**更深的几何洞察**：
- k=32（单 patch）= 全局 AdaIN = 最强空间平滑 = lpips 最低（0.3722）
- k=16（4 patch）= 局部统计 = 风格更精细 = clip 最高（0.7295）
- k=8（16 patch）= 更局部 = 风格过精细 = clip 下降（0.7290）+ lpips 上升（0.4497）
- k=4（64 patch）= 最局部 = 统计噪声 = clip 下降（0.7242）+ lpips 暴涨（0.5196）

**最优 kernel**：k=16 是"风格-内容平衡点"——足够局部以捕捉风格纹理（clip 峰值），又足够全局以保持统计稳定性（lpips 可接受）。

---

## 四、帕累托前沿分析

### 4.1 clip-lpips 帕累托前沿

基于所有击败 I7 的点，绘制帕累托前沿：

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

| 帕累托点 | clip_style | lpips | 特点 |
|---------|-----------|-------|------|
| I7 baseline | 0.7017 | 0.3625 | 最低 lpips |
| **U4(α0.1)** | 0.7225 | 0.3660 | **最佳综合点**（clip +2.97%, lpips +0.97%）|
| V6(k32) | 0.7262 | 0.3722 | 更高 clip，lpips 稍增 |
| V3(k16) | 0.7295 | 0.3963 | 最高 clip，lpips 较高 |

### 4.2 最佳点推荐

**U4(α0.1) 是最佳综合点**：
- clip 提升 +2.97%（0.7017→0.7225）
- lpips 仅增 +0.97%（0.3625→0.3660）
- 几乎无副作用的风格增强，适合作为默认推理配置

**V6(k32) 是 clip 增益最大的点**：
- clip 提升 +3.49%（0.7017→0.7262）
- lpips 增 +2.67%（0.3625→0.3722）
- 如果能接受 lpips +0.01，V6 提供更高的风格迁移强度

---

## 五、下一步方向

### 5.1 U4+V6 联合实验（推荐优先）

U4(α0.1) 与 V6(k32) 位于帕累托前沿相邻位置，且作用点正交（U 输入侧 / V 计算侧，见 3.7），探索联合是否产生协同效应：
- 配置：style_extrap_alpha=0.1 + patch_adain_kernel=32
- **基于作用点分析的预期**：
  - U 放大 style 信号（输入侧），V 用大核平滑（计算侧）—— 两者不冲突
  - clip 可能叠加（+5-6%）：U 的统计放大 + V 的全局匹配
  - lpips 可能叠加（+1.2-1.5%）：U 的副作用（+0.0035）+ V 的副作用（+0.0097）
  - **风险**：V6 的全局匹配可能"吸收"U 的统计放大（大核平滑掉 style 信号的增益），导致 U 的边际效益递减
- 方法：从 I7 checkpoint 生成联合变体，评估

### 5.2 更小 α 探索

U4(α0.1) 已接近无副作用区间，探索 α=0.05/0.08：
- 预期：clip 增益可能进一步降低，但 lpips 可能低于 I7
- 如果 α=0.05 时 lpips < 0.3625 且 clip > 0.7017，则突破帕累托前沿
- **理论依据**：α 越小，style 统计量放大越温和，fiber 分布偏离自然流形的程度越小

### 5.3 W 方向重新评估（基于根本矛盾分析）

基于 3.8 的根本矛盾分析，W 方向需要重新评估：

**选项 A：放弃 W 方向**（推荐）
- W 约束 input-target 距离与风格迁移目标冲突
- 即使改 soft hinge，根本矛盾不解决
- 算力应优先投入 U/V 联合和 BASE LOCKING 放松

**选项 B：重新定义 W 为 output 统计约束**
- 不约束 input-target 距离，而是约束 output 的统计分布
- 目标：防止 output 过度白化（WFI > 0.40）
- 方法：约束 output fiber 的 std 不要过低（白化的统计特征）

**选项 C：soft hinge 改造**（如果坚持原 W 方向）
- `softplus(margin - dist_input)` 替代 `F.relu(margin - dist_input)`
- 目标：loss 在 dist_input 接近 margin 时仍有梯度
- 但根本矛盾未解决，预期效果有限

### 5.4 I7 基础训练到 5 epoch

当前 U4 继承自 I7 epoch_0002 checkpoint。探索 I7 训练到 5 epoch 后再应用 U4：
- **关键澄清**：U 是推理期参数，不需重新训练。这里指**重新训练 I7 到 5 epoch**，然后在新 I7 checkpoint 上应用 U4(α0.1)
- 预期：I7 训练更充分 → velocity 预测更准 → U4 的 style 放大建立在更优基础上
- 风险：过拟合可能导致 lpips 恶化（参考 E4-long epoch 5 是最佳停止点）
- **依据**：历史经验显示 epoch 5 是自然收敛的最佳停止点

### 5.5 BASE LOCKING 部分放松探索

当前 BASE LOCKING 完全锁死 content lowpass。探索部分放松：
- 使用 tri_band_lock 模式（L945-955），允许 mid 频带部分 blend
- 调整 `tri_band_edge_alpha` 控制放松程度
- 目标：放松 base 锁死可能突破 clip 上限，但需监控 lpips
- **理论依据**：三层纤维动力学（1.6）表明 mid 对 lpips 有影响（T 方向 Δ=0.0034），放松 mid 是"可控风险"的放松
- **配置建议**：tri_band_edge_alpha 从 0.0（完全锁死）逐步增加到 0.1/0.2/0.3，监控 clip 和 lpips

### 5.6 hh 优先策略（基于三层纤维动力学）

基于 1.6 的三层纤维动力学和 3.6 的 hh/mid 正交性：

**策略**：优先用 hh 参数提 clip，mid 参数保持保守
- hh 作用于 LPIPS 不敏感的频带，是"免费"的 clip 提升
- mid 作用于 LPIPS 敏感的频带，需要谨慎

**具体方案**：
- 配置：multiband_adain_mode='two_level' + hh_adain_scale=0.5~0.7 + mid_adain_scale=0.1~0.2
- 预期：hh 提 clip（+0.007+），mid 保 lpips
- **与 U4 联合**：U4(α0.1) + hh_adain_scale=0.5，输入侧放大 + 高频风格强化
- **风险**：T 方向所有变体 lpips 都 0.66+，需验证单独提高 hh（不提高 mid）是否能控制 lpips

---

## 六、工程教训

### 6.1 "代码已写" ≠ "功能已生效"

**教训**：Phase 3 deepfix 前的 9 个变体 LPIPS 全部 0.4180（N1 死路径），但代码中确实写了 N1 块。必须有运行时 probe 验证（`n1_adain_executed=1.0`）才能确认开关生效。

**改进**：新增 probe gate（[run_rtuv_eval.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/exp/625_fc_sb/run_rtuv_eval.py)），评估后自动检查 `n1_adain_executed`，失败标记 INVALID。

### 6.2 observability 指标语义必须明确

**教训**：`model_endpoint_style_high_abs` 被误认为是 N1 块的执行指标，实际测量的是 `forward()` 的 endpoint head 投影层（L518-520），与 N1 块无关。

**改进**：为 N1 块新增独立的可观测性（`n1_adain_executed`, `n1_ep_fiber_abs`, `n1_hh_final_abs` 等），指标命名要明确语义。

### 6.3 远程环境路径陷阱

**教训**：远程 SSH shell 是 cmd.exe（非 bash），`/mnt/i/...` 路径会被误解析为 `C:\mnt\i\`，导致 SCP 文件落到错误位置。

**改进**：远程路径统一用 Windows 风格 `I:\...`，SCP 目标用 `I:/...`。

### 6.4 hinge loss 的梯度失效

**教训**：W hinge loss `F.relu(margin - dist_input)` 在模型一步推过 margin 后梯度归零，后续训练无效果。看似 loss 值很低（接近 0），实际是梯度失效而非约束满足。

**改进**：训练时监控 loss 的梯度范数（非 loss 值），或改用 soft hinge 确保梯度连续。

---

## 附录：关键文件索引

### 代码文件
- [src/model620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py)：N1 块（L676-847）、BASE LOCKING（L943-957）、integrate_transport（L553）
- [src/losses620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/losses620.py)：W loss（L636-676）
- [src/utils/inference.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/utils/inference.py)：style_latent_tensor 传递（L548-551）
- [src/utils/run_evaluation.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/utils/run_evaluation.py)：style_latent_tensor 构造（L3174-3248）

### 实验脚本
- [exp/625_fc_sb/gen_i7_direction_configs.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/exp/625_fc_sb/gen_i7_direction_configs.py)：变体 checkpoint 生成
- [exp/625_fc_sb/run_rtuv_eval.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/exp/625_fc_sb/run_rtuv_eval.py)：批量评估（含 probe gate）
- [exp/625_fc_sb/run_w_batch.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/exp/625_fc_sb/run_w_batch.py)：W 批量训练（含 config 校验）

### Spec 文档
- [.trae/specs/fc-sb-phase3-deepfix/](file:///g:/GitHub/Latent_Style/SchrodingerBridge/.trae/specs/fc-sb-phase3-deepfix/)：开关修复 spec
- [.trae/specs/fc-sb-phase3-search/](file:///g:/GitHub/Latent_Style/SchrodingerBridge/.trae/specs/fc-sb-phase3-search/)：参数搜索 spec

### 实验数据
- I7 baseline: `exp/625_fc_sb/from_scratch_win/init_I7/epoch_0002.pt`
- U/V 变体: `exp/625_fc_sb/from_scratch_win/rtuv_variants/`
- W 变体: `exp/625_fc_sb/from_scratch_win/w_W2b/` ~ `w_W2e/`
- 评估结果: `exp/625_fc_sb/from_scratch_win/rtuv_variants/<name>_eval_v2/summary.json`
