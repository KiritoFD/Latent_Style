# docs/math — FC-SB 理论描述

> FC-SB (Frequency-Conditioned Schrödinger Bridge) 的完整理论体系。
> 主参考: [docs/72/02_theory.md](../72/02_theory.md) (含公式推导) + 本文档的精炼索引版。
> 配套文档: [docs/baseline/README.md](../baseline/README.md) (baseline 评估), [docs/exp/experiment_audit.md](../exp/experiment_audit.md) (实验脉络)。

---

## 0. 理论谱系一览

| 层级 | 文档 | 内容 |
|------|------|------|
| **核心理论 (本文)** | `docs/math/README.md` | FC-SB 完整理论框架 (本文 §1-§11) |
| 详细公式 | `docs/72/02_theory.md` | Schrödinger Bridge SDE + Haar DWT 推导 + 1:8 trade-off 系统证明 |
| 历史 OT 探索 | `docs/archive/620/OT.md` + `docs/archive/620/math.md` | Phase OT 失败分析 (已废弃, 仅供历史追溯) |
| Fiber Bundle | `docs/archive/612-phase2/FIBER_BUNDLE_DESIGN.md` | 早期 fiber bundle 设计 (未采用, 历史) |
| 619 理论验证 | `docs/archive/619/model/04_theoretical_validation_and_gradient_dynamics.md` | 早期梯度动态分析 (pre-FC-SB) |
| 622 统一模型 | `docs/archive/622/history/10_unified_mathematical_model.md` | 622 时代统一数学模型 (已被 FC-SB 取代) |

---

## 1. 问题定义

**任务**: 给定源域 content latent `x_0 ∈ R^{4×32×32}` (SDXL VAE 编码) 与目标风格 `s ∈ {1,…,5}`, 生成 `x_1` 使其:
- **内容保真**: `x_1` 与 `x_0` 在感知结构上一致 (LPIPS 低)
- **风格一致**: `x_1` 与 style `s` 的参考图分布在 CLIP 嵌入空间中一致 (CLIP-S 高)

**双超目标** (论文用):
- `all_pairs_clip > 0.7319` (4F.1 远程 SOTA clip)
- `all_pairs_lpips < 0.3068` (4J.1 本地 DWT route lpips)

**评价指标**:
| 指标 | 计算 | 含义 |
|------|------|------|
| `all_pairs_clip_style` | 5×5 风格矩阵 (150 src × 5 tgt) 平均 CLIP 余弦相似度 | 风格相似度 (越高越好) |
| `all_pairs_content_lpips` | 同矩阵 LPIPS(AlexNet) 平均 | 内容差异 (越低越好) |

详见 [docs/tools/README.md §2 评估协议](../tools/README.md)。

---

## 2. Schrödinger Bridge 框架

### 2.1 一般理论

Schrödinger Bridge 寻找在两个分布 `π_0` (content) 和 `π_1` (style) 之间的最可能路径, 即 entropy-regularized optimal transport:

```
SB* = argmin_{π: π_0→π_1} KL(π || Brownian motion prior)
```

解是随机过程 `{X_t}_{t∈[0,1]}`, 满足 SDE:
```
dX_t = v(X_t, t) dt + σ(t) dW_t
```
其中 `v` 是 velocity field, `σ` 是噪声调度。

### 2.2 本项目的简化 (deterministic Flow Matching)

FC-SB 采用 **deterministic Flow Matching (FM)** 简化 (`σ=0`):
- 训练目标: `v_target = x_1 - x_0` (time-independent)
- 插值路径: `x_t = (1-t)·x_0 + t·x_1`
- 推理: ODE 积分 `dx/dt = v_θ(x, t, s)`, 从 `x_0` 到 `x_1`

**4I.10 Probe 诊断**: velocity field 在 `t=0.5` (轨迹中点) 几乎完全死亡 (cos similarity ≈ 0.01)。FM 固有歧义: 中点方向不确定, 因为 `x_0.5 = 0.5·content + 0.5·target`。**模型本质是端点校正器, 而非 ODE**。

---

## 3. Haar DWT 频域解耦

### 3.1 Haar 变换

单级 2D Haar DWT 将 latent `[B,4,32,32]` 分解为 4 子带 `[B,4,16,16]`:

```
inv_sqrt2 = 1/√2
LL = (a + b + c + d) / 2     # 低频 (平均)
LH = (a + b - c - d) / 2     # 垂直高频
HL = (a - b + c - d) / 2     # 水平高频
HH = (a - b - c + d) / 2     # 对角高频
```

### 3.2 正交性 (关键性质)

Haar 矩阵 `H = [1,1;1,-1]/√2` 是正交矩阵: `H·H^T = I`。

**推论**:
1. `IDWT(DWT(x)) = x` (无信息损失)
2. 能量守恒: `||LL||² + ||LH||² + ||HL||² + ||HH||² = ||x||²`
3. 各子带统计量独立 (理论上)

这是 per-subband AdaIN/WCT 统计隔离的理论基础。

### 3.3 多级分解

| Level | LL 尺寸 | 含义 | 实验 |
|-------|---------|------|------|
| 1 | 16×16 | 标准分解 | 4B baseline |
| 2 | 8×8 | 中尺度结构 | 4D BREAKTHROUGH |
| **3** | **4×4** | **SOTA (4F.1)** | **clip=0.7319** |
| 4 | 2×2 | 过激, 丢位置信息 | 4F.2 FAIL |

**4F 趋势**: 1→2 (+0.0040 clip), 2→3 (+0.0018), 3→4 (-0.0003)。**3-Level 是峰值**。

### 3.4 子带的物理意义

| 子带 | 信息 | 风格贡献 | 内容贡献 |
|------|------|----------|----------|
| **LL** | 全局色调、光照、色相 | **高** (+0.014 clip, 4G.1a vs 4G.1b) | **高** (lpips 锚) |
| **LH/HL** | 边缘、纹理、笔触方向 | 中 | 中 |
| **HH** | 噪点、细节、对角纹理 | 中 | 低-中 |

**4G.1 关键发现**: LL 不是纯内容锚, 是"内容 + 全局风格"的混合载体。
- LL velocity 应用: +0.0141 clip
- LL velocity 训练: -0.0091 lpips (梯度回流改善 backbone 内容理解)

### 3.5 HH 删除 (628 L8)

训练 `head_hh` 与不训练 clip 差异 Δ=±0.0001 (DEAD)。HH velocity head 在 SOTA 配置中删除, 模型输出仅 `{ll, lh, hl}`。

---

## 4. 频域解耦架构

### 4.1 整体数据流

```
x [B,4,32,32]
    ↓ dwt2_haar
LL, LH, HL, HH [B,4,16,16] each
    ↓ stack along channel
[B,16,16,16]
    ↓ input_proj (Conv2d 16→dim=64)
h [B,64,16,16]
    ↓ 4× SpatialBridgeBlock620 (Self-Attn → Cross-Attn → FFN)
h [B,64,16,16]
    ↓ 3× SpectralVelocityHead (zero-init Conv2d)
{v_ll, v_lh, v_hl}  # HH removed
```

### 4.2 SpectralBridgeBlock620

每个 block 的前向:
```
x → norm1 → AdaLN(time_emb) → Self-Attention → +x
  → norm2 → Cross-Attention(content × style_memory) → tanh_gate → +α·x
  → norm3 → FFN(Conv1×1 → SiLU → Conv1×1) → +x
```

- **AdaLN**: 时间条件通过 Adaptive Layer Norm 注入 (调制 mean)
- **Cross-Attention Gate**: `style_cross_attn_gate_init=0.05`, 输出经 `tanh` 门控后残差注入
- **Attention Mode**: `relu2` (active): `gates = relu(q·kᵀ·scale/temp)²`, 比 softmax 更稀疏激活

### 4.3 三个独立 Velocity Head

`SpectralVelocityHead` 是 zero-init Conv2d, 输出 `{v_ll, v_lh, v_hl}`。

**独立训练目标** (`SpectralODEObjective620`):
```
loss = w_ll · FM(v_ll_pred, v_ll_target) +
       w_lh · FM(v_lh_pred, v_lh_target) +
       w_hl · FM(v_hl_pred, v_hl_target)
```

**T11 配置**: `w_ll=0.0, w_lh=1.0, w_hl=1.0`
- LL 不训练 → LL 自由漂移 (clip 最佳)
- LH/HL 训练 → 中频风格传输

**T18 验证**: 恢复 `w_ll>0` 是 content-heavy trade-off (lpips 降但 clip 也降)。T11 `w_ll=0.0` 是 clip 最优点。

---

## 5. Style Conditioning

### 5.1 Learnable Style Memory

```python
self.style_memory = nn.Parameter(randn(num_styles=5, 256, 384) * 0.02)
```

5 个风格各 256 tokens × 384 dim。`patch_proj`/`cls_proj` 投影到 64 dim 作为 cross-attention 的 K/V。

### 5.2 "Style Is Learned, Not Extracted" (4C 核心)

**4C 实验**: 用 DINOv2 提取 reference 图特征作为 style condition → clip -0.018 (FAIL)。

**根因**: DINOv2 是 content-biased (物体语义污染风格)。learnable style_memory 通过端到端训练学到任务最优的风格表征, 不被 content 信号污染。

**结论**: 本项目所有 style 信号来自 `style_memory`, 不用外部模型。

### 5.3 Masking (Phase 2, Blindfolded Tokenizer)

- `random`: 随机 dropout tokens (信息瓶颈, 强迫 style_memory 学更鲁棒表征)
- `shuffle`: 空间打乱 (破坏位置信息)
- **Phase 2 最优**: `random_50` (mask 50% tokens)

### 5.4 Frequency Masking (Phase 4B)

- `avg_pool`: box 低通, 减去低频 (4B-1)
- `haar_dwt`: DWT 后 LL 子带 ×(1-α), IDWT 重建 (4B-3)
- 两者效果相当, 但 `haar_dwt` 提供正交分解的理论纯净性

---

## 6. DWT Route Cross-Attention (4J.1 核心)

### 6.1 设计动机

**问题**: 标准 cross-attention 让所有空间位置 (含 LL) query style_memory。style_memory 被迫学"维持结构", 分散了表达笔触/色彩的能力。

**解决**: 对特征图做 DWT, **LL bypass**, 仅 LH/HL/HH tokens query style_memory。

### 6.2 实现

```python
# blocks620.py, SpatialBridgeBlock620.forward()
if use_dwt:
    ll_f, lh_f, hl_f, hh_f = dwt2_haar(x_f)
    # LL bypass: 不参与 cross-attention query
    ca_in = torch.cat([lh_tokens, hl_tokens, hh_tokens], dim=1)  # 仅高频
else:
    ca_in = x.flatten()  # 全空间 query

q = self.q_proj(ca_in)
k = self.k_proj(style_tokens)
v = self.v_proj(style_tokens)
attended = attention(q, k, v)

if use_dwt:
    # IDWT 重建: LL 保持原值, 高频被 cross-attn 输出替换
    attended_2d = idwt2_haar(ll_f, lh_out, hl_out, hh_out)
else:
    attended_2d = attended.reshape(b, c, h, w)
```

### 6.3 理论收益

style_memory 100% 容量表达笔触/色彩, 不被迫学"维持结构"。

### 6.4 架构固有 1:8 trade-off (T5-T12 系统性证明)

**核心矛盾**: CLIP-S 衡量整体风格 (含低频), 但 LL bypass 阻止 style_memory 影响低频结构。

T5/T10/T11/T12 共 15 个配置系统性证明: 在当前 DWT route 架构下, 风格注入和内容保护存在固有的 1:8 trade-off (clip 每提升 1 单位, lpips 损失 8 单位)。要达成双超目标需要 trade-off 比 ≤ 1:1.9, **架构上不可达**。

---

## 7. Stochastic DWT Route (T11 核心)

### 7.1 设计动机

**T5 失败根因**: 训练时全空间 query, 推理时 DWT route → q_proj 输入分布严重不匹配 (clip=0.7061 FAIL)。

**T10 (p=0.5) 失败根因**: 50% 概率看 DWT 系数仍不足以让 q_proj 精通 DWT 分布。q_proj 倾向于"平均"两种分布 (clip=0.7083 FAIL)。

### 7.2 T11 (p=0.8) 设计

```python
if self.training and self.dwt_route_train_prob > 0.0:
    use_dwt = self.dwt_route and (torch.rand(1).item() < self.dwt_route_train_prob)
else:
    use_dwt = self.dwt_route  # 推理始终 DWT route
```

**`p=0.8` 的含义**: 训练时 80% 步用 DWT route, 20% 用全空间 query。
- **80% DWT** → q_proj 精通 DWT 系数分布 → 推理时 DWT route 有效
- **20% 全空间** → style_memory 学到更完整风格表达 (不被高频偏向完全主导)

### 7.3 T11 结果

- clip=0.7213 (本地 SOTA, 差 4F.1 远程目标 0.7319 共 0.0106)
- lpips=0.2868 (**PASS** 首次低于 0.3068 目标, 余量 0.0200)

### 7.4 p 扫描趋势

| p | clip | lpips | 备注 |
|---|------|-------|------|
| 1.0 (4J.1) | 0.7226 | 0.3068 | DWT route 起点 |
| **0.8 (T11)** | **0.7213** | **0.2868** | **本地 SOTA** |
| 0.5 (T10) | 0.7083 | 0.2480 | lpips BEST |
| 0.0 (T5) | 0.7061 | 0.2606 | clip FAIL |

clip 随 p 增大而升 (q_proj 越精通 DWT), 上限在 4J.1 的 0.7226 附近。

---

## 8. Endpoint AdaIN / WCT

### 8.1 AdaIN (mean+std matching)

```python
def _adain_match_subband(content, style):
    μ_c, σ_c = content.mean(dim=[2,3]), content.std(dim=[2,3])
    μ_s, σ_s = style.mean(dim=[2,3]), style.std(dim=[2,3])
    return (content - μ_c) / (σ_c + eps) * σ_s + μ_s
```

只匹配对角协方差, 丢失通道相关性。

### 8.2 WCT (完整协方差匹配)

```python
def _wct_match_fiber(content, style):
    μ_c, Σ_c = content.mean(...), cov(content)  # 完整协方差
    μ_s, Σ_s = style.mean(...), cov(style)
    Σ_c_reg = Σ_c + eps * I  # 对角线正则化避免奇异
    Σ_s_reg = Σ_s + eps * I
    whitened = (content - μ_c) @ Σ_c_reg^{-1/2}
    return whitened @ Σ_s_reg^{1/2} + μ_s
```

WCT 捕获通道相关结构, 理论上比 AdaIN 更精准。

**T19a 数值稳定性修复**: depth=6 时协方差矩阵病态, `eigh` 分解失败。修复: 对角线正则化 `Σ + eps·I` + try-except 回退到 AdaIN。

### 8.3 per_subband_wct 模式 (推理末步)

```python
if adain_mode == "per_subband_wct":
    ll, lh, hl, hh = dwt2_haar(h)
    s_ll, s_lh, s_hl, s_hh = dwt2_haar(style_latent)
    # LL 不动 (内容锚), 高频做 WCT
    lh_new = (1-α)·lh + α·_wct_match_fiber(lh, s_lh)
    hl_new = (1-α)·hl + α·_wct_match_fiber(hl, s_hl)
    hh_new = (1-α)·hh + α·_wct_match_fiber(hh, s_hh)
    h = idwt2_haar(ll, lh_new, hl_new, hh_new)
```

T11 配置: `endpoint_adain_mode=per_subband_wct, scale=0.5`。

### 8.4 EOTA (End-of-Trajectory AdaIN, 4H.1)

**4G.2b 发现**: 多步 Euler 迭代累积使 α=0.5≡α=1.0。残差 `r_n = (1-α)^n`, 对于 n=12 步、α=0.5: `r = 0.5^12 = 0.024%`, α 参数被迭代累积 invalidate。

**EOTA 解决方案**: `only_last_step=True`, 仅在第 8 步应用 AdaIN。解耦 ODE 求解与风格注入, 恢复 α 作为有效 trade-off 旋钮。

**理论意义**: 匹配 SB 理论, 风格是 terminal condition 而非 per-step perturbation。

---

## 9. ODE 求解器

### 9.1 三种求解器

| 求解器 | 阶数 | 截断误差 | 前向调用/步 | 公式 |
|--------|------|----------|-------------|------|
| Euler | 1 | O(h²) | 1 | `h_{i+1} = h_i + v(h_i)·dt` |
| **Heun** | **2** | **O(h³)** | **2** | predictor `h_pred = h_i + v(h_i)·dt`; corrector `h_{i+1} = h_i + (v(h_i)+v(h_pred))/2·dt` |
| RK4 | 4 | O(h⁴) | 4 | 经典四阶 Runge-Kutta |

### 9.2 4I.2 结构性突破

**Euler → Heun 是结构性 DOF** (打破 1D Pareto 前沿):
- 4I.2a (Heun 3ep) ≈ 4H.1g-5ep (Euler 5ep): 高阶 solver 的精度增益 ≈ +2 epochs 训练
- 4I.2b (Heun 5ep): clip +0.0015, lpips -0.0052 vs 4H.1g SOTA — **双提升**

**Heun → RK4 饱和** (4I.6): 其他误差源 (训练噪声、AdaIN 离散化、velocity field 精度) 主导。

### 9.3 复合效应

Heun 的精度优势随训练时长**复合增长** (非静态):
- Euler 3ep→5ep: lpips 仅降 -0.0002 (饱和)
- Heun 3ep→5ep: lpips 降 -0.0050 (**25x 更多**)

---

## 10. Time Schedule

### 10.1 四种 schedule

```python
def _schedule(s):
    if time_schedule == "cosine":
        return (1 - cos(π·s)) / 2          # S 形, 两端慢中间快
    elif time_schedule == "warp_cos":
        return (1 - cos(π·s^p)) / 2        # 参数化 cosine
    elif time_schedule == "quad":
        return s·s                           # 内容偏置
    elif time_schedule == "rquad":
        return 1 - (1-s)·(1-s)              # 风格偏置
    return s                                 # linear (T11 使用)
```

### 10.2 4I.5/4I.8 分类 (关键理论贡献)

| Schedule | clip | lpips | 偏置 |
|----------|------|-------|------|
| cosine | 0.7272 (4I.7b) | 0.3218 | 内容偏置 |
| linear | 0.7266 (4I.2b) | 0.3229 | 中性 |
| rquad | 0.7293 (4I.5c) | 0.3429 | 风格偏置 |
| warp_cos p=0.8 | 0.7282 (4I.8b) | 0.3271 | 轻度风格偏置 |

**理论分类**:
- **结构性 DOF** (打破 Pareto 前沿): solver order (Euler→Heun)
- **Pareto-mapping knob** (沿前沿移动): schedule shape, alpha, training duration, mask ratio, loss weights, model capacity

**所有非 solver-order 的自由度都映射到同一 1D Pareto 前沿**。这是 Phase 4I 的核心理论贡献。

---

## 11. 理论总结

### 11.1 三层频域解耦

```
LL velocity (全局色调/光照)         +0.014 clip (4G.1a vs 4G.1b)
    +
per-subband WCT (笔触/色彩/噪点)     LL 不动, LH/HL/HH 独立 WCT
    +
Spectral ODE (频域速度场)            3 个独立 head, w_ll=0 让 LL 自由漂移
    =
FC-SB 完整解耦架构
```

### 11.2 关键理论洞察

1. **Style Is Learned, Not Extracted** (4C): learnable style_memory 优于外部 DINOv2 特征
2. **LL Is Not Pure Content Anchor** (4G.1): LL 携带 +0.014 clip 的全局风格信息
3. **Numerical Precision as Structural DOF** (4I.2): solver order 是独立于 α 的结构自由度
4. **Schedule Shape is Pareto-Mapping Knob** (4I.5/4I.8): 沿前沿移动, 不打破前沿
5. **Stochastic DWT Route** (T11): p=0.8 让 q_proj 精通 DWT 同时 style_memory 学完整风格
6. **EOTA Restores Alpha Effectiveness** (4H.1): 解耦 ODE 求解与风格注入
7. **1:8 Trade-off Inherent to DWT Route** (T5-T12): CLIP-S 看低频, LL bypass 阻止低频风格注入

### 11.3 未解决的根本矛盾

**双超目标 vs 架构限制**: CLIP-S 衡量整体风格 (含低频色调), 但 DWT route 的 LL bypass 阻止 style_memory 影响低频结构。在当前架构下, clip 上限被锁在 ~0.7226 附近。突破需要:
- (A) 全新架构让风格注入和内容保护完全解耦
- (B) 独立的全局风格信号源 (不通过 DWT route)
- (C) 调整双超目标阈值

---

## 12. 历史理论文档索引

以下文档保留作为历史追溯, 但**其结论已被 FC-SB 取代**:

| 文档 | 时代 | 状态 |
|------|------|------|
| `docs/archive/620/OT.md` | 620 OT 时代 | 已废弃 (OT 失败, 转 SB) |
| `docs/archive/620/math.md` | 620 OT 数学 | 已废弃 |
| `docs/archive/620/theory/` | 620 fog 理论 | 已废弃 (fog 实验失败) |
| `docs/archive/612-phase2/FIBER_BUNDLE_DESIGN.md` | 612 fiber bundle | 未采用 |
| `docs/archive/619/model/04_theoretical_validation_and_gradient_dynamics.md` | 619 早期 | pre-FC-SB |
| `docs/archive/622/history/10_unified_mathematical_model.md` | 622 统一模型 | 已被 FC-SB 取代 |
| `docs/archive/622/FC.md` | 622 FC 早期 | 已被 FC-SB 整合 |

---

**最后更新**: 2026-07-03 (M24, docs/math 框架建立)
**主参考**: [docs/72/02_theory.md](../72/02_theory.md)
**维护原则**: 理论修正同步更新本文档 + 02_theory.md, 历史文档保留不删
