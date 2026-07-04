# Spectral ODE Bridge：架构设计理论基础

> 基于 `docs/archive/theory/SpectralODE_Bridge.md`、`docs/72/02_theory.md` 与 `docs/archive/620/fog/theory/*.md` 的提炼。

---

## 一、核心问题：为什么传统 Flow Matching 不够？

标准 Flow Matching（FM）在欧氏 latent 空间中把 content 与 style 的所有频率成分同等对待。这带来一个本质冲突：

- **低频（LL）** 承载内容结构、全局色调、光照，需要**小幅度移动**；
- **中高频（LH/HL/HH）** 承载笔触、纹理、边缘方向，需要**大幅度移动**。

统一的速度场无法同时满足"保内容"与"换风格"。Spectral ODE Bridge 的解决思路是：**把运输问题搬到小波域，让不同频段拥有独立的速度场与损失权重**。

---

## 二、Haar DWT 子带拆分与信息分工

单级 2D Haar DWT 将 latent $x \in \mathbb{R}^{B \times C \times H \times W}$ 正交分解为 4 个子带：

| 子带 | 数学含义 | 承载信息 | 在模型中的角色 |
|------|----------|----------|----------------|
| **LL** | $(a+b+c+d)/2$ | 低频平均：全局色调、光照、色相、大体结构 | "内容锚"，但也携带显著全局风格信息（4G.1：开启 LL velocity 可提升 clip +0.014） |
| **LH** | $(a+b-c-d)/2$ | 垂直高频：垂直边缘、纹理方向 | 中频风格传输主通道 |
| **HL** | $(a-b+c-d)/2$ | 水平高频：水平边缘、轮廓 | 中频风格传输主通道 |
| **HH** | $(a-b-c+d)/2$ | 对角高频：对角纹理、噪点、细粒度细节 | 628 L8 实验显示其 velocity head **DEAD**（Δclip = ±0.0001），已被移除 |

Haar 变换的关键性质是**正交性**与**能量守恒**：

- $\text{IDWT}(\text{DWT}(x)) = x$（无信息损失）
- $\|LL\|^2 + \|LH\|^2 + \|HL\|^2 + \|HH\|^2 = \|x\|^2$

正交性保证了各子带的统计量相互独立，使得 per-subband AdaIN/WCT 与 per-subband FM loss 在数学上是合理的：在 DWT 域独立调制不同频段，等价于在原空间做频域解耦的风格注入。

---

## 三、Shared Backbone + 独立 Velocity Heads

模型先将 4 个子带沿通道堆叠，通过 `input_proj` 送入共享的 `SpatialBridgeBlock620` 主干，再由 3 个独立的 `SpectralVelocityHead` 分别输出 $v_{\text{LL}}, v_{\text{LH}}, v_{\text{HL}}$。

**为什么有效？**

1. **共享主干**学习 content 与 style 的共享表征，让网络"看懂"源图与目标风格的关系；
2. **独立 heads**避免不同频段的速度场互相耦合。低频可以学得慢、高频可以学得快，互不干扰；
3. **zero-initialization**（标准差 $10^{-3}$、偏置为 0）让模型在训练初期近似 identity 映射，防止早期训练不稳定；
4. 独立输出也支持独立的损失权重配置，例如 `w_ll=0` 可以让 LL 自由漂移以获得最佳 CLIP-S。

---

## 四、Endpoint AdaIN / WCT 的理论依据

AdaIN 与 WCT 的本质是**统计匹配**：

- **AdaIN**：仅匹配 mean/std，相当于对角协方差匹配；
- **WCT**：通过 $Σ^{-1/2}$ 白化再乘以 $Σ^{1/2}$ 着色，实现完整协方差匹配。

在 Spectral ODE Bridge 中，它们被用作 **terminal condition** 而非每步扰动：

1. 将 latent 拆分为 base（低频 LL）与 fiber（高频）；
2. base 锁定 content，fiber 与目标风格的 fiber 做一阶或二阶统计匹配；
3. 通过 $s_{\text{adain}}$ 控制 base/fiber 的混合比例。

这与 Schrödinger Bridge 的理论一致：风格迁移应作为端点条件注入，而不是在 ODE 轨迹中反复扰动。`tri_band` 路径进一步明确：LL 锁定 content、Mid α-blended、HH 完全来自 target，确保运输集中在风格相关频段。

---

## 五、ReLU² Attention 对 Softmax 的改进

标准 cross-attention 使用 softmax：

$$\text{CA}(x, S) = \text{softmax}\left(\frac{Q(x)K(S)^T}{\sqrt{d}}\right) V(S)$$

在 620 系列分析中发现 softmax 存在三个数学问题：

1. **归一化导致信息稀释**：256 个 token 的权重几乎均匀（entropy ≈ 99.9% 均匀），输出接近 $\bar{V}(S)$；
2. **Query 与 style 无关**：$Q = W_Q x$，注意力权重对不同 style 几乎相同；
3. **条件期望坍缩**：模型学到边缘期望 $\mathbb{E}_s[v]$，而非条件于特定 style 的速度场。

ReLU² attention 用 $ \text{relu}(q k^T / \text{temp})^2 $ 替代 softmax，具有：

- **稀疏激活**：只保留高相似度的 token，避免平均化；
- **无归一化约束**：输出幅度由相似度本身决定，style 信号不被稀释；
- **更清晰的梯度路径**：style 差异能更直接地影响输出。

---

## 六、DWT-Route / LL Bypass 解决了什么问题？

**问题**：标准 cross-attention 让所有空间位置（包括 LL）都去 query style memory。这迫使 style memory 必须学习"如何维持结构"，从而分散了它表达笔触、色彩等风格属性的能力。

**DWT-Route 解决方案**：

- 对特征图做 Haar DWT；
- **LL bypass**：LL 子带不参与 cross-attention query；
- 仅 LH/HL/HH 的 tokens 作为 query 去查 style memory；
- 之后用 IDWT 重建：LL 保持原值，高频被 cross-attention 输出替换。

**收益**：style memory 100% 的容量用于表达风格。

**代价**：CLIP-S 衡量整体风格（包含低频），LL bypass 阻止了 style memory 影响低频结构，导致架构存在 **1:8 trade-off**（clip 每提升 1 单位，lpips 损失约 8 单位）。为缓解训练-推理分布不匹配，采用 **Stochastic DWT Route（p=0.8）**：训练时 80% 步用 DWT route，20% 步用全空间 query。

---

## 七、Spectral FM Loss 为什么只保留 LL/LH/HL？

训练目标为 per-subband FM loss：

$$\mathcal{L}_{\text{spectral}} = w_{\text{ll}} \text{MSE}(v_{\text{LL}}, \Delta_{\text{LL}}) + w_{\text{lh}} \text{MSE}(v_{\text{LH}}, \Delta_{\text{LH}}) + w_{\text{hl}} \text{MSE}(v_{\text{HL}}, \Delta_{\text{HL}})$$

**HH 被移除的原因**：

1. 628 L8 消融实验显示 HH velocity head 对 CLIP-S 影响 **DEAD**（Δclip = ±0.0001）；
2. HH 主要承载对角纹理、噪点和细粒度细节，对可感知风格的贡献有限，反而容易引入不稳定性；
3. 去掉 HH head 与 loss 后，推理时 HH 子带也不再积分，简化了 ODE 轨迹并降低了噪声。

保留 LL loss（尽管 $w_{\text{ll}}$ 较小）的原因是它能让梯度回流到主干，改善 backbone 对内容结构的理解（4G.1：可降 lpips）。

---

## 八、7 个核心洞察

1. **频域解耦是内容-风格分离的正确表示**：Haar DWT 的正交性让不同频段可被独立控制，避免了欧氏空间中"保内容"与"换风格"的冲突。
2. **LL 不是纯内容锚**：LL 同时承载全局色调、光照等风格信息，对 CLIP-S 有显著贡献。
3. **独立子带速度场避免频率耦合**：shared backbone 提取共享表征，独立 heads 让各频段按自身速度移动。
4. **风格是学出来的，不是抽出来的**：learnable style memory 比外部 DINOv2 特征更有效，避免了 content bias 污染风格表征。
5. **Softmax attention 会稀释风格信号**：ReLU²/gated 注意力通过稀疏激活与无归一化，让 style 条件更有效地注入网络。
6. **DWT Route 释放 style memory 的容量**：LL bypass 让 style memory 不必学习"维持结构"，专注于表达笔触与色彩。
7. **HH velocity head 是冗余的**：实验表明 HH 对风格指标无贡献，移除后模型更简洁、更稳定。

---

## 九、30 秒故事线

> 传统 Flow Matching 把 latent 当成平直向量，但风格迁移里低频是结构、高频是笔触，必须分开处理。Spectral ODE Bridge 先用 Haar DWT 把 latent 拆成 LL、LH、HL、HH 四个正交子带：LL 保内容，LH/HL 传笔触，HH 太噪被去掉。一个共享 backbone 看全局，再为每个子带配独立 velocity head，让网络知道哪里该快、哪里该慢。cross-attention 走 DWT Route——LL  bypass，只让高频 tokens 查 style memory——这样 style memory 不用浪费容量去学"维持结构"。最后在端点用 AdaIN/WCT 做统计匹配，把风格当作 terminal condition 注入。结果：稳定的 ODE 轨迹，内容保住，风格到位。

---

*整理时间：2026-07-03*
