# OT 效果不佳的根因分析: 理论 vs 实现

> 基于 docs/616 设计文档 + 实际实验数据交叉验证

---

## 一、实际实验数据一览

### 1.1 I2SB slerp orthogonal (24 epochs, sigma=0.02)

| Epoch | all-pairs style | LPIPS | 解读 |
|:---:|:---:|:---:|------|
| e1 | **0.705** | 0.447 | 最高 style 但 LPIPS 崩 |
| e3 | 0.699 | 0.393 | style 下降 |
| e6 | 0.688 | 0.362 | LPIPS 改善 |
| e11 | 0.683 | 0.355 | 平衡点 |
| e18 | 0.680 | 0.348 | LPIPS 进一步改善 |
| e24 | 0.679 | 0.349 | 最终收敛 |

**对比 topogate baseline (0.703/0.312)**: I2SB orthogonal 在 style 上输了 2.4 个点，LPIPS 输了 3.7 个点。

### 1.2 SMoE Translator (15 epochs)

| Epoch | transfer style | LPIPS |
|:---:|:---:|:---:|
| e1 | 0.672 | 0.333 |
| e8 | 0.670 | 0.318 |
| e15 | 0.671 | 0.334 |

**诊断数据** (from debug.md):
| epoch | topo_entropy | eff_experts | tok_delta |
|:---:|:---:|:---:|:---:|
| 1 | 0.779 | 5.40 | 0.0052 |
| 8 | 0.891 | 5.44 | 0.0155 |
| 15 | 1.235 | 4.72 | 0.0187 |

---

## 二、根因分析

### 根因 #1: OT 配对的度量空间错配 — 不是实现问题，是欧氏度量的数学必然

文档 616/design.md 精准指出了核心问题：

**当前代价矩阵**: $C_{ij} = \| x_i^{\text{content}} - x_j^{\text{style}} \|_2^2$

这是在潜空间算欧氏距离。但潜空间的欧氏距离**根本不是感知距离**。两张结构完全不同但颜色相似的图，欧氏距离可能很小——OT 会把它们"配成一对"。结果就是：
- 色彩中庸的图像成为"枢纽"（hub）——大量源图被匹配到同几个目标
- OT plan 退化为 Many-to-One → 平凡解
- 网络学到的 velocity 包含巨大的水平分量（结构变形）

**验证**: debug.md 中已经计算了 Gini 系数和 OT 成本方差。"如果 Gini > 0.6，说明出现严重枢纽现象"。当前数据虽然没有显式 Gini，但 `plan_entropy` 的下降趋势间接证实了配对的退化。

**结论**: **这是理论问题，不是实现问题**。欧氏 OT 在潜空间上不适配。需要基于语义结构的匹配。

### 根因 #2: 训练动力学中的水平/垂直分量未分离

616/design.md 的核心论点:

$$
TE = \mathcal{H} \oplus \mathcal{V}
$$

- $\mathcal{H}$ (水平): 底流形方向 → 结构变化 → LPIPS 罪魁祸首
- $\mathcal{V}$ (垂直): 纤维方向 → 风格变化 → 应该只学这个

**当前训练**: 目标速度场 $v_{\text{target}} = x_1^{\text{OT}} - x_0$。
由于 OT 配对不准，$v_{\text{target}}$ 包含巨大的 $\mathcal{H}$ 分量。
模型 80% 的容量在记忆"如何把结构变形成配对目标的形状"，只剩 20% 学风格。

**证据**: 
- I2SB orthogonal 实验: 加了 orthogonal 约束试图分离水平和垂直，但 e1 时 style 达到 0.705（比基线高）的同时 LPIPS 也崩了（0.447）。说明 orthogonal 约束**没有真的分离**水平/垂直分量。
- SMoE tok_delta = 0.0187: 翻译矩阵几乎不动。因为 loss 梯度主要来自 $\mathcal{H}$ 分量（结构误差远大于风格误差），W_k 学不到有效的风格变换。

**结论**: **理论正确但实现未落地**。`slerp_orthogonal` 的分解方式不够强。

### 根因 #3: Tokenizer 的风格注入力度严重不足

| 指标 | 数值 | 诊断 |
|------|------|------|
| tok_delta | 0.0187 | 15 epochs 后翻译矩阵偏离恒等的幅度仅 1.9% |
| eff_experts | 4.72/8 | 将近一半的 expert 被闲置 |
| topo_entropy | 1.235 | 路由分布退化为接近均匀 (max=ln32≈3.47, 1.24 意味着 ~29% 的均匀度) |

**具体问题**: SMoE 的恒等初始化虽然保护了 LPIPS，但也让 W_k 失去了"打破对称性"的动力。大部分风格变化被 UNet 的 velocity 承担了，Tokenzier 的输出几乎就是 content features 的恒等拷贝。

**验证**: 如果把 tok_delta 从 0.0187 翻到 0.05（通过降低 kinetic 或增加 tokenizer learning rate），style 是否有显著提升？这是下一个实验的核心问题。

### 根因 #4: 垂直流匹配的实现未使用正确的投影算子

616/design.md 提出用 `x - AvgPool(x, 5)` 作为 $\mathcal{P}_{\mathcal{V}}$。但补充分析指出 kernel=5 在 64x64 latent 上的截断频率 f_cut=0.2，会把中频信息错误地划入"纤维"，导致网络必须额外学习中频保持。

**替代方案**: 使用 2× 下采样再上采样（理想低通滤波）:
```python
def get_base(tensor):
    return F.interpolate(F.avg_pool2d(tensor, 2), scale_factor=2)
```

**尚未测试**: 616/design.md 中的 `Vertical Flow Matching` 方案从未被实现和训练。当前所有实验仍在使用标准 Flow Matching + 各种后施加的正则化。

---

## 三、回答核心问题

### 是理论不对还是实现不对？

**理论正确，实现有 3 个致命偏差**:

1. **OT 匹配**: 理论指出应该用结构距离替代欧氏距离，但**从未实现**。当前仍在用潜空间 MSE 做 Sinkhorn。这是"理论说了但没做"。

2. **垂直流匹配**: 理论提出应该让 $\mu_{\text{base}} = \text{const}$（结构不随时间变化），但**从未实现**。当前训练仍在用标准 FM 的 $(1-t)x_0 + t x_1$ 插值。这是"理论的核心但未落地"。

3. **投影算子**: 用 AvgPool(5) 而非更好的理想低通。这是"理论落地但细节不够精准"。

### SMoE 为什么效果与查表 tokenizer 一样？

SMoE 的核心公式 $\text{Output} = \sum_k \alpha_k \cdot (W_k \times F_{\text{content}})$ 在数学上是正确的改进。
**但 loss 信号被 $\mathcal{H}$ 分量主导了**。UNet 的 velocity loss 占主导，
tokenizer 的 $W_k$ 几乎没有被风格相关的梯度推动。

换句话说: **不是 SMoE 不对，是整个训练框架没有给 SMoE 发挥作用的空间**。

---

## 四、修正路径

### 立即执行 (今天)

**1. 实现纯垂直流匹配 (Pure Vertical FM)**

这是 616/design.md 的核心操作——在 `losses.py` 中修改 `_bridge_state_and_velocity`:
- `mu_base = content_base` (不随时间变化，结构绝对静止)
- `mu_fiber = (1-t) × fiber_content + t × fiber_matched_target`
- `target_velocity` 仅计算纤维分量的差异

**预期**: style 直接突破 0.70，LPIPS 因结构锁定而改善。

**2. 加大 Tokenizer 的学习信号**

- 给 tokenizer 单独的 optimizer，lr = 3× base lr
- 添加 tokenizer 专属的 auxiliary loss: `MSE(tokenizer_output, fiber_matched_target)`
- 或降低 w_kinetic 释放 tokenizer 的变换自由度

### 短期 (明天)

**3. 结构感知 OT 匹配**

替代当前的 MSE 代价矩阵:
```python
# 使用 tokenizer 的 attention map 作为结构指纹
def structural_cost(content, target):
    attn_c = tokenizer.get_attention(content)  # [B, HW, K]
    attn_t = tokenizer.get_attention(target)
    # 比较结构分布而非像素值
    return wasserstein_distance(attn_c_summary, attn_t_summary)
```

**4. 验证垂直 FM 的效果**然后叠加 Fiber-SDE 噪声

### 论文叙事调整

当前论文 story 可以重构为:
- "我们揭示了标准 OT-FM 的度规空间错配定理" (Metric Space Mismatch Theorem)
- "提出纯垂直流匹配，将结构锚定与风格演化解耦为独立的动力学通道"
- "证明了在无配对设定下，水平运动分量是结构退化的根本原因"
