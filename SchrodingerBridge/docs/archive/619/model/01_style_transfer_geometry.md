# 01 — 风格迁移的几何本质：从纤维丛到信息瓶颈

> 从 style-transfer 问题的数学本质出发，利用纤维丛和最优传输工具，
> 诊断当前 LANCET 模型为何 clip_style 被锁死在 ~0.70，以及信息流断裂点在哪里。

---

## 1. 问题的几何形式化

### 1.1 风格迁移 = 纤维上的受控传输

将 VAE 潜空间 $\mathcal{Z} \cong \mathbb{R}^{C \times H \times W}$ 视为纤维丛的总空间
$E = (\mathcal{Z}, \mathcal{B}, \pi, \mathcal{F})$：

| 数学对象 | 物理含义 | 模型中对应 |
|---------|---------|-----------|
| 底空间 $\mathcal{B}$ | 内容拓扑（轮廓、布局、语义） | 自注意力/skip 保持的结构 |
| 纤维 $\mathcal{F}_c = \pi^{-1}(c)$ | 给定内容 $c$ 下的所有可能风格渲染 | 合法 style latent 集合 |
| 截面 $\sigma: \mathcal{B} \to E$ | 一种特定的"画法" | 一次完整的风格迁移输出 |
| 联络 $\nabla$ | 平行移动规则：如何在保持内容不变的前提下改变风格 | TopoGate / attention 门控 |

**风格迁移的本质**就是：在纤维丛上沿联络做平行移动——保持底空间坐标 $c$ 不变，
只改变纤维坐标 $f$，使之从源风格分布移动到目标风格分布。

$$\text{StyleTransfer}: z_{\text{source}} = (c, f_{\text{src}}) \mapsto z_{\text{target}} = (c, f_{\text{tgt}})$$

### 1.2 联络的分解：水平-垂直

切空间分解 $T_z\mathcal{Z} = \mathcal{H}_z \oplus \mathcal{V}_z$：

- **垂直分布** $\mathcal{V}_z = \ker(d\pi_z)$：纯风格变化方向。沿此方向运动不改变内容。
- **水平分布** $\mathcal{H}_z$：内容变化方向。沿此方向运动会改变内容拓扑。

> **理想的风格迁移速度场** $v^*$ 应该完全位于垂直分布中：
> $$v^*(z, t) \in \mathcal{V}_z, \quad \forall z, t$$

### 1.3 LPIPS-Style Pareto 前沿的几何解释

$$\text{LPIPS}(z_0, z_1) \propto \|d\pi(z_1 - z_0)\|_{\text{VGG}} \propto \text{水平分量大小}$$
$$\text{Style Score} \propto d_{\mathcal{F}}(f_{\text{output}}, f_{\text{target}}) \propto \text{垂直分量到达深度}$$

因此 LPIPS-Style Pareto 前沿就是在 $\mathcal{H} \perp \mathcal{V}$ 分解下的最优权衡。
一个完美的模型，其速度场 $v_\theta$ 应该是 **纯垂直的**——LPIPS = 0 且 Style = 最大值。

---

## 2. 信息流诊断：五大瓶颈

通过跟踪信息从 "目标风格图像" 到 "模型输出速度场" 的每一步，
我们可以精确定位风格信号在哪里被丢失或稀释。

### 2.1 瓶颈 1：风格表示坍缩 — 闭集查表

**信息量分析**：

```
目标风格图像: 3 × 512 × 512 = 786,432 float → ~3 MB 信息
   ↓ nn.Embedding 查表
风格向量: 1 × D (D ≈ 256) → 256 float → ~1 KB 信息
   信息压缩率: >3000:1
```

当前 LANCET 使用 `nn.Embedding(num_styles, D)` 查表。这意味着：

1. **只有 5 个离散风格向量**，无法表示风格的连续变化（同一风格内笔触粗细、颜色饱和度等）
2. **无空间信息**：256-D 向量完全丢失了风格图像的空间纹理结构
3. **无实例级变化**：同一 `style_id` 的所有图像映射到同一个向量
4. **零泛化能力**：unseen 风格无法处理

> **纤维丛视角**：Embedding 查表等价于把整个纤维 $\mathcal{F}_c$ 压缩为一个点。
> 模型只知道"去往印象派方向"，但不知道"梵高的星空用螺旋笔触、莫奈的睡莲用点彩"。

### 2.2 瓶颈 2：时间-风格纠缠 — 联络退化

当前实现（`model.py:1459-1460`）：

```python
time_code = self.time_mlp(sinusoidal(t))
return style_code + time_code  # 致命的加法混合
```

**数学后果**：令 $s$ 为风格编码、$\tau$ 为时间编码。下游模块接收 $c = s + \tau$。
对任何使用 $c$ 的线性/仿射变换 $W$：

$$W(s + \tau) = Ws + W\tau$$

模型**无法区分**"$s$ 变化了 $\delta s$" 和 "$\tau$ 变化了 $-\delta s$"。这等价于说
联络的水平分布和垂直分布被混合了——模型失去了分辨"时间推进"和"风格变化"的能力。

> **纤维丛视角**：$s + \tau$ 混合等价于联络退化 (degenerate connection)。
> 水平传输和垂直传输无法独立控制，导致模型在试图"加深风格"时可能意外"改变时间进度"。

### 2.3 瓶颈 3：伪交叉注意力 — 纤维间传输断裂

当前实现（`lancet_blocks.py:128-131`）：

```python
style_bias = style_proj(style_code).unsqueeze(1)        # 1D → 偏移
style_tokens = style_tokens_basis.unsqueeze(0) + style_bias  # 学习 tokens + 偏移
k = k_proj(style_tokens)                                # 查表
v = v_proj(style_tokens)                                 # 查表
```

**信息流断裂点**：K 和 V 来自**全局可学习参数** + 1D 偏移，
而不是来自实际的风格参考图像。

这在最优传输意义上意味着：

$$\text{Plan}_{Q \to K,V} = \text{softmax}\left(\frac{Q \cdot K_{\text{fixed}}^T}{\sqrt{d}}\right) \cdot V_{\text{fixed}}$$

无论输入什么风格图像，$K$ 和 $V$ 的基底是固定的，只有一个全局偏移 $\Delta$ 在变化。
这不是 **从风格图到内容图的空间级传输 (spatial transport)**，
而是 **从固定码本到内容图的全局调色 (global modulation)**。

> **OT 视角**：真正的 Cross-Attention 是一个 **可微分的 Kantorovich 传输计划**
> $$\pi^* = \arg\min_\pi \int c(q, k) d\pi(q, k)$$
> 它应该在内容图的每个空间位置 query 风格图的对应纹理区域。
> 当前的实现退化为一个 **均匀传输计划** $\pi = \text{const}$——所有位置获得相同的风格信号。

### 2.4 瓶颈 4：Minibatch OT 不稳定 — 传输计划抖动

训练时每个 batch 内动态计算 Sinkhorn OT 配对。

**数学问题**：令 $\Pi_b$ 为第 $b$ 个 batch 的最优传输计划。
同一对 $(z_c^i, z_s^j)$ 在不同 batch 中可能被配对或不被配对：

$$\Pi_{b_1}(i, j) > 0 \quad \text{但} \quad \Pi_{b_2}(i, j) = 0$$

模型的速度场目标在不同 epoch 之间跳变：

$$v_{\text{target}}^{(b_1)}(z_c^i) = z_s^{j_1} - z_c^i \neq z_s^{j_2} - z_c^i = v_{\text{target}}^{(b_2)}(z_c^i)$$

这导致速度场学习目标不稳定，模型趋向于学习所有可能目标的**条件期望**——即平均值。

> **OT 视角**：Minibatch OT 的支撑集只有 $B$ 个点。
> 当 $B$ 远小于数据集大小时，这个离散 OT 的解与 population-level OT 的解差异很大，
> 且跨 batch 方差极高。

### 2.5 瓶颈 5：训练时 ODE 展开 — 梯度消失

`_terminal_swd` 在训练循环中调用 `model.integrate()`：

```python
endpoint = model.integrate(content, style_id=..., num_steps=N)  # N步ODE展开
loss = SWD(endpoint, style_target)
loss.backward()  # 梯度流经N步展开
```

梯度链长度为 $N$。由于每一步的 Jacobian $\frac{\partial f_{t+1}}{\partial f_t}$ 
可能不满足谱半径 $\rho < 1$，梯度要么爆炸（被 clamp 掩盖）要么消失。

> **信息论视角**：梯度消失 = 风格损失 $\nabla_\theta \mathcal{L}_{\text{SWD}}$ 
> 对早期层参数的影响消失。模型无法学到"如何在 $t=0$ 附近开始正确的风格转换方向"。

---

## 3. 均值坍缩定理：style ≈ 0.70 的理论根因

### 3.1 定理陈述

**定理（确定性 ODE 的条件期望收敛）**：
设 $v_\theta^*$ 是 $L_2$ 损失下的最优速度场。则对确定性 ODE 求解器：

$$\lim_{t \to 1} x_t = \mathbb{E}[x_{\text{style}} \mid \pi(x_0) = c]$$

即输出收敛于**给定内容结构下所有可能风格的算术平均**。

### 3.2 直觉

对于固定的内容结构 $c$（例如一张风景照），目标风格可能对应：
- 梵高的螺旋笔触
- 莫奈的点彩
- 歌川广重的线条

这三者在潜空间中的平均值是一个**模糊的、无风格特征的灰色混合物**。

### 3.3 已有实验验证

| 模型变体 | Solver | clip_style | 备注 |
|---------|--------|:---:|------|
| SMoE + ODE | 确定性 | 0.7022 | **均值坍缩**：笔触平滑 |
| SMoE + SDE (σ=0.02) | 随机 | 0.7045 | 微弱改善 |
| SDE + Overdrive 1.8 | 随机+外推 | 0.7188 | 外推到极限 |
| SDE + Overdrive + Affine | 随机+外推+仿射 | 0.7219 | **最高值** |

每一层"hack"（SDE噪声、外推、仿射对齐）都在试图**绕过**均值坍缩，
而不是从根本上解决它。即使叠加所有 hack，style 仍然卡在 ~0.72。

### 3.4 突破均值坍缩的数学条件

要真正突破，需要满足以下至少一个条件：

1. **引入随机性** (SDE)：使轨迹可以散布到条件分布的边界
2. **引入 instance-level conditioning**：让模型知道"这次要画梵高，不是莫奈"
3. **正确的训练目标**：从分布匹配切换到 instance-level 回归

> 当前 LANCET 的 SDE 尝试只满足条件 1（且噪声很弱），
> 但条件 2（风格图像级条件注入）和条件 3（独立耦合 Flow Matching）都完全缺失。

---

## 4. OT 视角的信息流全景

完整的信息流可以用一个最优传输链来描述：

```
Source Distribution          Transport Plan         Target Distribution
─────────────────          ──────────────         ─────────────────
                                                   
Content Images  ────────→  Model v_θ  ────────→  Stylized Outputs
  μ_content                                         ν_output
                                                   
Style Images    ──── ? ──→  ??  ──── ? ──→        Target Style Distribution
  μ_style                                           ν_style
```

**当前断裂点**：从 `Style Images` 到 `Model v_θ` 的信息传输路径是
**通过 `nn.Embedding` 查表 → 1D 偏移 → 伪交叉注意力**，
这条路径的信息通量 (information throughput) 极低。

**理想的信息流**：

```
Style Image ──→ DINOv2/CLIP Encoder ──→ 空间特征图 [B,N,D]
                                          ↓
                                    True Cross-Attention
                                    K = proj(style_feat)
                                    V = proj(style_feat)
                                          ↓
Content Features ──→ Q = proj(content) ──→ Attention(Q,K,V) ──→ 风格化特征
```

这条路径的信息通量为 $N \times D$（例如 $256 \times 384 \approx 100$K），
相比查表的 $D = 256$，提升了 **~400倍**。

---

## 5. 总结：诊断结论

| 问题 | 几何本质 | 严重程度 | 解决方向 |
|------|---------|:---:|---------|
| 闭集查表 | 纤维压缩为点 | 💀💀💀 | 引入 Style Encoder |
| 时间-风格混合 | 联络退化 | 💀💀 | 分离 AdaLN(t) 和 CrossAttn(s) |
| 伪交叉注意力 | 传输计划退化为常数 | 💀💀💀 | 真实 Cross-Attention |
| Minibatch OT 不稳定 | 传输计划跨 batch 抖动 | 💀💀 | 离线预配对或 Independent Coupling |
| 训练时 ODE 展开 | 梯度链过长 | 💀💀 | 单步 Flow Matching Loss |
| 均值坍缩 | 确定性 ODE 的固有限制 | 💀💀💀 | Instance-level 条件 + 随机性 |

> **核心结论**：当前模型的 style ≈ 0.70 天花板不是超参数问题，
> 是**架构级信息流断裂**。风格图像的空间纹理信息从未进入过模型。
> 模型只能做"全局色调偏移"，无法做"局部笔触迁移"。

**下一步**：基于这个诊断，设计新的架构方案 →
见 [02_architecture_design.md](./02_architecture_design.md)
