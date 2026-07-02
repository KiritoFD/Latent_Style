# StyleFiLM：绕过Cross-Attention平均化瓶颈

## 1. 问题诊断

### 1.1 探针发现
- cos_sim(v(style_1), v(style_2)) = **0.9995** — 模型输出的velocity方向几乎与style无关
- 无论输入什么style，模型都输出相同的velocity方向

### 1.2 根因分析：Cross-Attention的平均化效应

Cross-attention的数学形式：

$$\text{CA}(x, S) = \text{softmax}\left(\frac{Q(x) \cdot K(S)^T}{\sqrt{d}}\right) \cdot V(S)$$

其中 $x$ 是content特征，$S = \{s_1, ..., s_{256}\}$ 是256个style tokens。

**关键问题**：attention weights由 $Q(x)$（content-dependent）决定，而style tokens $K(S)$ 对不同的style图像都会产生相似的attention分布。softmax将所有token平均化后，$V(S)$ 的加权和在不同style之间几乎不变。

数学上，这等价于模型学到的不是 $v_\theta(x, t, \text{style})$，而是：

$$v_\theta(x, t, \text{style}) \approx \mathbb{E}_{\text{style}}[v_\theta(x, t, \text{style})] = \bar{v}(x, t)$$

这就是**条件期望坍缩**（Conditional Expectation Collapse）——模型输出的是对style的边缘期望，而非条件于特定style的velocity。

### 1.3 梯度路径分析

没有FiLM时，style信号到loss的梯度路径：

$$\frac{\partial \mathcal{L}}{\partial \theta_{\text{style}}} \propto \frac{\partial \mathcal{L}}{\partial v} \cdot \frac{\partial v}{\partial \text{CA}} \cdot \frac{\partial \text{CA}}{\partial \text{softmax}} \cdot \frac{\partial \text{softmax}}{\partial QK^T} \cdot \frac{\partial K}{\partial \theta_{\text{style}}}$$

softmax的梯度在多token情况下被稀释，导致style encoder收到的梯度信号极弱。

## 2. StyleFiLM：直接Style→Feature调制

### 2.1 数学定义

FiLM (Feature-wise Linear Modulation) 提供了一条绕过cross-attention的style注入路径：

$$\text{FiLM}(x; s) = (1 + \gamma(s)) \odot x + \beta(s)$$

其中：
- $x \in \mathbb{R}^{B \times C \times H \times W}$：特征图
- $s \in \mathbb{R}^{B \times C}$：style_global向量（来自style_conditioner的全局池化）
- $\gamma(s), \beta(s) \in \mathbb{R}^{B \times C}$：由MLP从style_global预测的通道级调制参数

### 2.2 在Block中的位置

```
Self-Attention + AdaLN(time)
    ↓
Cross-Attention(content × style) → style_delta
    ↓
Shortcut: x = α · x + style_delta
    ↓
StyleFiLM: x = (1 + γ(s)) · x + β(s)    ← 新增
    ↓
FFN
```

### 2.3 关键设计选择

1. **Zero-init**：$\gamma$ 和 $\beta$ 的投影层权重初始化为0，确保训练开始时FiLM = identity，不破坏已有训练状态。

2. **LayerNorm on style_global**：投影前对style_global做LayerNorm，稳定训练。

3. **Per-block FiLM**：每个block有独立的 $\gamma/\beta$ 预测器，不同深度的block可以学习不同层次的style调制。

## 3. 为什么StyleFiLM能解决条件期望坍缩

### 3.1 梯度路径对比

有FiLM时，style信号到loss的梯度路径：

$$\frac{\partial \mathcal{L}}{\partial \theta_{\text{style}}} \propto \frac{\partial \mathcal{L}}{\partial v} \cdot \frac{\partial \gamma}{\partial \theta_{\text{style}}}$$

这条路径是**直接的**——不需要经过softmax、不需要经过cross-attention。梯度信号强度大幅提升。

### 3.2 理论保证

即使cross-attention输出完全相同的style_delta（$\Delta_{\text{CA}}(s_1) = \Delta_{\text{CA}}(s_2)$），FiLM层仍然可以产生不同的输出：

$$\text{FiLM}(x + \Delta_{\text{CA}}; s_1) \neq \text{FiLM}(x + \Delta_{\text{CA}}; s_2)$$

因为 $\gamma(s_1) \neq \gamma(s_2)$ 且 $\beta(s_1) \neq \beta(s_2)$。

### 3.3 与AdaIN/StyleGAN的关系

StyleFiLM是AdaIN的泛化形式：
- AdaIN: $x' = \gamma(s) \cdot \frac{x - \mu(x)}{\sigma(x)} + \beta(s)$ — 需要先归一化
- StyleFiLM: $x' = (1 + \gamma(s)) \cdot x + \beta(s)$ — 直接在残差流中调制，更灵活

## 4. 预期效果

1. cos_sim(v(style_1), v(style_2)) 从 0.9995 显著下降
2. 模型学会输出style-specific的velocity方向
3. style transfer质量提升
4. film_gamma_abs和film_beta_abs从0开始增长，表明模型在学习使用FiLM

## 5. 潜在风险

1. **FiLM过强**：如果gamma/beta增长过快，可能压制cross-attention的贡献。缓解：zero-init + 监控film_gamma_abs/beta_abs。
2. **style_global信息不足**：全局池化可能丢失细节。缓解：后续可升级为per-token FiLM。
3. **训练不稳定**：新增的调制可能引起分布偏移。缓解：LayerNorm + zero-init。

## 6. 验证指标

训练过程中监控：
- `film_gamma_abs`：gamma的平均绝对值，应从0开始增长
- `film_beta_abs`：beta的平均绝对值
- `cos_sim(v1, v2)`：shuffled style的velocity余弦相似度，目标从0.9995降到<0.95
- WFI指标：确保白化不恶化

## 7. StyleFiLM v2：Pre-Cross-Attention FiLM

### 7.1 v1的局限性

v1实验（3 epochs）发现：
- film_gamma_abs 从 0.01275 → 0.01966（+54%，增长缓慢）
- cross_attn_entropy 始终卡在 5.531/5.545（接近均匀分布）
- cos_sim 从 0.9995 降到 0.9882（有改善但不够）

**根因**：FiLM在cross-attention之后才调制特征。此时attention已经产生了均匀分布，调制"已经被平均化"的特征效果有限。

### 7.2 v2核心改进：Query-Level FiLM

在cross-attention之前对query特征进行FiLM调制：

$$\text{FiLM}_Q(x; s) = (1 + \gamma_q(s)) \cdot x + \beta_q(s)$$

然后使用调制后的特征作为cross-attention的query：

$$Q = W_Q \cdot \text{FiLM}_Q(x; s)$$

**关键效果**：由于Q变成了style-dependent，attention weights也随之变化：

$$\text{attn}(s_1) = \text{softmax}(Q(s_1) \cdot K^T / \sqrt{d}) \neq \text{attn}(s_2) = \text{softmax}(Q(s_2) \cdot K^T / \sqrt{d})$$

这从源头打破了cross-attention的均匀平均化。

### 7.3 v2架构

每个SpatialBridgeBlock620中有两个FiLM层：

1. **Pre-Cross-Attention FiLM** (`film_q_proj`)：调制query特征
   - 位置：cross-attention之前
   - 作用：使Q style-dependent → attention weights style-specific
   
2. **Post-Cross-Attention FiLM** (`film_proj`)：调制融合特征
   - 位置：cross-attention shortcut之后
   - 作用：直接style→feature调制

### 7.4 预期监控指标

新增debug字段：
- `pre_film_gamma_abs`：query FiLM的gamma强度
- `pre_film_beta_abs`：query FiLM的beta强度
- `cross_attn_entropy`：目标从5.531降到<5.0（注意不再均匀）
- `film_gamma_abs`：post FiLM的gamma强度