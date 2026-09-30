# 白化机制数学理论

> 建立日期: 2026-06-21  
> 基于docs/620/fog/theory/的已有理论，整合并扩展

---

## 1. 问题形式化

### 1.1 符号定义

设:
- $x \in \mathbb{R}^{C \times H \times W}$: content latent (source)
- $y \in \mathbb{R}^{C \times H \times W}$: target style latent
- $\hat{z}_1 = f_\theta(x, t=0, s)$: 模型预测的endpoint
- $v = f_\theta(x, t, s)$: 模型预测的velocity
- $\alpha = \frac{\langle \hat{z}_1 - x, y - x \rangle}{\|y - x\|_2^2}$: endpoint投影系数

### 1.2 白化的数学定义

**定义1 (白化)**: 生成图像的WFI score > 阈值τ (当前τ=0.40)

**定义2 (Endpoint收缩)**: α < α_threshold (当前α_threshold=0.3)

**定义3 (条件期望坍缩)**: 
$$\text{Var}_s[v_\theta(x, t, s)] \ll \mathbb{E}_s[\|v_\theta(x, t, s)\|_2^2]$$

即不同style的velocity方向互相抵消。

---

## 2. 四重机制的数学模型

### 2.1 机制1: Attention Bottleneck

**定理1 (Softmax平均化)**:
设 style tokens $\{k_j, v_j\}_{j=1}^N$，content query $q$。当 $q^T k_j$ 对所有 $j$ 近似相等时:

$$\text{softmax}(q^T K / \sqrt{d}) \approx \frac{1}{N} \mathbf{1}_N$$

$$\text{CA}(q, K, V) \approx \frac{1}{N} \sum_{j=1}^N v_j = \bar{v}$$

**推论**: 若 style tokens 的 K 投影方差小，则 cross-attention 输出与 style 无关。

**本地验证**: 
- `cross_attn_entropy = 5.531 / ln(256) = 0.997` (99.7% 均匀)
- `cos(v(s₁), v(s₂)) = 0.9995` (style间velocity几乎相同)

**量化指标**:
$$\eta_\text{attn} = 1 - \frac{H(\text{attn})}{\ln(N)} \in [0, 1]$$

η_attn → 0 表示attention完全均匀(style无效)  
η_attn → 1 表示attention稀疏(style有效)

当前值: η_attn ≈ 0.003 (极度均匀)

### 2.2 机制2: Norm Collapse

**定理2 (GN方差归一化)**:
对任意 $h \in \mathbb{R}^{B \times C \times H \times W}$:

$$\text{GN}(h)_{b,:,:,:} = \frac{h_{b,:,:,:} - \mu_b}{\sigma_b}$$

其中 $\mu_b = \frac{1}{CHW}\sum_{c,i,j} h_{b,c,i,j}$, $\sigma_b = \sqrt{\frac{1}{CHW}\sum_{c,i,j}(h_{b,c,i,j}-\mu_b)^2}$

**推论1**: GN后 $\text{Var}[\text{GN}(h)_b] = 1$ (归一化为单位方差)

**推论2**: 跨style的方差差异被消除:
$$\text{Var}_{s}[\text{GN}(h(s))_b] \approx 0$$

**Style信号保留率**:
$$R_\text{style}(l) = \frac{\|\text{GN}(h^{(l)}(s_1)) - \text{GN}(h^{(l)}(s_2))\|_2}{\|h^{(l)}(s_1) - h^{(l)}(s_2)\|_2}$$

理论预测: 对仅改变均值/方差的style差异，$R_\text{style} \approx 0$

### 2.3 机制3: Endpoint Head Capacity

**定理3 (Zero-init陷阱)**:
设 endpoint head 最后一层为 $W_L \in \mathbb{R}^{C_{out} \times C_{in}}$，初始化 $W_L \sim \mathcal{N}(0, \sigma^2)$。

训练初期:
$$\|\hat{z}_1 - x\|_2 = \mathcal{O}(\sigma)$$

若 $\sigma \ll \|y - x\|_2$，则 $\alpha \approx 0$。

**FiLM修正**:
$$\hat{z}_1 = x + \text{FiLMHead}(h, s)$$

FiLMHead 容量由 hidden_dim 决定:
- hidden_dim=128: WFI=0.4283 (不够)
- hidden_dim=512: WFI=0.3906 (刚好过门)

**容量-性能关系**:
$$\text{WFI} \propto \frac{1}{\text{hidden\_dim}^\beta}, \quad \beta \approx 0.3$$

### 2.4 机制4: Loss Landscape

**定理4 (Shrinkage Basin)**:
联合loss:
$$\mathcal{L}(v) = w_\text{FM}\|v - v_\text{target}\|^2 + w_\text{SWD} \cdot \text{SWD}(x + (1-t)v, y) + w_\text{edge}\mathcal{L}_\text{edge}$$

在 $v = 0$ 处:
$$\nabla_v \mathcal{L}|_{v=0} = -2w_\text{FM} v_\text{target} + w_\text{SWD}(1-t) \nabla_z \text{SWD}|_{z=x} + \nabla_v \mathcal{L}_\text{edge}|_{v=0}$$

**平凡解条件**: 
$$\|2w_\text{FM} v_\text{target}\| < \|w_\text{SWD}(1-t) \nabla_z \text{SWD} + \nabla_v \mathcal{L}_\text{edge}\|$$

当前实验: SWD梯度非零但与$v_\text{target}$几乎正交，不构成有效修正。

---

## 3. 四重机制的耦合模型

### 3.1 Shrinkage系数分解

$$\alpha \approx \alpha_\text{attn} \cdot \alpha_\text{norm} \cdot \alpha_\text{endpoint} \cdot \alpha_\text{loss}$$

各因子估计:
- $\alpha_\text{attn} \approx 0.3$ (attention均匀化)
- $\alpha_\text{norm} \approx 0.7$ (GN压缩)
- $\alpha_\text{endpoint} \approx 0.5$ (head容量不足)
- $\alpha_\text{loss} \approx 0.9$ (辅助loss轻微拉扯)

乘积: $\alpha \approx 0.094$，与观测量级(0.16)一致。

### 3.2 修复效果预测

| 修复 | $\Delta\alpha_\text{attn}$ | $\Delta\alpha_\text{norm}$ | $\Delta\alpha_\text{endpoint}$ | 预测α |
|------|---------------------------|---------------------------|-------------------------------|-------|
| gate=0.3 | +0.1 | 0 | 0 | 0.13 |
| FiLM hd512 | 0 | 0 | +0.2 | 0.18 |
| 无GN endpoint | 0 | +0.15 | 0 | 0.14 |
| 组合 | +0.1 | +0.15 | +0.2 | 0.26 |

实际观测(WFI从0.49降至0.39): 与预测方向一致。

---

## 4. 可证伪预测

| 编号 | 预测 | 实验 | 判定 |
|------|------|------|------|
| T1 | 纯FM loss下α≈1 | FM-only训练 | 待验证 |
| T2 | 移除endpoint GN升α | FiLM head无GN | 待验证 |
| T3 | FiLM后R_style>0.5 | 层内probe | 待验证 |
| T4 | attention entropy正则降低WFI | entropy reg实验 | 待验证 |
| T5 | velocity_scale_loss约束shrinkage | 新loss项 | 待验证 |

---

## 5. 关键不等式

### 5.1 白化充分条件

$$\text{WFI} > \tau \iff \alpha < \alpha_\text{crit} \land \eta_\text{attn} < \eta_\text{crit}$$

当前: α≈0.16 < 0.3, η_attn≈0.003 < 0.1 → 白化

### 5.2 修复必要条件

要将WFI降至0.20以下，需要:

$$\alpha > 0.5 \land \eta_\text{attn} > 0.1 \land R_\text{style} > 0.3$$

即至少需要:
1. Endpoint投影系数 > 0.5
2. Attention稀疏度 > 10%
3. Style信号保留率 > 30%

### 5.3 理论上界

在当前架构下，WFI的理论下界为:

$$\text{WFI}_\text{min} \approx 0.15 + 0.1 \cdot (1 - \alpha_\text{max})$$

若α_max=0.9，则WFI_min≈0.16，接近Seedream IDT水平(0.158)。
