# Cross-Attention数学分析：是否应该去掉？

## 1. Cross-Attention在Schrödinger Bridge中的角色

在SB框架中，模型需要学习velocity field $v_\theta(x, t, s)$，条件于style $s$。Cross-attention的作用是让content特征 $x$ "查询"style特征 $S = \{s_1, ..., s_N\}$：

$$\text{CA}(x, S) = \text{softmax}\left(\frac{Q(x) \cdot K(S)^T}{\sqrt{d}}\right) \cdot V(S)$$

## 2. 三个数学问题

### 问题1：Softmax归一化导致信息稀释

softmax强制 $\sum_{i=1}^{N} w_i = 1$。当 $N = 256$ 时：

- 均匀分布：每个token权重 $w_i = 1/256 \approx 0.004$
- 即使最重要的token，权重也很难超过0.05（因为 $Q \cdot K^T$ 的值域有限）
- **结果**：$V(S)$ 的加权和几乎是所有token的平均 $\bar{V}(S) = \frac{1}{N}\sum_i V(S)_i$

**实验证据**：`cross_attn_entropy = 5.531 / ln(256) = 5.545` → 99.9%均匀

### 问题2：Query与style无关

$$Q = W_Q \cdot x$$

Q完全由content $x$ 决定，与style $s$ 无关。因此：

$$\text{attn weights} = \text{softmax}(Q(x) \cdot K(S)^T)$$

对不同的style $s_1, s_2$，如果 $K(S_1) \approx K(S_2)$（style encoder产生的K相似），则attention weights几乎相同。

**实验证据**：`cos_sim(v(s_1), v(s_2)) = 0.9995` → 模型完全忽略style

### 问题3：条件期望坍缩

Cross-attention实际学到的是：

$$\text{CA}(x, S) = \sum_i w_i(x) \cdot V(S)_i \approx \sum_i \frac{1}{N} V(S)_i = \mathbb{E}_{i}[V(S)_i]$$

这是对style tokens的**边缘期望**，而非条件于特定style的信息。模型学到的是：

$$v_\theta(x, t, s) \approx \mathbb{E}_s[v_\theta(x, t, s)] = \bar{v}(x, t)$$

## 3. 去掉Cross-Attention的影响分析

### 3.1 可能的坏处

| 坏处 | 严重性 | 缓解方案 |
|------|--------|---------|
| 丢失空间对齐能力 | **低** | style transfer不需要空间对齐，style是全局的 |
| 丢失token-level style信息 | **中** | FiLM多层MLP可以从全局向量恢复丰富调制 |
| 表达能力下降 | **低** | DiT用AdaLN，StyleGAN用AdaIN，都证明全局条件化足够 |

### 3.2 可能的好处

| 好处 | 机制 |
|------|------|
| 消除softmax瓶颈 | style信息通过FiLM直接注入，不经过归一化 |
| 减少计算量 | 去掉O(N²)的attention计算 |
| 避免条件期望坍缩 | FiLM的 $\gamma(s), \beta(s)$ 直接由style决定 |
| 梯度路径更直接 | $\partial\mathcal{L}/\partial s$ 不经过softmax |

### 3.3 数学论证

**用cross-attention**：
$$v_{\text{style}} = \text{softmax}(Q(x) K(S)^T) V(S) \approx \bar{V}(S)$$
→ style信息被平均化

**用FiLM/AdaLN**：
$$v_{\text{style}} = \gamma(s) \cdot x + \beta(s)$$
→ style信息直接传递，不经过归一化

**关键区别**：FiLM的 $\gamma(s)$ 和 $\beta(s)$ 是style的**确定性函数**，不存在"平均化"。每个style $s$ 产生唯一的 $\gamma, \beta$，从而产生唯一的velocity。

## 4. 三种改进方案

### 方案A：Gated Attention（替换softmax）

用sigmoid门控替换softmax：

$$\text{attn}_i = \sigma\left(\frac{Q \cdot K_i^T}{\sqrt{d}} + b(s)_i\right)$$
$$\text{output} = \sum_i \text{attn}_i \cdot V_i$$

**优点**：每个token独立门控，不归一化，style_bias $b(s)$ 直接影响每个token
**缺点**：输出尺度不固定，可能需要额外归一化

### 方案B：Sparsemax（稀疏softmax）

$$\text{sparsemax}(z) = \arg\min_{p \in \Delta^d} \|p - z\|^2$$

**优点**：产生精确的0，强制稀疏化
**缺点**：实现复杂，梯度计算需要排序

### 方案C：完全去掉Cross-Attention，用AdaLN

$$\text{AdaLN}(x; s, t) = (1 + \gamma(s, t)) \cdot \text{Norm}(x) + \beta(s, t)$$

**优点**：数学最干净，DiT标准做法，计算高效
**缺点**：丢失token-level信息（但对style transfer不关键）

## 5. 推荐策略

**渐进式实验**：

1. **先试方案A（gated attention）**：保留cross-attention结构，最小改动
2. **如果A无效，试方案C（AdaLN替换）**：完全去掉cross-attention
3. **方案B（sparsemax）作为备选**：如果A和C都不行

## 6. 结论

**Cross-attention在当前设置下确实有数学问题**：
1. softmax归一化导致信息稀释（entropy = 99.9%均匀）
2. query与style无关导致attention weights不区分style
3. 最终学到边缘期望而非条件期望

**去掉不会造成严重坏影响**，因为：
1. style transfer不需要空间对齐
2. FiLM/AdaLN提供足够的条件化能力
3. DiT和StyleGAN的成功证明了全局条件化的有效性

**推荐**：先尝试gated attention（最小改动），如果无效则完全去掉cross-attention，用AdaLN替换。