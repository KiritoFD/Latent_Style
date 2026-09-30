# 621 Cross-Attention信息瓶颈理论

> 建立日期: 2026-06-21

---

## 1. 问题定义

Cross-attention: content Q × style K/V

$$\text{CA}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

当softmax输出接近均匀分布时，style信号被平均化，模型学到的是条件期望:
$$v_\theta(x, t, s) \approx \mathbb{E}_s[v_\theta(x, t, s)] = \bar{v}(x, t)$$

---

## 2. Softmax平均化的数学分析

### 2.1 均匀化条件

设style tokens的K投影为$\{k_j\}_{j=1}^N$，content query为$q$。

当:
$$q^T k_j \approx c \quad \forall j$$

即所有style token对query的亲和度近似相等时:

$$\text{softmax}(q^T K / \sqrt{d}) \approx \frac{1}{N} \mathbf{1}_N$$

### 2.2 为什么会出现这种情况

1. **Style tokens的K投影方差小**: style tokens经过projector后，K空间的方差被压缩
2. **Content query与style K的内积饱和**: $q^T k_j$ 都很大或都很小
3. **Temperature效应**: $\sqrt{d}$ 缩放使logits差异变小

### 2.3 量化指标

**Attention entropy**:
$$H(\text{attn}) = -\sum_{j=1}^N \text{attn}_j \log(\text{attn}_j)$$

**归一化entropy**:
$$\eta = \frac{H(\text{attn})}{\ln(N)} \in [0, 1]$$

- η → 0: attention稀疏(one-hot)，style有效
- η → 1: attention均匀，style无效

**当前值**: η ≈ 5.531 / ln(256) = 0.997 (99.7% 均匀)

---

## 3. 条件期望坍缩

### 3.1 数学表述

设$v_\theta(x, t, s)$是模型对content $x$、time $t$、style $s$的输出。

当attention均匀时:
$$v_\theta(x, t, s) = \text{MLP}(\text{SelfAttn}(x) + \text{CA}(x, S) + \text{FiLM}(x, s))$$

其中$\text{CA}(x, S) \approx \bar{v}_S$ (与$s$无关)。

因此:
$$v_\theta(x, t, s) \approx \text{MLP}(\text{SelfAttn}(x) + \bar{v}_S + \text{FiLM}(x, s))$$

### 3.2 Style sensitivity

定义:
$$\text{style\_sensitivity} = \text{std}_s[v_\theta(x, t, s)]$$

当FiLM弱时:
$$\text{style\_sensitivity} \approx \text{std}_s[\text{FiLM}(x, s)] \approx 0$$

### 3.3 实验验证

固定$(x, t)$，对5个不同style $s_1, ..., s_5$:

$$\cos(v_\theta(x, t, s_i), v_\theta(x, t, s_j)) \approx 0.9995 \quad \forall i, j$$

**结论**: 条件期望坍缩已发生。

---

## 4. 修复方案分析

### 4.1 方案1: 增大Gate (0.05→0.3)

**原理**: 增大style_delta的幅度

$$\text{style\_delta} = \tanh(\text{gate}) \cdot \text{CA}(Q, K, V)$$

gate=0.05 → tanh(gate)≈0.05 → style_delta被压缩95%
gate=0.3 → tanh(gate)≈0.29 → style_delta被压缩71%

**效果**: style sensitivity从~0.05提升到~0.2 (4倍)

**局限**: 不能解决softmax均匀化问题，只是放大了均匀化的结果

### 4.2 方案2: StyleFiLM (绕过attention)

**原理**: 直接用style_global调制feature map

$$x'' = (1 + \gamma(s)) \cdot x + \beta(s)$$

完全不经过cross-attention，直接注入style信号。

**效果**: 
- style sensitivity恢复到~10
- 但FiLM容量不足时(hd128) WFI=0.4283
- 增大容量(hd512)后 WFI=0.3906

**优势**: 绕过了attention bottleneck

### 4.3 方案3: Gated Attention

**原理**: 用sigmoid替代softmax，不强制归一化

$$\text{attn}_i = \sigma(q^T k_i / \sqrt{d})$$

每个token独立gate，style_bias直接控制。

**效果**: 
- WFI从0.49降至0.49 (无显著改善)
- 但content LPIPS保持良好

**局限**: 仍受style tokens的K投影方差限制

### 4.4 方案4: Pre-FiLM (让Q style-dependent)

**原理**: 在计算Q之前用FiLM调制content features

$$ca\_in = (1 + \gamma_q(s)) \cdot ca\_in + \beta_q(s)$$

这样$Q = W_Q \cdot ca\_in$也依赖于style，attention weights变为style-specific。

**效果**: 
- attention entropy从5.531降至~4.5 (但仍高)
- style sensitivity提升

**优势**: 在attention内部解决均匀化问题

### 4.5 方案5: Style Bias (直接加偏置)

**原理**: 在attention logits上加per-token bias

$$\text{logits} = QK^T / \sqrt{d} + \text{bias}(s)$$

bias直接由style_global生成，不受QK^T限制。

**效果**: 
- softmax后的attention分布被bias调制
- 即使QK^T均匀，bias也能产生非均匀分布

**优势**: 最直接的style注入方式

---

## 5. 信息论分析

### 5.1 Mutual Information

Style→Output的互信息:
$$I(S; V) = H(V) - H(V|S)$$

当attention均匀时:
$$H(V|S) \approx H(V)$$

因此$I(S; V) \approx 0$，style几乎不提供信息。

### 5.2 Information Bottleneck

Cross-attention可以看作信息瓶颈:
$$S \to \underbrace{\text{softmax}(QK^T/\sqrt{d})}_{\text{bottleneck}} \to V$$

瓶颈的容量由attention分布的entropy决定:
- 低entropy → 高容量 → style信息保留
- 高entropy → 低容量 → style信息丢失

### 5.3 修复的信息论预测

| 修复 | 预期$I(S; V)$ | 预期WFI变化 |
|------|--------------|------------|
| gate=0.3 | +0.1 bits | -0.05 |
| FiLM hd512 | +0.3 bits | -0.10 |
| Pre-FiLM | +0.2 bits | -0.08 |
| Style bias | +0.2 bits | -0.08 |
| 组合 | +0.5 bits | -0.15 |

---

## 6. 实验验证方案

### 6.1 Attention Pattern可视化

```python
def visualize_attention(model, content, style):
    # 获取attention weights
    attn_weights = []
    def hook(module, input, output):
        # output是attended values，需要重新计算attention
        q = module.q_proj(input[0])
        k = module.k_proj(style)
        attn = F.softmax(q @ k.T / sqrt(d), dim=-1)
        attn_weights.append(attn.detach())
    
    # 注册hook
    for block in model.blocks:
        block.register_forward_hook(hook)
    
    # Forward
    model(content, style_dino_patches=style)
    
    # 可视化
    for i, attn in enumerate(attn_weights):
        plt.imshow(attn[0].cpu().numpy())
        plt.title(f'Block {i} Attention')
        plt.savefig(f'attention_block{i}.png')
```

### 6.2 Style Sensitivity Probe

```python
def probe_style_sensitivity(model, content, styles, t=0.0):
    velocities = []
    for style in styles:
        v = model(content, t=t, style_dino_patches=style['patches'], style_dino_cls=style['cls'])
        velocities.append(v.flatten(1))
    
    velocities = torch.stack(velocities)
    # Pairwise cosine similarity
    cos_sim = F.cosine_similarity(velocities.unsqueeze(0), velocities.unsqueeze(1), dim=-1)
    # Std across styles
    style_std = velocities.std(dim=0).mean()
    
    return {
        'mean_cos_sim': cos_sim.mean().item(),
        'style_std': style_std.item(),
    }
```

### 6.3 Information-theoretic Probe

```python
def probe_mutual_information(model, content, styles, t=0.0):
    """Estimate I(S; V) via variational bound"""
    # 实现略 - 需要VAE或其他密度估计方法
    pass
```
