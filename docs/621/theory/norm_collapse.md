# 621 归一化统计塌缩理论

> 建立日期: 2026-06-21

---

## 1. GroupNorm(1) 的数学效应

### 1.1 定义

对特征图 $h \in \mathbb{R}^{B \times C \times H \times W}$:

$$\text{GN}(h)_{b,c,i,j} = \frac{h_{b,c,i,j} - \mu_b}{\sqrt{\sigma_b^2 + \epsilon}}$$

其中:
$$\mu_b = \frac{1}{CHW}\sum_{c,i,j} h_{b,c,i,j}, \quad \sigma_b^2 = \frac{1}{CHW}\sum_{c,i,j}(h_{b,c,i,j}-\mu_b)^2$$

### 1.2 效应1: 方差归一化

**定理**: GN(1)后 $\text{Var}[\text{GN}(h)_b] = 1$ (当affine=False)

**证明**:
$$\text{Var}[\text{GN}(h)_b] = \text{Var}\left[\frac{h_b - \mu_b}{\sigma_b}\right] = \frac{\text{Var}[h_b - \mu_b]}{\sigma_b^2} = \frac{\sigma_b^2}{\sigma_b^2} = 1$$

**影响**: 跨style的方差差异被消除，style调制的二阶统计信息丢失。

### 1.3 效应2: 均值归零

**定理**: GN(1)后 $\mathbb{E}[\text{GN}(h)_b] = 0$

**证明**:
$$\mathbb{E}[\text{GN}(h)_b] = \frac{\mathbb{E}[h_b] - \mu_b}{\sigma_b} = \frac{\mu_b - \mu_b}{\sigma_b} = 0$$

**影响**: style调制的一阶统计信息(均值偏移)被完全消除。

### 1.4 效应3: 通道方差拉平

在4通道VAE latent中，每个通道编码不同属性(亮度、色度等)。GN(1)在所有通道上计算一个均值和方差，导致:

$$\text{Var}[h_{b,c}] \approx 1 \quad \forall c$$

不同通道间的相对方差差异被消除。

---

## 2. Style信号保留率

### 2.1 定义

$$R_\text{style}(l) = \frac{\|\text{GN}(h^{(l)}(s_1)) - \text{GN}(h^{(l)}(s_2))\|_2}{\|h^{(l)}(s_1) - h^{(l)}(s_2)\|_2}$$

### 2.2 理论预测

**情况1**: style差异仅改变均值/方差

设 $h(s_1) \sim \mathcal{N}(\mu_1, \sigma_1^2)$, $h(s_2) \sim \mathcal{N}(\mu_2, \sigma_2^2)$

则:
$$\text{GN}(h(s_1)) \sim \mathcal{N}(0, 1), \quad \text{GN}(h(s_2)) \sim \mathcal{N}(0, 1)$$

$$R_\text{style} = \frac{\|\mathcal{N}(0,1) - \mathcal{N}(0,1)\|_2}{\|\mathcal{N}(\mu_1,\sigma_1^2) - \mathcal{N}(\mu_2,\sigma_2^2)\|_2} \approx 0$$

**情况2**: style差异改变分布形状(高阶矩)

此时GN后保留高阶信息，$R_\text{style} > 0$。

### 2.3 实验预测

在以下位置测量$R_\text{style}$:

| 位置 | 预期$R_\text{style}$ | 白化信号 |
|------|---------------------|----------|
| block输入 | 0.8-1.0 | — |
| GN后 | 0.1-0.3 | GN压缩 |
| Cross-attention后 | 0.6-0.8 | style注入 |
| FiLM后 | 0.7-0.9 | style增强 |
| FFN(GN)后 | 0.1-0.3 | GN再压缩 |

**关键**: 每次GN都会大幅降低$R_\text{style}$，style信号在多层传递中逐渐衰减。

---

## 3. AdaLN(time) 的效应

### 3.1 定义

$$\text{scale}, \text{shift}, \text{gate} = \text{time\_adaln}(\text{time\_emb}).\text{chunk}(3)$$

$$h_\text{time} = \text{GN}(h) \odot (1 + \text{scale}) + \text{shift}$$

### 3.2 Gate饱和效应

当gate→0时:
$$\text{sa\_delta} = \sigma(\text{gate}) \cdot \text{SA}(h) \to 0$$

self-attention分支被关闭，style只能通过cross-attention和FiLM传递。

### 3.3 Scale/Shift效应

若scale→-1:
$$h_\text{time} = \text{GN}(h) \odot 0 + \text{shift} = \text{shift}$$

特征被归零，信息完全丢失。

---

## 4. 与白化的联系

### 4.1 Latent统计→Decode图像

VAE decode $D: \mathbb{R}^{4 \times 64 \times 64} \to \mathbb{R}^{3 \times 512 \times 512}$

若latent的:
- 均值被拉高 → decode后图像偏亮
- 方差被压缩 → decode后对比度降低
- 通道间统计被拉平 → decode后饱和度降低

### 4.2 定量预测

设latent方差压缩因子为$\kappa$ (GN前$\sigma^2$, GN后$\sigma^2/\kappa$):

$$\text{contrast\_ratio}_\text{gen} \approx \kappa \cdot \text{contrast\_ratio}_\text{target}$$

若$\kappa = 0.3$ (GN压缩70%)，则:
$$\text{contrast\_ratio}_\text{gen} \approx 0.3 \times 0.42 = 0.126$$

这与观测值(contrast_ratio≈0.15)一致。

---

## 5. 修复方向

### 5.1 方案1: 移除endpoint head的GN

FiLMEndpointHead已移除GN，但block内仍有GN。

**预测**: 移除block内GN会显著增加$R_\text{style}$，但可能导致训练不稳定。

### 5.2 方案2: 使用Adaptive GroupNorm

用style调制GN的$\gamma, \beta$:
$$\text{AdaGN}(h, s) = \gamma(s) \cdot \frac{h - \mu}{\sigma} + \beta(s)$$

这样即使GN归一化了方差，style仍能通过$\gamma, \beta$调制。

### 5.3 方案3: 减少GN使用

在关键路径(如FiLM后)不使用GN，只在FFN等非关键位置使用。

### 5.4 方案4: 使用RMSNorm

RMSNorm只归一化方差，不减去均值:
$$\text{RMSNorm}(h)_{b,c,i,j} = \frac{h_{b,c,i,j}}{\sqrt{\frac{1}{CHW}\sum_{c,i,j}h_{b,c,i,j}^2 + \epsilon}}$$

保留了均值信息(一阶style信号)。

---

## 6. 层内Probe设计

```python
class NormCollapseProbe:
    def __init__(self, model):
        self.hooks = []
        self.stats = {}
        # 注册到每个GN
        for i, block in enumerate(model.blocks):
            self.hooks.append(block.norm1.register_forward_hook(self._make_hook(f'block{i}_norm1')))
            self.hooks.append(block.norm2.register_forward_hook(self._make_hook(f'block{i}_norm2')))
    
    def _make_hook(self, name):
        def hook(module, input, output):
            self.stats[name] = {
                'input_var': input[0].detach().float().var().item(),
                'output_var': output.detach().float().var().item(),
                'compression_ratio': input[0].detach().float().var() / (output.detach().float().var() + 1e-8),
            }
        return hook
    
    def compute_R_style(self, outputs_s1, outputs_s2):
        """计算每层的style信号保留率"""
        R_style = []
        for out1, out2 in zip(outputs_s1, outputs_s2):
            diff_in = (out1 - out2).flatten(1).norm(dim=1).mean()
            gn_out1 = F.group_norm(out1, 1)
            gn_out2 = F.group_norm(out2, 1)
            diff_out = (gn_out1 - gn_out2).flatten(1).norm(dim=1).mean()
            R = diff_out / (diff_in + 1e-8)
            R_style.append(R.item())
        return R_style
```
