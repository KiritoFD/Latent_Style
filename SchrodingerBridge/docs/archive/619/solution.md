# 619: 重构方案 — 理论·数学·实现·实验

> 基于代码审计, 外部审查, 以及我们对风格迁移问题的理解.

---

## 一、理论基础

### 1.1 Flow Matching 标准范式

给定源分布 $p_0$ 和目标分布 $p_1$, 定义条件概率路径:
$$p_t(x \mid x_0, x_1) = \mathcal{N}(x \mid (1-t)x_0 + t x_1, \sigma^2 I)$$

对应速度场:
$$u_t(x \mid x_0, x_1) = \frac{x_1 - x_0}{1}$$

网络学习 $v_\theta(x, t)$ 拟合 $u_t$. 当 $\sigma \to 0$, 路径退化为直线:
$$x_t = (1-t)x_0 + t x_1, \quad v = x_1 - x_0$$

### 1.2 Independent Coupling

在无配对风格迁移中, 我们不知道哪张内容图 $x_c$ 应该对应哪张风格图 $x_s$.

**Independent Coupling** 的做法: 随机配对. $x_0 = x_c$, $x_1 = x_s$, 其中 $x_s$ 是与 $x_c$ 同 batch 的任意风格图 (或随机选取).

**为什么可行**: Flow Matching 允许轨迹交叉. 即使 $x_c$ 和 $x_s$ 结构完全不同, 模型在大量训练样本上通过条件信息学会: "保留 $x_0$ 的结构, 改变 $x_0$ 的纹理到 $x_1$ 的目标". 这需要强大的条件注入 (见 3.2 节).

### 1.3 为什么不做 latent 像素重排

**数学原因**: VAE decoder $D_\phi$ 是一个卷积神经网络. 卷积算子 $*$ 假设输入 $z$ 在空间域上光滑:
$$D_\phi(z) = \text{Conv}(\text{Conv}(z, w_1), w_2, ...)$$

如果用 Sinkhorn plan $\Pi$ 重排 latent: $\hat{z} = \Pi \times z$, 则相邻像素可能来自原 latent 的任意位置 → 破坏了 $z$ 的空间连续性 → 卷积输出产生 checkerboard artifacts.

**正确做法**: 不需要重排. 让 Cross-Attention 在特征空间做软对齐:
$$A = \text{softmax}(Q_c K_s^T / \sqrt{d}), \quad \text{Output} = A \times V_s$$

这是特征空间的"软 Transport", 保持 latent 空间的光滑性.

---

## 二、数学模型

### 2.1 目标函数

$$\mathcal{L}(\theta) = \mathbb{E}_{t \sim U(0,1), (x_c, x_s) \sim \text{batch}} \left[ \|v_\theta(x_t, t, s) - (x_s - x_c)\|^2 \right]$$

其中 $x_t = (1-t)x_c + t x_s$. $s$ 是风格条件 (见 3.2).

### 2.2 风格条件注入的数学形式

**Cross-Attention (推荐)**:
$$h' = \text{softmax}\left(\frac{Q(h) \cdot K(f_s)^T}{\sqrt{d}}\right) \cdot V(f_s)$$

其中 $h$ 是 UNet 当前层的特征, $f_s$ 是 StyleEncoder 从风格图提取的空间特征.

**AdaLN (备选, 轻量)**:
$$h' = h \cdot (1 + \gamma(s)) + \beta(s)$$

其中 $\gamma(s), \beta(s) = \text{MLP}(s)$ 只做全局调制. 学到的是色调/对比度, 不是局部笔触.

### 2.3 时间条件的数学形式

**AdaLN-Zero (DiT/SD3 范式, 推荐)**:
$$h' = h \cdot (1 + \alpha(t) \cdot \gamma) + \alpha(t) \cdot \beta$$

其中 $\alpha(t) = \text{MLP}(\text{sinusoidal}(t))$. $\gamma, \beta$ 是每层的可学习参数.
**Zero-init**: $\alpha(0) = 0$ → 初始时刻 $h' = h$, 训练从恒等开始.

**为什么不能用 time+style 加法**: $f(t) + g(s)$ 使模型无法区分 "t 改变了" 和 "s 改变了". 解耦要求 $t$ 和 $s$ 注入到不同的调制路径.

---

## 三、架构

```
Style Image ──→ StyleEncoder ──→ f_s [B, N, D] ──┐
                                                   ├→ CrossAttn(K,V) ──┐
Content z_c ──→ UNet Encoder ──→ h ──→ [AdaLN] ──→ UNet Decoder ──→ Δz
                                  ↑                                    ↑
                              time_mlp(t)                          AdaLN(t)
```

### 3.1 Time 注入: AdaLN (每一层)

```python
t_emb = sinusoidal(t, dim=256)          # [B, 256]
t_mod = time_mlp(t_emb)                 # [B, 6*C]
scale_1, shift_1, gate_1, scale_2, shift_2, gate_2 = t_mod.chunk(6, dim=1)

# 每个 ResBlock:
h = h * (1 + scale_1) + shift_1         # AdaLN 调制
h = h + gate_1 * conv_block(h)          # ResBlock 主体
h = h * (1 + scale_2) + shift_2
h = h + gate_2 * attn_block(h)
```

### 3.2 Style 注入: Cross-Attention (每个 Attention Block)

```python
f_s = StyleEncoder(style_image)         # [B, N, D], N = 16×16 = 256
f_s = style_proj(f_s)                   # 投影到 UNet 维度

# 在 attention block 中:
q = q_proj(h)                           # Query 来自当前特征
k = k_proj(f_s)                         # Key 来自风格特征
v = v_proj(f_s)                         # Value 来自风格特征
h_style = softmax(q @ k.T / sqrt(d)) @ v
h = h + h_style                         # 残差
```

### 3.3 StyleEncoder 的实现选择 → **需要实验**

**选项 A: Frozen DINOv2 backbone (推荐初始方案)**

```python
class FrozenStyleEncoder(nn.Module):
    def __init__(self):
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        for p in self.dino.parameters():
            p.requires_grad = False
    
    def forward(self, img):
        return self.dino.get_intermediate_layer(img, n=1)  # [B, 256, 384]
```

**选项 B: 可训练的轻量 StyleEncoder**

```python
class TrainableStyleEncoder(nn.Module):
    def __init__(self):
        self.encoder = ResNet18(first_conv=...)  # 从 scratch 训练
        self.proj = nn.Linear(512, style_dim)
    
    def forward(self, img):
        return self.proj(self.encoder(img))
```

**选项 C: CLIP image encoder (语义强但风格弱)**

StyleShot 论文证明 CLIP encoder 的风格表征不够好 → 不推荐.

**需要实验**: A vs B — DINOv2 的泛化能力 vs 从 scratch 训练的特异性.

---

## 四、需要实验确定的实现选择

### 4.1 Coupling Strategy → **需要实验**

| 选项 | 训练公式 | 优势 | 劣势 |
|------|---------|------|------|
| **A: Independent** | $v = x_s - x_c$, 随机配对 | 简单, FM 理论保证 | 可能结构漂移 |
| **B: Weak Semantic** | DINO CLS cos top-50 → 随机选 | 配对更有意义 | 预处理成本 |
| **C: Pure Score** | $v$ 不显式定义, 用 L1+SWD | 无需配对 | 训练慢, 可能不稳定 |

**推荐先跑 A**: 最小改动, 看模型是否能自发学结构保持. 如果 LPIPS > 0.45, 切换到 B 或加 content loss.

### 4.2 StyleEncoder → **需要实验**

| 选项 | 特征 | 参数量 | 训练速度 |
|------|------|:---:|:---:|
| A: Frozen DINOv2 | [B, 256, 384] | 22M (frozen) | 快 |
| B: Trainable ResNet | [B, 256, 256] | 11M (trainable) | 中 |
| C: No encoder (查表) | [B, D] | 极小 | 极快 (但风格差) |

**推荐 A**: 冻结 DINOv2 作为初始 encoder. 后续可替换为 B.

### 4.3 Cross-Attention 层数 → **需要实验**

| 选项 | 注入位置 | 预期效果 |
|------|---------|------|
| A: 仅 bottleneck | 只在 UNet 最深层做 CrossAttn | 全局风格调制 |
| B: 每层 | 所有 decoder 层都做 | 多尺度风格注入 |
| C: decoder 层 | 只在 decoder 做 | 风格+内容对齐 |

**推荐 B**: 多层注入 → 粗尺度学全局色调, 细尺度学局部笔触.

### 4.4 结构保持机制 → **需要实验**

如果 Independent Coupling 下 LPIPS > 0.45:

| 选项 | 做法 | 成本 |
|------|------|:---:|
| A: Content Loss | $L_{\text{content}} = \text{L1}(\hat{x}_1, x_c)$ | 单步, 无展开 |
| B: PC Solver | 训练无约束, 推理时校正 | 推理时慢 |
| C: AdaLN content | content 特征也通过 AdaLN 注入 | 中等 |

**推荐 A**: 训练时用单步预测 $\hat{x}_1 = x_t + (1-t)v_\theta$ 算 content L1. 权重 $w \approx 0.1$.

---

## 五、训练循环

```python
for batch in dataloader:
    z_c = batch["z_content"]          # [B, 4, H, W]
    z_s = batch["z_style"]            # [B, 4, H, W]  
    s_img = batch["style_image"]      # [B, 3, H_img, W_img]  ← 注意: 风格图作为图像输入
    
    t = torch.rand(B, 1, 1, 1)
    z_t = (1 - t) * z_c + t * z_s     # Independent Coupling: 随机配对
    
    v_pred = model(z_t, t, style_image=s_img)
    v_true = z_s - z_c
    
    loss_fm = MSE(v_pred, v_true)
    
    # 可选: 单步预测 content loss
    pred_z1 = z_t + (1 - t) * v_pred   # 从 z_t 直接预测 z_1
    loss_content = L1(pred_z1, z_c)    # 结构约束
    
    loss = loss_fm + 0.1 * loss_content
    loss.backward()
```

---

## 六、与当前实现的关键差异

| 当前实现 | 重构方案 | 为什么 |
|---------|---------|--------|
| `style_code + time_code` | 独立 AdaLN(time) + CrossAttn(style) | 解耦 |
| learned tokens K,V | StyleEncoder 空间特征 K,V | 真实风格信息 |
| Minibatch OT 匹配 | Independent Coupling | 稳定目标 |
| Terminal SWD ODE 展开 | 移除 (单步 MSE 足够) | 消除梯度爆炸 |
| Tokenizer Embedding lookup | StyleEncoder | 泛化能力 |
| 大量 heuristic losses | 仅 FM loss + 可选 content loss | 极简 |
