# 620 信息流深度分析 — OT, SWD, Attention, Encoder

> 四个核心问题的理论分析. 每个问题给出多条可行实现路线及其理论依据.

---

## 问题 1: OT 要不要复用 Attention 信息?

### 当前

OT 配对用 DINO CLS cos-sim → 离线 top-K 语义匹配. Attention 和 OT **完全独立** — attention 只在训练时做 CrossAttn (content Q → style K,V), 不参与配对决策.

### 理论分析

**Attention 天然是"最优传输计划"**. CrossAttn 公式:

$$A = \text{softmax}(Q_c K_s^T / \sqrt{d})$$

$A_{ij}$ 解释为"内容像素 i 从风格 token j 取纹理的概率". 这是**可微的软传输计划** — 和 Sinkhorn plan 在数学上同类.

**因此 Attention map 天然提供了"当前模型认为哪些内容像素应该匹配哪些风格 token"的信息**. 如果把这个信息反馈到 OT 配对中:

$$\text{Cost}(x_c, x_s) = \| A_{x_c} - A_{x_s} \|_F^2$$

两张图如果 CrossAttn 的 attention pattern 相似 → 它们的"被模型感知的结构"相似 → 应该配对.

### 三条路线

**路线 A: Attention-free OT (当前, DINO CLS 离线)**

优势: 离线计算, 稳定, 和训练解耦.
劣势: 只用全局语义, 没用到模型的"真实感知".

**路线 B: Attention-informed OT (训练中动态)**

每 N 个 step, 用当前模型的 CrossAttn attention map 的**统计量** (entropy, 空间分布的 spectral norm) 来更新配对. 不直接比较 $A$ 矩阵 (太昂贵).

```python
def attention_complexity(model, x):
    attn = model.get_last_cross_attn()      # [B, HW, 256]
    entropy = -(attn * attn.log()).sum(-1)  # [B, HW]
    return torch.stack([entropy.mean(), entropy.std(), entropy.max()])  # [B, 3]
```

配对: `Cost(x_c, x_s) = ||attention_complexity(x_c) - attention_complexity(x_s)||_2^2` — "结构复杂度和风格复杂度匹配".

**路线 C: Attention-guided SWD (不改变 OT, 改 SWD)**

SWD 的投影方向从"随机"变为"attention-informed": SWD 在 attention map 加权的分布上计算 — 高 attention 区域获得更多 SWD 权重:

$$\text{SWD}_{\text{attn}}(P, Q) = \text{SWD}(A \odot P, A \odot Q)$$

**推荐**: 路线 A 做 baseline (已有), 路线 C 进入实验 (见 P4). 路线 B 太昂贵 (需训练中重算 OT), 不优先.

---

## 问题 2: SWD 的设计 — 如何让模型在"内容相近"的块上学?

### 当前

单步 SWD 在整体 latent 上算: `SWD(ẑ₁, z_s)`. 比较的是**全局分布** — 所有像素混在一起.

### 问题

"天空"的 SWD 和"建筑"的 SWD 不应该混在同一个分布里. 天空需要匹配天空的笔触, 建筑需要匹配建筑的笔触.

### 三条路线

**路线 A: Per-Cluster SWD (fiber-wise, 616 提案)**

用 CrossAttn 的 attention map 做 soft mask, 把特征分成 K 个"语义区", 每个区单独算 SWD.

$$\mathcal{L}_{\text{SWD}} = \sum_{k=1}^K \text{SWD}(\text{Mask}_k \odot \hat z_1, \; \text{Mask}_k \odot z_s)$$

Mask 来自 attention map 的 argmax 或 top-k. K 可以是 DINO spatial token 数 (256, 太细) 或 attention head 数 (4-8, 合适).

**路线 B: Multi-scale SWD**

在不同分辨率分别算 SWD:
- 64×64: 全局分布 → 全局风格
- 32×32: downsampled → 中尺度纹理
- 16×16: 进一步 downsampled → 粗尺度色调

$$\mathcal{L}_{\text{SWD}} = \sum_{s \in \{1, 0.5, 0.25\}} w_s \cdot \text{SWD}(\text{down}(\hat z_1, s), \text{down}(z_s, s))$$

**路线 C: Attention-weighted SWD (与问题 1 路线 C 相同)**

用 CrossAttn attention map 的 entropy map 作为 SWD 的 pixel 级权重 — 高 entropy 区域 (模型不确定该用什么纹理) 获得更高 SWD 权重 → 模型被强制在这些区域学好.

**推荐**: 路线 B (多尺度) 最稳健, 进入实验. 路线 A (per-cluster) 在 P3 中自然出现 (与多分辨率 CrossAttn 协同).

---

## 问题 3: 风格表征 — Encoder 设计

### 当前

DINOv2 frozen → 256 spatial tokens (16×16) + 1 CLS token. CrossAttn 的 K,V 来自 256 spatial tokens.

### 风格表征的三层需求

| 层次 | 信息 | DINO 提供? | 缺失? |
|------|------|:---:|------|
| 全局 | 色调, 亮度, 对比度 | ✅ CLS token | — |
| 区域 | "印象派的天空画法" | ✅ spatial tokens | 16×16=256 grid, 够用 |
| 局部 | "这个具体的笔触纹理" | ⚠️ | 256 tokens 对于 64×64=4096 pixels 太稀疏 |

**DINO 的局限**: spatial tokens 只有 256 个. 每个 token 覆盖了 4×4 的 latent pixel. 对于**笔触级别的纹理细节**, 这个分辨率不够.

### 三条路线

**路线 A: DINO 多尺度 (当前, 可改进)**

改进: 取 DINO 的多个中间层, 而不是单层. 浅层有更多纹理, 深层有更多语义.

```python
F_s = concat(DINO.layer[4], DINO.layer[8], DINO.layer[11])  # 多尺度特征
F_s = proj(F_s)  # [B, 256×3, D_model]
```

**路线 B: DINO + Trainable Local Encoder**

在 DINO 基础上, 加一个小的可训练 CNN 处理风格图的浅层特征 → 捕获高频笔触.

```python
dino_feat = DINO(style_image)              # [B, 256, 384] 语义+中尺度
local_feat = LocalCNN(style_image)         # [B, 64, 64] 高频纹理, 可训练
# CrossAttn K,V = concat(dino_proj, local_proj)
```

**路线 C: 完全可训练的 Style Encoder (类似 StyleShot)**

从 scratch 训练一个专用 Style Encoder, 输入风格图, 输出风格特征. 需要较多的风格训练数据. 当前 WikiArt 5 类 × 平均 3700 张 = 共 ~18K 张, 勉强够用.

**推荐**: 路线 A 立即测试 (DINO 多尺度). 如果 style 仍然 < 0.71, 尝试路线 B (加 local encoder). 路线 C 仅在数据扩充后考虑.

---

## 问题 4: 如何让模型在"内容相近的块"上学习?

### 核心问题

模型需要学会: "这个天空区域应该用这种笔触, 那个建筑区域应该用那种笔触". 这要求:

1. **空间定位**: Q 能区分"天空"和"建筑"
2. **风格选择**: K,V 能提供"天空的笔触"和"建筑的笔触"两种选项
3. **匹配精度**: attention map 正确地把"天空 Q"匹配到"天空 K"

### 条件分析

**条件 1 (空间定位)**: Q 来自 content feature. Content feature 经过 UNet encoder, 天然带有空间语义. 浅层有纹理定位, 深层有语义定位. ✅ 天然满足.

**条件 2 (风格选择)**: K,V 来自 DINO 风格特征. DINO 的 spatial tokens 也带有空间语义 — "天空区域的 DINO token" 和 "建筑区域的 DINO token" 有不同的特征. ✅ 天然满足.

**条件 3 (匹配精度)**: 这是关键瓶颈. Attention 能否学会"天空 Q 匹配天空 K"? 取决于:
- (a) Q 和 K 的特征空间是否对齐 (投影矩阵 $W_Q, W_K$ 的训练)
- (b) 训练信号是否明确 — 如果 SWD 是全局的, 模型分不清哪些 Q 该匹配哪些 K

**(b) 是当前最可能的瓶颈**: 全局 SWD 对所有区域的 Q→K 匹配给出同一个分布信号. 模型只能学到"平均匹配".

### 解决路线

**路线 A: Per-region SWD (即问题 2 路线 A)**

把风格分布匹配分解为 per-region 的任务 → 每个区域有独立的风格信号.

**路线 B: Attention entropy 正则化**

鼓励 CrossAttn attention map 接近 one-hot (高 confidence, 低 entropy) — 这样每个 Q 明确匹配一个特定的 K.

$$\mathcal{L}_{\text{ent}} = -\lambda \cdot H(A)$$

H 是 attention map 的熵. 负号使 H 降低. 但注意: 熵太低 → attention collapse.

**路线 C: Contrastive Q-K learning**

在 CrossAttn 前加一个对比损失: 同一语义区域的 Q 和 K 应该接近, 不同区域的应该远离. 类似 SimCLR 但在注意力层做.

$$
\mathcal{L}_{\text{contrast}} = -\log \frac{\exp(Q_i \cdot K_{i+} / \tau)}{\sum_j \exp(Q_i \cdot K_j / \tau)}
$$

其中 $K_{i+}$ 是 $Q_i$ 对应的"正确"风格 token (由 DINO 预配对决定).

**推荐**: 路线 A (per-region SWD) 优先 — 最低成本, 最高信号. 路线 B (entropy 正则) 作为备选.
