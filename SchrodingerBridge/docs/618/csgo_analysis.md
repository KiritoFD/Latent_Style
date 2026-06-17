# CSGO (NeurIPS 2025) 深度分析

> 论文: arXiv 2408.16766 (13pp) | 代码: github.com/instantX-research/CSGO | HF: InstantX/CSGO

---

## 一、核心方法

### 数据构建管线

CSGO 的第一个贡献是构建 IMAGStyle 数据集:
- 内容图: MSRA10K + ImageNet-Sketch (11K images)
- 风格图: WikiArt + Midjourney 生成 (10K images)
- 生成: B-LoRA (content LoRA + style LoRA 合并) × SDXL → 自动清洗
- 清洗: CAS (Content Alignment Score) = $\|Ada(\phi(C)) - Ada(\phi(T))\|^2$
- $Ada(F) = (F - \mu(F)) / \sigma(F)$ (AdaIN 内容提取, 去风格化)
- 取 CAS 最低的生成图为正样本
- 最终: **210K 三元组** (content, style, stylized)

### 模型架构

CSGO 基于 SDXL, 使用三个关键模块:

**内容控制**:
1. ControlNet (Tile): 注入 up-sampling blocks → $D'_i = D_i + \delta_c \times C_i$
2. 解耦 cross-attention: CLIP 编码内容图 → 投影层 → 额外 cross-attn 注入 down blocks
3. 训练: $\lambda_c=1.0$, 推理: $\delta_c=0.5$

**风格控制**:
1. Perceiver Resampler: $\mathcal{F}(S) \in \mathbb{R}^{o \times d} \to \mathcal{F}(S)' \in \mathbb{R}^{t \times d}$
2. 额外 cross-attention 注入 up-sampling blocks
3. 关键设计: **也在 ControlNet 分支中注入风格特征** (防止 ControlNet 泄漏内容风格)

### 训练与推理
- SDXL base, ViT-H image encoder, Tile ControlNet
- 8×H800 (80GB), batch 20/GPU, 80000 steps
- Token数 t=16 (Resampler)
- Drop rate: text/content/style = 0.15
- CFG classifier-free guidance

---

## 二、与我们方法的对比

### 根本差异

| 维度 | CSGO | 我们 |
|------|------|------|
| 基础模型 | SDXL (扩散模型, RGB 空间) | VAE latent (潜空间, 4×64×64) |
| 训练数据 | IMAGStyle 210K 三元组 (人工生成) | WikiArt 18888 原始图像 (无配对) |
| 风格表征 | Perceiver Resampler + CLIP 编码 | Tokenizer style_values lookup |
| 内容保持 | ControlNet Tile + 解耦 cross-attn | TopoGate attention blending |
| 训练成本 | 8×H800, 80K steps | 1×3060, b32, 24 epochs |
| 无配对 | 假配对 (LoRA 生成) | 真无配对 (OT 动态匹配) |

### CSGO 的优势

1. **端到端训练, 推理极快**: 一次训练, 推理时不需要 inversion/optimization. SDXL 推理 ~2s/image.
2. **三元组监督**: 有 ground truth stylized image → 训练目标明确
3. **工业化程度高**: 210K 数据 + 8×H800 → 如果能承受成本, 质量上限高于我们

### CSGO 的局限

1. **数据依赖**: 需要 210K 三元组 → 风格迁移数据集构建是它们的主要贡献
2. **三元组质量**: 用 B-LoRA 生成的假 GT → 不是真配对 → 和我们的 OT 匹配面临同样的"配对质量"问题
3. **不处理无配对**: 本质上是有监督训练, 不是真正的 unpaired style transfer

### 我们的优势

1. **纯潜空间**: 只在 4×64×64 latent 上操作 → 计算成本低
2. **真无配对**: OT 动态匹配 → 不需要三元组
3. **训练门槛低**: 单卡 3060 可训练
4. **LPIPS 更好**: 0.31 vs CSGO 0.50 (在 StyleGallery 的评测中)

---

## 三、对我们的启发

### 启发 1: 内容保持的双重控制

CSGO 用了两种内容保持机制:
- ControlNet Tile: 保持空间布局
- 解耦 cross-attention: 保持语义信息

**对我们的意义**: 我们的 TopoGate 只做了 attention 层面的内容保持. 可以考虑增加类似的结构级保持——比如在 latent 解码时增加 skip connection 的权重控制.

### 启发 2: Perceiver Resampler 风格编码

CSGO 用 Perceiver Resampler 将 CLIP 特征映射到固定数量的 token → 风格信息被压缩为紧凑表示.

**对我们的意义**: 我们的 tokenizer 用 embedding lookup (style_id → fixed vector). CSGO 用 Resampler (style_image → learnable projection). 后者可以从风格图像中提取**实例级别的风格信息**, 而不是类别级别的. 这是我们 tokenizer 的根本缺陷.

### 启发 3: CAS (Content Alignment Score)

CSGO 的 CAS 用 AdaIN 去风格化后比较内容:
$$CAS = \|Ada(\phi(C)) - Ada(\phi(T))\|^2$$

**对我们的意义**: 我们可以把这个指标用于**OT 匹配的代价矩阵**: 用 AdaIN 去风格化后比较内容 → 这是比欧氏距离更好的内容相似度度量. 避免了"颜色相似导致错误匹配"的问题.

### 启发 4: 风格注入的位置设计

CSGO 在 ControlNet 分支中也注入风格 → 防止 ControlNet 的内容特征泄漏源图风格.

**对我们的意义**: 我们的 spatial_map 只在 body_blocks 中注入. 如果 UNet 的 skip connection 也在传递源图风格, 我们需要在多个位置做风格注入. 这和我们之前讨论的"多尺度 TopoGate"一致.

---

## 四、对论文的启发

### Related Work 定位
CSGO 代表"端到端训练 + 三元组监督"范式. 我们代表"无配对 + OT 动态匹配 + 纯潜空间"范式.

对比点:
1. CSGO 需要 210K 人工生成的三元组 → 我们不需要
2. CSGO 用 SDXL (RGB 空间) → 我们用 VAE latent
3. CSGO 的 CAS 启发我们改进 OT 代价矩阵

### 可引用的技术
- CAS 作为内容保持度量: 引用 CSGO 的工作, 我们将 CAS 思想引入 OT 代价矩阵设计
- Perceiver Resampler: 引用其风格压缩设计, 解释为什么我们的查表 tokenizer 不够好
