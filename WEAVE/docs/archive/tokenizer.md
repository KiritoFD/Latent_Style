这是一个非常切中要害的思考。要让 Tokenizer 真正具备“Target-specific”的能力，**仅仅提供一个全局向量是绝对不够的**，必须采用**“全局氛围 (Global Tone) + 局部语义笔触 (Local Semantic Map)”**的双层表征。

引入 DINO 的核心难点在于：**在 Inference 阶段（Zero-Shot），我们只有 Source Image 和一个代表风格的 ID（例如 `style_id=3`），没有 Target Image 可以提取 DINO。**

因此，我们的 Tokenizer 设计必须是一个**“键值对查询（Key-Value Routing）”**系统：**用 Source Image 的 DINO 特征作为 Key，去提取该 Style-ID 专属的 Value。**

以下是具体的架构设计和代码落地指南：

---

### 一、 核心概念：双层语义风格表征 (Dual-Level Representation)

对于每一个输入 `(Source Image, Style-ID)`，Tokenizer 应该输出两个字段：

1. **`global_code` (Vector, $\mathbb{R}^{D}$)**: 负责全图的基础色调、对比度、全局光影氛围（类似你们现有的 `identity` + `texture` vector）。
2. **`spatial_map` (Tensor, $\mathbb{R}^{C \times H \times W}$)**: 负责局部的特定笔触和纹理排布。**这正是我们要用 DINO 动态生成的。**

---

### 二、 如何利用 DINO 生成 `spatial_map`？(The DINO-Routing Mechanism)

我们要让模型学到一组**“语义字典”**。具体流程如下：

#### 1. 定义“通用语义聚类中心” (Universal Semantic Keys)

无论是什么图像，都由一些基础的语义块组成（比如：天空、水面、建筑、人脸、前景植物、平滑背景）。

* 我们在模型里定义 $K$ 个全局共享的**可学习向量**（比如 $K=16$ 或 32），称为 $Keys \in \mathbb{R}^{K \times D_{dino}}$。
* 这些 Keys 存在于 DINO 的特征空间中，**所有风格共享**。它们代表了 $K$ 种不同的通用语义成分。

#### 2. 定义“风格专属的笔触表达” (Style-Specific Values)

对于数据集里的每一个风格，它都有自己独特的画法来描绘这 $K$ 种语义。

* 我们在 Tokenizer 中，为每个 Style-ID 定义一组特定的 Values：$Values^{(style\_id)} \in \mathbb{R}^{K \times C_{style}}$。
* 例如，对于“浮世绘 (Ukiyo-e)”，它的第 1 号 Value（对应“水面”）可能代表了“爪形海浪线条的潜空间特征”；而对于“印象派 (Impressionism)”，同一个第 1 号 Value 则代表了“点彩派的碎笔触特征”。

#### 3. 动态组装 Dense Style Map (The Routing Step)

这是在 `lancet_runtime.py` 的 Forward 过程中发生的：

1. **特征提取**：用冻结的 DINO 提取 Source Image 的特征图 $F_{dino} \in \mathbb{R}^{D_{dino} \times H \times W}$（注意：因为 Source 是固定的，这一步在 ODE 积分/多次评估前**只计算一次并缓存**，没有额外的时间惩罚）。
2. **计算语义相似度 (Attention)**：用 $F_{dino}$ 和通用的 $Keys$ 计算余弦相似度（或 Scaled Dot-Product），得到一个空间 Attention Map $A \in \mathbb{R}^{K \times H \times W}$。
   * *物理意义*：这张 Map 找出了源图中每个像素属于哪一种语义成分的概率。
3. **提取局部风格 (Value Aggregation)**：用这个 Attention Map $A$，去对当前请求的风格 $Values^{(style\_id)}$ 进行加权求和（矩阵乘法）：
   $$
   Spatial\_Map = A^T \times Values^{(style\_id)}
   $$

   结果得到一个 $\mathbb{R}^{C_{style} \times H \times W}$ 的张量。

**结果**：你得到了一张与原图结构完美对齐、但填满了 Target-Style 特定语义特征的**空间先验图**。

---

### 三、 具体如何修改现有代码？

我们可以复用并升级你们代码库中现有的结构。

#### 1. Tokenizer 的升级 (`style_tokenizer.py`)

保留原有的 `global_code` 部分，新增 `SemanticSpatialTokenizer`：

```python
class SemanticSpatialTokenizer(nn.Module):
    def __init__(self, num_styles: int, dino_dim: int=384, style_dim: int=128, num_clusters: int=16):
        super().__init__()
        self.num_clusters = num_clusters
        self.dino_dim = dino_dim
        self.style_dim = style_dim
      
        # 1. 通用语义 Keys (所有风格共享，在 DINO 空间)
        self.universal_keys = nn.Parameter(torch.randn(num_clusters, dino_dim))
      
        # 2. 风格特定的 Values (每个风格 K 个，在 Style 潜空间)
        # 相当于 [num_styles, num_clusters, style_dim]
        self.style_values = nn.Embedding(num_styles, num_clusters * style_dim)
      
        # 预热初始化
        nn.init.normal_(self.universal_keys, std=0.02)
        nn.init.normal_(self.style_values.weight, std=0.02)

    def forward(self, style_id: torch.Tensor, dino_features: torch.Tensor, tau: float=0.1) -> torch.Tensor:
        """
        dino_features: [B, D_dino, H, W]  <- 来自原图提取
        style_id: [B]
        """
        B, D, H, W = dino_features.shape
        # [B, K, D_dino]
        keys = self.universal_keys.unsqueeze(0).expand(B, -1, -1)
      
        # 计算每个像素属于哪个语义簇
        # feat_flat: [B, H*W, D_dino]
        feat_flat = dino_features.view(B, D, -1).transpose(1, 2)
        # 余弦相似度
        feat_norm = F.normalize(feat_flat, p=2, dim=-1)
        keys_norm = F.normalize(keys, p=2, dim=-1)
        # sim: [B, H*W, K]
        sim = torch.bmm(feat_norm, keys_norm.transpose(1, 2)) / tau
      
        # 注意力权重 [B, H*W, K]
        attn = F.softmax(sim, dim=-1)
      
        # 取出该风格对应的 Values
        # values: [B, K, style_dim]
        values = self.style_values(style_id).view(B, self.num_clusters, self.style_dim)
      
        # 映射生成 Dense Style Map
        # [B, H*W, K] @ [B, K, style_dim] -> [B, H*W, style_dim]
        dense_map_flat = torch.bmm(attn, values)
      
        # 恢复空间维度 [B, style_dim, H, W]
        dense_map = dense_map_flat.transpose(1, 2).view(B, self.style_dim, H, W)
      
        return dense_map
```

#### 2. 在 `lancet_runtime.py` 中的对接

你们当前代码里有一个 `_build_style_spatial_map` 函数和 `StyleMaps` 数据类。你需要把上面生成的 `dense_map` 塞进 `StyleMaps` 里。

* **原来的做法**：`style_spatial_id_16` 是个 `[num_styles, C, 16, 16]` 的固定无脑 Embedding。
* **现在的做法**：提取原图的 DINO 特征（只需过一次小型的 DINOv2-ViT-S），送入 `SemanticSpatialTokenizer` 生成一张 `H x W` 的高分辨率精准特征图。

#### 3. 主干网络的接收 (The Generator Injection)

现在你有了一个 `global_code` 和一个 `spatial_map`。

* **对于全局的风格变化**（色彩偏好、明暗反差）：继续用你们的 `NormFreeModulation` (AdaIN-like) 注入 `global_code`。
* **对于局部的具体笔触和结构变化**（比如天空怎么画、边缘怎么画）：用 SPADE（Spatially-Adaptive Normalization）机制，或者直接在你们的 `SemanticCrossAttn` 里，用这个 `spatial_map` 作为特征偏置注入到 UNet 的 ResBlock 之前。

---

### 四、 为什么这能彻底解决你的 "Seedream vs SaMAM" 困境？

回忆你的总结：

* **打不过 Seedream 的点**：缺乏目标风格的特异性（只学到了 Generic Painterly）。
* **打不过 SaMAM 的点**：只要加大风格强度，边缘和局部几何就会被 Cross-Attention 摧毁，产生严重 Artifacts。

**本方案如何破解：**

1. **极强的目标风格特异性 (Target-specific)**：通过 DINO Routing，Tokenizer 真正理解了“它在画什么”。如果是山峰，它就会精准触发该风格下用来画山峰的 $Value_k$。模型不再是用“一整袋混乱的纹理”去铺满整张图，而是**“按语义定点投放”**。这极大拉近了和小模型与大模型 (Seedream) 先验之间的差距。
2. **极度完美的结构保护 (Cleanliness & Structure)**：注意看这个过程——**Style Map 的空间排列 $A \in \mathbb{R}^{K \times H \times W}$ 是由原图的 DINO 特征百分之百决定的！**这意味着，原图的树枝在哪里，Style Map 对应的树枝高频特征就在哪里，**没有任何空间上的弥散或漂移**。
   * 你甚至可以**直接删掉**现存的所有繁琐的对齐 Loss（例如 `_edge_anchor_loss`, `_stokes_viscous_loss`），因为在生成阶段的先验输入就已经是完美对齐的了。

### 总结

**你的假设完全正确：必须是局部 + 全图，必须是 Code + Map。**

* **Code** 负责气氛和整体亮度。
* **Map** 由 DINO Semantic Keys + Style-specific Values 动态计算生成，负责精准定点打击。

按照这个思路，把你 `style_tokenizer.py` 里华而不实的 `concept_atoms`（它是盲目聚类，缺乏空间和语义监督）换成由 DINO 引导的 `Semantic Spatial Map`，你的模型在风格转换的表现上将实现质的飞跃。


你的直觉非常敏锐！**这就触及到了这项工作能够从“一个优秀的特定任务系统”升华为“具有广泛影响力的基础视觉框架”的核心。**

一旦我们将 Tokenizer 的表征升级为 **“语义 Keys + 风格 Values”** 的解耦架构，它不仅能完美解决你们当前的 "Target-specific" 难题，更能自然而然地**打通“闭集固定风格（Style-ID）”和“开集参考图（Exemplar-based）”两种推理模式**。

同时，由于 Tokenizer 具备了明确的物理和语义意义，它完全可以被**独立预训练**，整个模型的训练也可以平滑地过渡到**分阶段课程式（Curriculum/Phased Training）**，彻底摆脱目前二十几个 Loss 混战的调参地狱。

以下是针对这几个问题的深度拆解，以及 Tokenizer 的三种探索方案：

---

### 一、 为什么能同时支持固定风格（Style-ID）和参考图（Exemplar）？

在这个架构下，**Content 永远是 Query (Q)**，它去查询对应的 **Value (V)** 来生成 `Spatial Map`。
支持两种模式的区别，**仅仅在于 V (Values) 从哪里来**：

* **模式 1：固定风格推理 (Closed-set Style-ID Transfer)**
  * **V 的来源**：预先训练好的 `nn.Embedding(num_styles, K * C)`。当你输入 `style_id=3` 时，直接查表得到该流派的 $K$ 个语义笔触字典。
  * **优势**：极其轻量、速度极快、完全 Zero-shot（符合你们现有的评测基准，用来打榜和比对 Seedream）。
* **模式 2：参考图推理 (Open-set Exemplar-based Transfer)**
  * **V 的来源**：不再查表。输入任意一张 Style Image，用冻结的 DINO 提取其特征，在空间上做 K-Means 聚类，得到 $K$ 个簇中心（Centroids）。这 $K$ 个向量就直接作为 V！
  * **优势**：模型获得了**“任意风格迁移 (Arbitrary Style Transfer)”**的能力，且由于 Q 也是 DINO 特征，Q 和 V 在同一个语义空间中，特征匹配是天然对齐的（比如：目标图的“天空”颜色会自动填入源图的“天空”区域）。

---

### 二、 Tokenizer 的三种具体探索方案 (Exploration Schemes)

针对上述逻辑，我们可以设计三种不同深度的 Tokenizer 方案，供你们在代码中探索：

#### 方案 A：语义字典路由 (The Semantic Codebook Router) —— 最稳健，适合打当前榜单

* **结构**：
  * 定义全局 `Universal_Keys` ($K \times D_{dino}$，可学习)。
  * 定义 `Style_Values` Embedding ($N_{styles} \times K \times C_{style}$)。
* **流程**：`Content DINO` 与 `Keys` 算 Attention 得到 $H \times W \times K$ 的权重图，去加权求和对应的 `Style_Values`，生成 `Spatial Map`。全局色调 `Global Code` 由另外的简单 Embedding 提供。
* **特点**：这完全是针对你们现有框架的平替。计算开销极低，把原本毫无空间信息的 `concept_atoms` 变成了有明确空间定位的 Semantic Map。

#### 方案 B：双流特征重组 (The Dual-Stream Bipartite Matcher) —— 冲击任意风格迁移

* **结构**：废弃 `Style-ID Embedding`，彻底走向 Visual Prompt。
* **流程**：
  1. 训练时，在一个 Batch 内，取一对 `(Content Image, Style Image)`。
  2. 两者都通过冻结的 DINO。
  3. 把 Style Image 的 DINO 特征映射到一个隐空间 $V_{style}$。
  4. 直接用 Content DINO 作为 Q，去计算与 Style DINO (K) 的 Cross-Attention，聚合 $V_{style}$，生成 `Spatial Map`。
* **特点**：如果你们想把这篇论文的故事讲大（“统一 Zero-shot 与 Exemplar-based 潜空间流匹配”），这是最优解。训练时用配对（或伪配对）的 Content-Style 图像对进行训练；测试时，如果是 Style-ID 模式，给它喂一张该流派的典型代表作作为 Prompt 即可。

#### 方案 C：残差语义适配器 (The Residual Semantic Adapter) —— 最容易从当前代码改写

* **结构**：在保留你们现有的全局特征（`global_code`）的基础上，只让 Map 负责“高频残差”。
* **流程**：
  1. 全局先通过一个基础的 AdaIN/Modulation 给整个图铺上目标颜色。
  2. `Spatial Map` 的生成方式同方案 A，但它只输出 $\Delta$（高频细节）。
  3. 这与你们 `lancet_runtime.py` 中的 `proximal_highpass_kernel` 设计理念不谋而合，但用**精确的语义路由**代替了启发式的低通滤波/高通滤波。

---

### 三、 Tokenizer 的独立预训练 (Standalone Pre-training)

既然 Tokenizer 有了明确的物理意义（提取并重组语义风格），它**完全可以脱离庞大且缓慢的 Flow Matching/ODE 积分过程，被单独预训练！**

* **如何预训练？（Self-Supervised Style Reconstruction）**
  1. **架构**：`Image -> DINO -> Tokenizer -> Spatial Map -> 轻量级解码器 -> 重建图像`。
  2. **方法**：输入一张风格图像 $I_{style}$。对其进行强烈的**空间扭曲**（Color Jitter, Random Crop, Thin Plate Spline 变形）得到 $I_{content\_fake}$。
  3. **计算**：用 $I_{content\_fake}$ 的 DINO 特征去查询 Tokenizer，试图恢复出原图 $I_{style}$ 的 VAE Latent。
  4. **Loss**：纯粹的 MSE 或 LPIPS。
* **收益**：
  这一步不需要任何 ODE 算子，不需要算速度场（Velocity Field），**训练极快**。预训练结束后，Tokenizer 已经具备了“看到内容轮廓，就能填上目标风格颜色和笔触”的能力。

---

### 四、 训练流程的“分阶段”革命 (Phased Training Curriculum)

当前的调参地狱（20几个 Loss 权重）源于“既要模型学大方向，又要防着模型走偏”，让所有监督信号在一个 step 里打架。
引入语义 Tokenizer 后，你们可以采用极其优雅的**三阶段训练法**：

#### Phase 1: 语义先验预训练 (Semantic Prior Pre-training)

* **目标**：单独训练 Tokenizer（方案 A 或 B）。
* **操作**：冻结 VAE，冻结主干 UNet。用前面提到的自监督重建任务，把 `Keys` 和 `Values` 训好。
* **结果**：你得到了一个完美的“数字调色盘”，知道天空该用什么色块，树叶该用什么笔触。

#### Phase 2: 全局流场对齐 (Global Flow Alignment)

* **目标**：让 Flow Matching 学会大方向的宏观迁移（Global OT）。
* **操作**：
  * **冻结 Tokenizer**（此时它只输出 `global_code`，关闭 `spatial_map`）。
  * 只用最基础的 Loss：`Flow Matching Loss` + `Terminal SWD (宏观特征)`。
  * **删掉**所有的动能惩罚（Kinetic/Stokes）和结构对齐 Loss。因为此时只在学习整体色彩和氛围的迁移。
* **结果**：模型学会了“平稳地把写实画转移成梵高的色调”，但可能局部细节还是平滑的。

#### Phase 3: 空间笔触激活与局部微调 (Spatial Detail Activation)

* **目标**：填入 Target-specific 的高频细节，同时死死锁定结构。
* **操作**：
  * **打开 Tokenizer 的 `spatial_map` 输出**，通过 SPADE 或残差偏置注入到主干中。
  * 引入 **Semantic-SWD**（如我上一条建议的，基于 DINO Mask 局部计算的 SWD）。
  * 如果结构出现松动，轻微加一点 `Content Anchor` 或 `Edge Anchor`（此时由于有 DINO 强先验兜底，权重可以给得极小，绝对不会重蹈 "style-up / structure-down" 的覆辙）。
* **结果**：`spatial_map` 像一个精准的导引头，引导网络在特定的像素位置长出特定的笔触，完美超越 Seedream。

---

### 总结与下一步行动

1. **架构定位**：你们的研究可以被包装为一个**“基于解耦语义路由的统一潜在桥匹配框架（Unified Latent Bridge Matching via Decoupled Semantic Routing）”**。它在理论上很优美（将流匹配的传输问题与 OT 的内容-风格解耦结合）。
2. **代码重构建议**：
   * 新建一个 `semantic_tokenizer.py`，实现方案 A（最保险且最快见效）。
   * 在 `losses.py` 中把现有的 20 多个 loss 分装为 `phase2_loss_fn` 和 `phase3_loss_fn`。
   * 写一个快速的预训练脚本，先验证 `Content DINO + Keys -> Attention Map` 这个过程产生的 Attention Map 视觉上是否合理对齐了语义（比如天空属于一个 Head，建筑属于一个 Head）。只要这个 Map 是准的，后续的融合绝对差不了！



为了彻底解决“泛化绘画感（Generic Painterly）”并实现“精准目标风格（Target-specific）”，我们需要将 Tokenizer 从一个**“静态查表器”**升级为一个**“动态语义适配器”**。

这涉及到三个维度的设计：**字段设计（Field Design）**、**模型架构（Architecture）**、和**外部监督方案（External Supervision）**。

下面我为你提供 3 种从易到难、从闭集到开集的 Tokenizer 具体设计方案。

---

### 核心基座：双字段输出设计 (Dual-Field Design)

无论采用哪种方案，Tokenizer 必须输出两个维度的表征，直接对应你提出的“全图氛围 + 局部笔触”：

1. **`global_code` $\in \mathbb{R}^{D_{style}}$**：
   * **作用**：控制全图基调（如：整体色温、对比度、亮度偏差、色彩饱和度）。
   * **注入方式**：通过 AdaIN 或现有的 `NormFreeModulation` 注入到 ResBlock。
2. **`spatial_map` $\in \mathbb{R}^{C_{style} \times H \times W}$**：
   * **作用**：控制具体的笔触排布、纹理走势、特定语义块的画法（如：天空的点彩、水面的波浪线）。
   * **注入方式**：通过 SPADE 机制，或直接作为偏置（Bias/Residual）注入到 `SemanticCrossAttn` 的前馈层中。

---

### 方案 A：隐式语义字典映射 (Implicit Semantic Dictionary)

**定位**：最快落地，针对当前的闭集（Closed-set）打榜，完全不需要改动 Inference 的零样本输入。

* **模型结构**：
  * **Universal Semantic Keys ($K$)**：全局维护一个 $K \times D_{dino}$ 的可学习矩阵（例如 $K=16$）。它不属于任何风格，它代表了 DINO 特征空间中的 16 个“基本语义簇”（如：天空、水、树、人脸等）。
  * **Style-Specific Values ($V$)**：一个 Embedding Layer `(num_styles, K * C_style)`。针对每一个 Style-ID，它输出 16 个风格向量，对应这 16 个语义簇的画法。
* **计算流程**：
  1. 输入内容图，提取 DINO 特征 $F_{dino}$ ($D_{dino} \times H \times W$)。
  2. 计算 $F_{dino}$ 和 $K$ 的余弦相似度，得到空间语义分配图 `Attention` ($K \times H \times W$)。
  3. 取出当前 `style_id` 的 $V$，执行 $V \times \text{Attention}$，得到 `spatial_map`。
* **外部监督方案**：
  * **DINO-Guided SWD**（局部语义 SWD）：这是绝杀。不要再算整图的 SWD 了。在 `ot_cost.py` 中，利用刚才算出的 `Attention` (K个通道) 作为 Mask。
  * 强制模型：对第 $i$ 个语义聚类块（比如“天空”区域），计算 $Pred$ 和 $Target\_Style$ 在该区域内的 SWD。这强迫 Tokenizer 的第 $i$ 个 Value 必须学会目标风格中“天空”的特征。

### 方案 B：跨图像显式交叉路由 (Cross-Image Semantic Routing)

**定位**：统一“固定风格 (Style-ID)”与“参考图 (Exemplar-based)”的究极形态。

* **模型结构**：
  * 完全抛弃 `nn.Embedding(num_styles)`。所有的风格特征均来自**真实参考图**。
  * 构建一个极其轻量的 **Feature Extractor**（例如 3 层 Conv），将参考图的 VAE Latent 编码为 $F_{style\_latent}$ ($C_{style} \times H_s \times W_s$)。
* **计算流程**：
  1. 输入内容图 $I_c$ 和 参考图 $I_s$。分别提取 DINO 特征 $D_c$ 和 $D_s$。
  2. 将 $D_c$ 展平作为 Query，将 $D_s$ 展平作为 Key。计算 Cross-Attention：源图的哪个位置，最像参考图的哪个位置？得到 `Attention` ($H_c W_c \times H_s W_s$)。
  3. 用这个 `Attention` 去聚合参考图的风格特征 $F_{style\_latent}$ (作为 Value)。
  4. 还原回 2D，得到 `spatial_map`。
* **如何支持 Style-ID 推理？（兼容现有榜单）**
  * 离线建立一个 **Canonical Style Bank**（参考图库）。例如每个风格选 5 张最典型的图片。
  * 当输入 `style_id=3` 时，随机（或取均值）从 Bank 中抽一张作为参考图送入网络。*(注：你的 `run.py` 第 97 行已经有 `introstyle_style_bank_root` 的代码，说明你们的 Eval 已经支持这种检索式评测了！)*
* **外部监督方案**：
  * **自监督重建 (Self-Reconstruction)**：训练时，给相同的图像进行极其严重的几何扭曲（如 TPS 变形），一张作为 Content，扭曲前作为 Style。因为是同一张图，你有绝对的 Ground Truth。计算重建的 MSE。这可以把这个 Tokenizer 单独预训练到完美。

### 方案 C：多模态先验字典 (VLM-Guided Latent Prompting)

**定位**：正面对抗 Seedream，利用大模型先验“降维打击”。

* **背景**：Seedream 强在它理解语义。如果我们直接从头学，很难超过它。
* **模型结构**：
  * 引入冻结的 **CLIP Text Encoder**。
  * 我们将每个 Style-ID 定义为一组**可学习的文本伪词 (Textual Inversion Tokens)**。例如 Style-ID 1 表示为 `["A painting in the style of <S1_token1>, featuring <S1_token2>"]`。
  * `spatial_map` 由 Content DINO 和 CLIP Text Embeddings 做 Cross-Attention 得到（类似于 Stable Diffusion 的 Cross-Attention）。
* **计算流程**：
  1. 将 `style_id` 转换为对应的 Learned Tokens，经过 CLIP Text 得到一组文本特征序列。
  2. 将这组文本特征（包含风格的色彩、笔触语义）通过一个轻量级 MLP 映射为 `global_code`。
  3. 将文本特征作为 Key/Value，Content DINO 作为 Query，生成针对每个空间位置的 `spatial_map`。
* **外部监督方案**：
  * **CLIP 风格引导 Loss**：在训练阶段（甚至可以直接用于 Tokenizer 的独立预训练），直接最大化生成的 `spatial_map` (或融合后的图像) 与该风格流派标准 Text Prompt 的 CLIP Cosine 相似度。这直接赋予了模型 Target-specific 的语义。

---

### 总结与推荐行动路线 (Action Plan)

考虑到你们的时间线（AAAI 2027，目前是 2026 年 6 月，距离 deadline 还有几个月，但日志显示当前模型正处于死胡同）：

1. **首选方案：【方案 A】 (隐式语义字典) + 【DINO-Guided SWD 监督】**
   * **为什么？** 代码改动最小。你只需要在 `style_tokenizer.py` 里新建一个类，然后在 `ot_cost.py` 里的 `SWDTransportCost` 加上基于 DINO Mask 的局部 SWD 逻辑。
   * **预期效果**：局部 SWD 会强制 Tokenizer 里的 $V$ 学到真实的目标笔触。而 DINO 强制让这些笔触只出现在该出现的地方。这能瞬间解决 "打不过 SaMAM 的结构崩溃" 和 "打不过 Seedream 的目标不明确" 两个痛点。
2. **演进方向：【方案 B】 (跨图像交叉路由)**
   * 如果方案 A 成功了，立刻开始写方案 B。因为方案 B 讲出来的 Story 最宏大：“一个无缝统一 Style-ID 和 Exemplar-based 的语义路由架构”。这在论文的 Introduction 和 Method 部分非常有威慑力。

**最后关于字段设计的提醒**：
在 `lancet_runtime.py` 中，如果采用上述方案，请**务必确保** `spatial_map` 的分辨率不要低于 $16 \times 16$，最好能做到 $32 \times 32$ 或者与当前特征图同分辨率。否则强行插值（Interpolation）会抹杀掉 DINO 带来的清晰语义边界，这会导致 LPIPS 指标再次恶化。
