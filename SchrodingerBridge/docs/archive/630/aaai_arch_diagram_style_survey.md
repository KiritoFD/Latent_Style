# AAAI 论文主架构图视觉风格调研报告

> 调研对象：F:\papers\latent_style_cited_20260605\AAAI 目录下的 4 篇论文
> 调研重点：主架构图（Figure 1 / Figure 2 或对应总体框架图）
> 日期：2026-07-03

## 一、调研范围说明

本次调研阅读了以下 4 篇 AAAI 论文，并重点分析了各自的主架构图：

1. **AesFA**（Huang et al., AAAI 2024）— 主架构图为 **Figure 2**：The entire AesFA architecture for aesthetic feature-aware NST。
2. **S2WAT**（Xia et al., AAAI 2024）— 主架构图为 **Figure 5**：Overall pipeline of the proposed S2WAT（Figure 1/2 为问题示例与实验对比，非总体架构）。
3. **ArtBank**（Zhang et al., AAAI 2024）— 主架构图为 **Figure 3**：The overview of our proposed ArtBank（Figure 1/2 为生成样例）。
4. **Lancet / Latent Bridge Matching** — 主架构图为 **Figure 2**：Overview of Latent Bridge Matching。

以下逐篇总结其主架构图的视觉风格与表达技巧。

---

## 二、逐篇论文主架构图分析

### 1. AesFA — 频域双路径流程图

**图类型**：流程图式 + 模块图式混合。

**布局与结构**：整体采用**横向展开**，从左到右依次为 Content Image → Content Encoder / Aesthetic Feature Encoder → Kernel-Prediction Networks → Generator → Output Image。在编码器之后分出**上下两条并行路径**，分别用**蓝色箭头**和**绿色箭头**标注高频（high-frequency）与低频（low-frequency）特征处理流程，最终在 Generator 中融合。

**视觉元素**：
- **节点形状**：以圆角矩形模块为主，模块内部用颜色块区分不同子操作（AdaOct、OctConv、Up-sampling、Convolution）。
- **配色**：背景为纯白；蓝色与绿色箭头明确区分双频路径；输入/输出端嵌入真实图片缩略图（Content Image、Style Image、Output Image），增强直观性。
- **箭头**：实线箭头表示数据流向；图注说明“蓝色箭头 = 高频，绿色箭头 = 低频”。
- **图注位置**：位于图下方，采用常规 Figure Caption 形式。

**可借鉴点**：用颜色区分并行的多尺度/频域路径，并在输入输出端嵌入真实图像，使抽象流程具像化。

---

### 2. S2WAT — 分层 Transformer 模块图

**图类型**：模块图式，分块展示层级编码器、Transformer 解码器与注意力机制。

**布局与结构**：采用**横向分层 + 局部放大**的布局。主图分为 (a) Strips Window Attention Transformer Encoder（三层 Stage，横向展开）、(b) Strips Window Attention 细节、(c) Net Architecture of S2WAT（编码器-传输模块-解码器总览）、(d) Decoder、(e) Transformer Decoder Layer。整体呈现“由粗到细”的信息层级。

**视觉元素**：
- **节点形状**：以圆角矩形、虚线框模块为主；Stage 1/2/3 用虚线框包围，表达重复堆叠结构。
- **配色**：浅蓝、浅粉、浅绿区分不同功能区域；Patch Partition、Embedding、Attention Block 等使用淡色填充。
- **多尺度/Attention 表达**：通过 Stage 1→Stage 2→Stage 3 的逐步下采样箭头表达多尺度；用 Window Attention 的局部放大图（含 MLP、SpW Attention、Attn Merge）解释注意力机制。
- **箭头**：实线为主，部分使用双向或聚合箭头表达特征融合。
- **图注位置**：整体 caption 在图下方，子图 (a)-(e) 各自带小标题。

**可借鉴点**：用“主图 + 子图放大”的方式平衡整体 pipeline 与局部机制细节；用虚线框表达重复/堆叠模块。

---

### 3. ArtBank — 训练/推理双路径对比图

**图类型**：流程图式，明确区分训练与推理两条路径。

**布局与结构**：采用**左右两栏 + 底部分支**的复合布局。左栏为 Training（绿色调背景），右栏为 Inference；底部单独展示 Stochastic Inversion 子流程。整体纵向分层不明显，而是通过区域划分表达不同阶段。

**视觉元素**：
- **节点形状**：圆角矩形、表格/矩阵块（Implicit Style Prompt Bank）、扩散过程方块（Diffusion Process）等混合使用。
- **配色**：Training 区域使用浅绿色背景，Inference 区域使用浅米色/白色背景；Switch、Replace、Tokenizer 等关键模块用高亮色块突出。
- **训练 vs 推理表达**：用区域底色和标注文字“Training / Inference”直接区分；训练路径中包含可学习的参数矩阵和 artworks 集合，推理路径展示从 content image 到 stylized image 的完整扩散去噪流程。
- **箭头**：实线表示数据/控制流；虚线可能用于表示跨阶段的条件注入或可选替换。
- **图注位置**：位于图下方，对 Training、Inference、Stochastic Inversion 三部分均作说明。

**可借鉴点**：用背景色块清晰划分训练与推理；在扩散类方法中，把“可学习 bank”与“去噪推理”并置，突出方法核心贡献。

---

### 4. Lancet (Latent Bridge Matching) — 三带横向分层总览图

**图类型**：模块图式 + 公式化流程，强调 inference/training 的对比。

**布局与结构**：采用**三条横向色带**的纵向分层布局：
1. 顶层：Style Control（style ID → Style Tokenizer → style code）
2. 中层：Main Inference Path（content image → VAE Encode → latent → LANCET Vector Field → K-step Euler → VAE Decode → output）
3. 底层：Training（Supervision & Endpoint Construction），包含配对、OT+Sinkhorn、Transport supervision、SA-SWD terminal matching 等。

**视觉元素**：
- **节点形状**：圆角矩形模块，内部嵌入小图标（时钟表示 time、网络节点表示 semantic routing、U-Net 图形等）。
- **配色**：三条色带分别使用浅橙/浅黄、浅蓝、浅紫/浅红背景，一目了然；模块填充色与背景色形成轻微对比。
- **训练 vs 推理表达**：在图底部单独设置箭头图例——**实线箭头 = inference (active)**，**虚线箭头 = training (supervision)**；中层推理路径与底层训练路径通过虚线连接，表达训练只对推理提供监督约束。
- **箭头**：实线表示推理流，虚线表示训练监督；箭头从底层训练模块指向上层推理模块，清晰表达“训练为推理提供约束”的关系。
- **图注位置**：图下方，详细解释三层含义及虚实箭头约定。

**可借鉴点**：用横向色带 + 箭头图例系统化区分 inference 与 training；在 latent transport 类方法中，把“风格控制 / 推理 / 训练”三要素纵向分层，逻辑非常清晰。

---

## 三、可借鉴的绘图原则（5-8 条）

综合以上四篇 AAAI 论文的主架构图，可提炼以下绘图原则：

1. **明确区分 inference 与 training 路径**
   - 方法：用实线/虚线、不同背景色或分区标注区分训练与推理。
   - 示例：Lancet 的“实线 = inference，虚线 = training”图例；ArtBank 的左右分栏。

2. **用颜色编码并行路径或尺度/频域分支**
   - 方法：为高频/低频、内容/风格、不同 scale stage 分配固定颜色。
   - 示例：AesFA 的蓝/绿双箭头；S2WAT 的 Stage 1/2/3 淡色区分。

3. **主图 + 子图放大的信息层级**
   - 方法：在主架构图旁设置 (a)(b)(c) 子图，对关键模块进行放大说明。
   - 示例：S2WAT 将 Attention Block、Decoder Layer 单独拉出；AesFA 配合 Figure 3 放大 AdaOct。

4. **用虚线框表达重复/堆叠/可选模块**
   - 方法：虚线框适合表示“重复 N 次”或“可选子结构”。
   - 示例：S2WAT 的 Stage 1/2/3 虚线外框；Lancet 的 training 监督虚线。

5. **在关键输入输出位置嵌入真实图像缩略图**
   - 方法：在 Content/Style/Output 节点放置小图，使读者快速建立直觉。
   - 示例：AesFA、ArtBank、Lancet 均在输入输出端使用真实图片。

6. **设置清晰的箭头图例（legend）**
   - 方法：当图中存在多种线型或颜色时，在图内或 caption 中给出图例。
   - 示例：Lancet 在图底部设置“实线/虚线”图例；AesFA 在图内说明蓝/绿箭头含义。

7. **保持横向为主、纵向分层为辅的流水线阅读顺序**
   - 方法：数据流通常从左到右，阶段或路径差异可从上到下分层。
   - 示例：AesFA、S2WAT 主流程横向；Lancet 用纵向三带区分功能域。

8. **Caption 与图中文字互补，避免重复**
   - 方法：图中标注模块名和关键变量，caption 解释整体流程与视觉约定。
   - 示例：四篇论文均在 caption 中说明颜色/箭头含义，图中仅保留必要标签。

---

## 四、结论

四篇 AAAI 论文的主架构图在视觉语言上具有较高的共性：均倾向于**横向流程 + 模块化节点 + 颜色/线型区分 + 真实图像嵌入**。差异主要体现在：
- AesFA 强调**频域双路径**；
- S2WAT 强调**层级 Transformer 与注意力细节**；
- ArtBank 强调**训练/推理双阶段**；
- Lancet 强调**latent transport 的三带分层**。

对于我们自己绘制 AAAI 论文主架构图，建议优先采用：横向主流程 + 颜色区分关键路径 + 虚实箭头区分训练/推理 + 子图放大核心模块 + 清晰的图例说明。

---

*报告生成路径：g:\GitHub\Latent_Style\SchrodingerBridge\docs\630\aaai_arch_diagram_style_survey.md*
