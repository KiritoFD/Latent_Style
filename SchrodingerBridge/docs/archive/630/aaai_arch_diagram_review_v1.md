# AAAI 主架构图（v1）视觉与理论评审报告

**评审对象**：`docs/630/aaai_arch_diagram_v1.drawio`  
**评审依据**：`arch_diagram_theory_basis.md`、`aaai_arch_diagram_style_survey.md`  
**日期**：2026-07-03

---

## 总体印象

该图采用了与 Lancet（Latent Bridge Matching）相近的**三带横向分层**布局，将 Style Control、Main Inference Path、Training Supervision 纵向分离，并用颜色与虚实箭头区分不同路径。整体已经覆盖了 Spectral ODE Bridge 的核心组件（Haar DWT 子带拆分、共享主干、独立 velocity heads、Endpoint AdaIN/WCT、Spectral FM Loss）。但作为 AAAI 主架构图的**第一版**，仍存在关键连接错误、右侧布局拥挤、核心洞察表达不足等问题，需要修订。

---

## 一、理论准确性

**准确之处**：
- 明确画出 Haar DWT 将 latent 拆分为 LL/LH/HL/HH 四个正交子带，并标注 HH 为 "dead head" 被移除，与理论一致。
- 主推理路径完整：Content Image → VAE Encoder → z₀ → DWT → Stack & Project → Backbone → v_LL/v_LH/v_HL → Spectral ODE Integrator → iDWT → Endpoint AdaIN/WCT → z_T → Decoder → Output。
- 训练监督路径包含 x_t 构造、DWT、Target Δ_i 与加权的 Spectral FM Loss，公式基本正确。

**主要问题**：
1. **存在未定义节点**：`edge_2` 的目标为 `node_placeholder_cross_attn`，但 XML 中并未定义该节点。这会导致 draw.io 打开时报错或显示异常，且使得 Style Tokens 如何注入主干 cross-attention 的路径完全断裂。
2. **独立 velocity heads 结构不显式**：图中仅将 v_LL/v_LH/v_HL 作为 backbone 的输出标签，未将 "SpectralVelocityHead" 作为独立模块画出，削弱了 "shared backbone + per-subband heads" 这一核心设计。
3. **"Stack & Project (16 → 64)" 含义不明**：16 与 64 的物理意义未解释，读者无法直接理解为何是 16 通道、为何投影到 64。
4. **缺少核心洞察的视觉化**：图中没有对比或文字说明 "为何要把运输问题搬到小波域"，欧氏 Flow Matching 的局限性没有呈现。

---

## 二、视觉清晰度

**优点**：
- 三带分区清晰，legend 完整说明了 content / style / spectral / training 四种颜色以及虚实箭头含义。
- 数据流总体从左到右，符合阅读惯性。

**问题**：
1. **右侧超出页面边界**：`Stylized Output` 节点位于 x=1190、宽 80，右边缘达到 x=1270，超过 pageWidth=1200，打印或导出时可能被裁切。
2. **局部拥挤**：`(a) One Block` 子图与 `Backbone` 节点横向间距过小，且子图内部文字密集，易造成视觉拥堵。
3. **字号偏小**：损失函数节点与子图内部文字使用 fontSize=10，在论文印刷后阅读困难。
4. **HH dead head 标注不够醒目**："✗" 与 "dead head" 字号较小，且 HH 子带的颜色（灰）与背景接近，容易被忽略。
5. **颜色层次复杂**：色带背景色与模块填充色叠加，导致同一区域内出现多种相近颜色，干扰路径识别。

---

## 三、AAAI 风格一致性

与调研的 4 篇 AAAI 论文图相比：

**协调之处**：
- 三带分层、虚实箭头图例与 Lancet 高度一致。
- 区分 training 与 inference 的思路与 ArtBank、Lancet 一致。

**差距之处**：
- **缺少真实图像缩略图**：AesFA、ArtBank、Lancet 均在输入/输出端嵌入小图，增强直观性。
- **未用虚线框表示重复模块**：S2WAT 使用虚线框包围 Stage 1/2/3，本图 "×4 blocks" 直接写在节点内，重复结构不够醒目。
- **配色饱和度偏高**：学术图通常使用更柔和、低饱和度的配色，当前橙色、绿色、紫色略显鲜艳。

---

## 四、故事性

30 秒内可以大致理解 "输入 → 编码 → 频域拆分 → 速度场 → 积分 → 解码 → 输出" 的主流程。但以下要点未能快速传达：

- **为什么需要频域解耦**：图中没有说明低频保内容、高频传风格这一关键动机。
- **为什么 HH 被去掉**：仅标记 dead head，没有给出原因（对风格指标无贡献、易引入噪声）。
- **为什么 ReLU² attention 更好**：子图中仅列出模块名，没有与 softmax 的对比或稀疏激活的视觉暗示。

读者需要依赖 Figure Caption 才能理解模型 "为什么 work"。

---

## 五、具体修改建议

1. **修复未定义节点并明确 style 注入路径**
   - 将 `node_placeholder_cross_attn` 替换为实际节点（如主干上的 "Cross-Attn" 入口），并用虚线箭头将 Style Tokens 连接到该节点，说明 style 条件注入方式。

2. **显式画出独立 velocity heads**
   - 在 Backbone 与 v_LL/v_LH/v_HL 之间增加三个独立模块（或一个并排的 heads 组），强化 "shared backbone + per-subband heads" 的理论设计。

3. **嵌入真实图像缩略图**
   - 在 `Content Image`、`Style Image`、`Stylized Output` 节点内放置小图，使读者 5 秒内建立任务直觉，符合 AesFA / ArtBank / Lancet 的做法。

4. **增大关键文字字号**
   - 将损失函数、子图内部说明文字从 10pt 提升至 11-12pt；`Spectral ODE Integrator` 的公式与 `Endpoint AdaIN/WCT` 标签也应更清晰。

5. **调整布局避免右侧溢出**
   - 将 `Stylized Output` 左移或整体缩小模块宽度，确保所有节点在 pageWidth=1200 内；同时给 `(a) One Block` 子图留出更多横向留白。

6. **优化配色方案**
   - 降低色带背景饱和度，或改用更柔和的学术配色；确保 content（蓝）、style（橙）、spectral（绿）、training（紫）四色对比清晰但不刺眼。

7. **在 DWT 拆分处补充子带语义说明**
   - 在 LL/LH/HL/HH 节点旁增加简短标注：LL ≈ structure/tone，LH/HL ≈ brushstroke/edge，HH ≈ noise/removed，帮助读者快速理解频域分工。

8. **增加核心洞察的对比示意**
   - 在图左上角或图注附近增加一个微型对比框：左侧 "Euclidean FM: single velocity field"，右侧 "Spectral FM: per-subband velocity"，突出 "频域解耦" 这一核心卖点。

---

## 六、总结

v1 版架构图已经具备了 AAAI 论文主图的基本骨架，三带分层、颜色编码与虚实箭头图例都值得保留。下一步应优先修复 `node_placeholder_cross_attn` 未定义错误、避免右侧溢出、增大关键文字、补充真实图像，并更强烈地视觉化 "频域解耦" 与 "独立 velocity heads" 两大核心贡献。
