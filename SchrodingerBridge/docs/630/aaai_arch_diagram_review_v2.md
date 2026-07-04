# AAAI 主架构图（v2）视觉与理论评审报告

**评审对象**：`docs/630/aaai_arch_diagram_v2.drawio`  
**评审依据**：`docs/630/aaai_arch_diagram_review_v1.md`、`docs/630/arch_diagram_theory_basis.md`  
**日期**：2026-07-03

---

## 总体印象

v2 在 v1 基础上完成了关键的结构性修正：未定义节点已修复、独立 velocity heads 被显式画出、DWT 子带语义已补充、核心洞察以文字框形式加入，且右侧溢出已解决。整体骨架已接近 AAAI 主图要求，但在字号一致性、真实图像缩略图、配色柔和度等方面仍有提升空间，建议再做一轮精细化打磨后再定稿。

---

## 一、v1 八条修改建议落实情况

| 编号 | v1 建议 | v2 状态 | 说明 |
|---|---|---|---|
| 1 | 修复未定义节点 | ✅ 已落实 | 原 `node_placeholder_cross_attn` 已移除；Style Tokens 通过虚线箭头直接连入 Backbone。 |
| 2 | 显式画出独立 velocity heads | ✅ 已落实 | Backbone 下方新增 `v_LL` / `v_LH` / `v_HL` 三个独立模块，并标注 "Per-subband Heads"。 |
| 3 | 嵌入真实图像缩略图 | ⚠️ 未落实 | 本次无现成图片，`Content Image` / `Style Image` / `Stylized Output` 仍为纯色占位框。 |
| 4 | 增大关键文字字号 | ⚠️ 部分落实 | `z₀`、`v_i`、`iDWT`、`DWT` 字号已加大，但子带语义说明、One Block 内部、损失函数仍偏小（9–10 pt）。 |
| 5 | 避免右侧溢出 | ✅ 已落实 | `Stylized Output` 右边缘约 x=1135，整体在 `pageWidth=1200` 范围内。 |
| 6 | 配色更柔和学术 | ⚠️ 部分落实 | 色带背景已变浅，但模块填充色（橙/绿/紫/蓝）饱和度仍偏高。 |
| 7 | DWT 处补充子带语义 | ✅ 已落实 | LL=structure & tone，LH=vertical brush，HL=horizontal brush，HH=removed: noise。 |
| 8 | 增加核心洞察对比 | ✅ 已落实 | 左上角新增 `Core Insight` 文字框，对比 Euclidean FM 与 Spectral FM。 |

---

## 二、多维度评审

### 1. 理论准确性

主推理路径与训练监督路径均与理论基础一致：Content → VAE Encoder → `z₀` → Haar DWT → Stack & Project → Shared Backbone → 独立 velocity heads → Spectral ODE Integrator → iDWT → Endpoint AdaIN/WCT → `z_T` → Decoder → Output。训练路径中的 `x_t` 构造、DWT、Target `Δ_i` 与加权 Spectral FM Loss 公式正确。

值得肯定的是，One Block 内部明确写出 "DWT-Route Cross-Attn"，准确反映了 LL bypass 的设计；Spectral ODE Integrator 的更新公式 `h_i ← h_i + v_i·dt` 也足够简洁。但仍有两处可改进：

- **"Stack & Project" 维度不明**：v1 已指出的 "16 → 64" 含义仍未解释，读者无法直接理解其物理意义。
- **HH 丢弃的因果**：图中标注了 "removed: noise"，但未说明这是基于 628 L8 消融实验发现 HH velocity head DEAD（Δclip ≈ ±0.0001）的结果。

### 2. 视觉清晰度

三带分层与图例完整，数据流从左到右，符合阅读惯性。但仍存在以下问题：

- **字号不均衡**：核心洞察框仅 11 pt，损失函数节点 11 pt 且文字密集，子带语义仅 9 pt，印刷后阅读吃力。
- **One Block 子图拥挤**：160×80 的框内挤了 5 行技术术语，建议拆分为 2–3 组或增大框体。
- **HH 子带标注仍不醒目**：HH 灰色填充与背景接近，"✗" 和 "removed: noise" 容易被忽略。
- **训练路径与推理路径的空间交织**：虚实箭头已区分，但两条路径在垂直方向上交错，初次阅读者可能混淆。

### 3. AAAI 风格一致性

与 AesFA、ArtBank、Lancet 等 AAAI 风格迁移论文主图相比：

- ✅ 三带分层、虚实箭头图例与 Lancet 一致，区分 training/inference 的思路合理。
- ⚠️ 仍缺少真实图像缩略图，直观性不足。
- ⚠️ 配色饱和度仍高于典型 AAAI 论文图，建议转向更柔和、低饱和的学术色系。
- ⚠️ "×4 blocks" 直接写在 Backbone 节点内，未用虚线框表示重复模块（S2WAT 的常用做法）。

### 4. 故事性 / 30 秒理解度

核心洞察框显著提升了故事性，读者能快速抓住 "Euclidean FM 统一速度场 vs. Spectral FM 子带独立速度场" 的卖点。但仍有可优化之处：

- "vertical brush / horizontal brush" 表述偏口语，可改为 "vertical edges / texture" 和 "horizontal edges / texture"，更贴近小波理论语义。
- 缺少 "低频小幅度移动、高频大幅度移动" 的视觉化暗示，这是频域解耦的核心直觉。
- 训练监督带的 `x_t = (1 − t)·z₀ + t·z_target` 与主推理路径的关系可更明确：建议用箭头或颜色暗示 `x_t` 即 ODE 积分中的中间状态。

### 5. 是否可直接用于 AAAI 论文主图？

**尚不能直接定稿。** 虽然理论准确性和结构完整性已达标，但字号、图像缩略图、配色、局部标注等视觉细节仍未达到 AAAI 主图应有的印刷质量。建议按下面 5 条修改后再提交定稿。

---

## 三、具体修改建议

1. **嵌入真实图像缩略图**  
   在 `Content Image`、`Style Image`、`Stylized Output` 三个节点内嵌入 64×64 或 70×70 的示例小图，使读者 5 秒内建立任务直觉。这是 AAAI 风格迁移论文主图的常规做法。

2. **统一并增大关键文字字号**  
   将核心洞察框、损失函数节点、子带语义说明统一提升至 12 pt；One Block 内部若空间不足，可将 5 行说明拆为 "AdaLN + Self-Attn / DWT-Route Cross-Attn / ReLU² Gate + FFN" 三组，避免拥挤。

3. **降低配色饱和度**  
   将 style 橙、spectral 绿、training 紫、content 蓝调整为更低饱和度的学术色系，并确保四种颜色在灰度打印下仍可区分。

4. **优化 HH 子带与重复结构表达**  
   HH 节点用更醒目的方式标注丢弃（如加粗删除线、更深的灰色或独立注释框）；Backbone 的 "×4 blocks" 建议用虚线外框包围并标注重复次数。

5. **补充 Stack & Project 的维度说明**  
   在 "Stack & Project" 节点旁增加小字说明，例如 "4 sub-bands × C → hidden dim"，消除 "16 → 64" 的歧义。

---

## 四、总结

v2 已实质性解决 v1 中的结构性问题（未定义节点、独立 velocity heads、右侧溢出、DWT 语义、核心洞察），理论准确性和故事性均有明显提升。当前主要短板集中在视觉精细度：字号、图像缩略图、配色和局部标注。建议按上述 5 条再做一轮微调后即可定稿。
