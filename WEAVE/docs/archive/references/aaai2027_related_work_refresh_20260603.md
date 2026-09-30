# AAAI 2027 Related-Work Refresh Memo

Date: 2026-06-03  
Scope: tokenizer / representation / arbitrary style transfer / multi-style transfer / no-op-aware evaluation  
Constraint: memo only; no manuscript edits

## 0. 结论先行

基于当前 `paper_aaai2026.tex` 的 intro / related-work 表述，以及补查的近两三年原始论文页，当前 related-work **没有大的硬缺口**，但有三个值得优先修的“新鲜度/边界”问题：

1. **arbitrary / diffusion-style transfer 文献需要再补一个 2026 级别的新锚点**，否则当前 diffusion 线停在 `StyleSSP (CVPR 2025)` 会显得略旧。最值得补的是 `StyleGallery (CVPR 2026)`。
2. **如果想把 diffusion-style transfer 线写得更完整，最好补一个“data-at-scale / supervised diffusion”邻近工作**，最合适的是 `OmniStyle (CVPR 2025)`；否则当前第 68 行附近容易让人读成“近年的 diffusion style transfer 基本都是 training-free”。
3. **tokenizer 术语仍然有误读风险**。当前正文已经加了 disambiguation sentence，这是对的；若仍担心 reviewer 把 `style tokenizer` 听成 diffusion/image tokenizer，可选补 `StyleTokenizer (ECCV 2024)`，但它是**术语邻接**，不是主 baseline。

同时，一个重要的负结论依然成立：**我没有找到 2025-2026 的官方主源工作，足以替换 `SaMST` 作为你们在 compact multi-style / style-representation 方向上的最强直接 comparator。**

## 1. 先看当前 paper surface 的可能缺口

结合当前正文：

- line 38-40 已经把三条 regime 分开了：reference-guided arbitrary、compact multi-style、training-free / diffusion。这个方向是对的。
- line 62 已经明确 `style tokenizer` 不是 image tokenizer / target-image encoder / diffusion-token controller，这个修补也对。
- 但 line 68 当前举例仍集中在 `CAST / StyleID / Z* / SMS / StyleSSP`，**缺少 2026 年最新的 official paper anchor**。
- evaluation 段落已经把 `idt` 写成 paper-specific diagnostic，而不是 community standard；这一点应该继续保留。

所以，当前更像是“**需要补充一两个最新锚点并继续守住 claim boundary**”，不是“需要重写整个 related work”。

## 2. 建议补充或替换的文献

### A. 建议补充：StyleGallery (CVPR 2026)

- **链接**: [StyleGallery: Training-free and Semantic-aware Personalized Style Transfer from Arbitrary Image References](https://openaccess.thecvf.com/content/CVPR2026/papers/He_StyleGallery_Training-free_and_Semantic-aware_Personalized_Style_Transfer_from_Arbitrary_Image_CVPR_2026_paper.pdf)
- **为什么值得补**:
  - 是 2026 年的官方 CVPR paper；
  - 仍然属于 **reference-guided / diffusion-based arbitrary style transfer**；
  - 强调 semantic-aware region matching、multiple style references、personalized customization，这和你们当前在“semantic consistency / reference-guided”这条叙事线上是直接相邻的。
- **怎么定位**:
  - 它不是你们的直接 baseline family；
  - 它更适合作为“2026 年 arbitrary reference-guided diffusion style transfer 仍在快速推进”的 freshness anchor。

### B. 建议补充：OmniStyle (CVPR 2025)

- **链接**: [OmniStyle: Filtering High Quality Style Transfer Data at Scale](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_OmniStyle_Filtering_High_Quality_Style_Transfer_Data_at_Scale_CVPR_2025_paper.html)
- **为什么值得补**:
  - 它不是 training-free，而是 **data-at-scale + diffusion transformer** 的 style transfer 路线；
  - 说明近年的 style transfer diffusion 线不只是 `StyleID / StyleSSP` 这类 training-free adaptation，也包括“数据规模化 + supervised diffusion”。
- **怎么定位**:
  - 不是 compact multi-style comparator；
  - 更适合作为“modern large-prior / diffusion style transfer”补充锚点。
- **何时值得补**:
  - 如果你们只想维持最小 related-work 面，`StyleGallery` 比它更优先；
  - 如果你们想让 diffusion / arbitrary line 更完整，`OmniStyle` 值得加。

### C. 可选补充：StyleTokenizer (ECCV 2024)

- **链接**: [StyleTokenizer: Defining Image Style by a Single Instance for Controlling Diffusion Models](https://eccv.ecva.net/virtual/2024/poster/1020)
- **为什么值得补**:
  - 不是因为它是 direct baseline；
  - 而是因为它让 `style tokenizer` 这个词在 reviewer 心里更容易联想到 **diffusion-style control / image-conditioned tokenization**。
- **怎么定位**:
  - 只能作为**术语邻接**工作来提；
  - 不应该放进 main compact multi-style baseline chain 里。
- **建议**:
  - 如果正文已经保留 line 62 那句 disambiguation，其实可以不引；
  - 只有当你们预计 reviewer 会抓住 “tokenizer” 这个词不放时，再考虑补它。

### D. multi-style / representation 主 baseline 不建议替换

- **保留**: [SaMST: Pluggable Style Representation Learning for Multi-Style Transfer (ACCV 2024)](https://openaccess.thecvf.com/content/ACCV2024/html/Liu_Pluggable_Style_Representation_Learning_for_Multi-Style_Transfer_ACCV_2024_paper.html)
- **结论**:
  - 当前仍未发现更强的 2025-2026 官方主源工作，可替代 `SaMST` 作为你们在 **compact multi-style / style representation** 方向上的直接 comparator；
  - `SaMAM / Mamba-ST` 依然更适合放在 **executor / backbone efficiency** 邻近线，而不是 representation baseline 线上。

## 3. no-op-aware evaluation 方向的刷新结论

### 3.1 没有找到 2024-2026 的直接前例，能覆盖你们的 `idt` 诊断

我这轮补查后，仍然**没有**找到近两三年的官方主源工作，能直接覆盖下面这组组合：

- art-to-art transfer；
- unchanged source 本身就有很高 raw style score；
- 需要显式减去 unchanged-image prior。

因此：

- 你们的 `idt` / no-op-aware evaluation 仍应被写成 **paper-specific diagnostic / reporting contribution**；
- 不要把它写成 community standard，也不要暗示“已有 related work 已经把这个 protocol 讲清楚了”。

### 3.2 可选评价侧补充仍然只是“外围支持”

- [HPSv3 (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Ma_HPSv3_Towards_Wide-Spectrum_Human_Preference_Score_ICCV_2025_paper.html)
- [Image Quality Assessment: From Human to Machine Preference (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Image_Quality_Assessment_From_Human_to_Machine_Preference_CVPR_2025_paper.html)

它们可以支持“图像评价正在向 preference-aligned / broader evaluation 演化”这类大背景句子，但**不能**直接支持你们的 no-op / `idt` 诊断。

### 3.3 身份测试的最佳相邻前例仍然是旧一点但有用的工作

- [Scaling Painting Style Transfer](https://arxiv.org/abs/2212.13459)

它的价值仍然是：给“identity-style sanity check”提供一个相邻 precedent。  
但它也**不能**把你们的 `idt` 方案变成 community-standard evaluation。

## 4. 我们应该如何定位自己

当前最稳的定位仍然是：

1. **不是** reference-guided arbitrary style transfer；
2. **不是** diffusion large-prior style transfer；
3. **而是** compact, style-id-conditioned, domain-level artistic transfer；
4. 核心问题不是“如何从 style exemplar 对齐 correspondence”，而是“style-side control signal 在 renderer/executor 之后能否真正存活并形成 no-op-adjusted style gain”；
5. evaluation 贡献不是“发明通用新指标”，而是把 `idt` 作为一个 **scope-bounded reporting protocol** 引入 separated art-to-art transfer。

## 5. 哪些 claim 建议继续降调

1. **不要把 `idt` 写成 community norm**  
   最多写成：受 prior metric-calibration literature 启发，你们在这个 split 上引入了 paper-specific diagnostic。

2. **不要把 tokenizer 线写成通用理论**  
   目前最多能写到：在当前 Distinct5 tokenizer family / current successor evidence 下，executed representation 比 raw code geometry 更接近真正的问题。

3. **不要把 diffusion line 写窄成“training-free methods = 近年全部 diffusion style transfer”**  
   如果补 `OmniStyle` 或 `StyleGallery`，这一点尤其要注意。

4. **不要把 state-space line写成 representation comparator**  
   `SaMAM / Mamba-ST` 更像 executor/backbone 近邻，而不是 direct tokenizer / representation baseline。

## 6. 最小可执行建议

如果只做最小 related-work refresh，我建议：

1. **优先补 `StyleGallery (CVPR 2026)`**；
2. 如果还想再补一个，把 **`OmniStyle (CVPR 2025)`** 加进 diffusion / large-prior line；
3. `SaMST` 保持为 main compact multi-style representation comparator；
4. `StyleTokenizer` 只在你们决定继续高频使用 `style tokenizer` 术语时再补；
5. `idt` 继续按 **paper-specific diagnostic** 来写，不要升级成 community-standard claim。
