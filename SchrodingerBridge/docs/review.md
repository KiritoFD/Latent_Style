下面按 **AAAI-27 投稿要求 → 模拟审稿意见 → 强弱点 → writing 改法 → 图表改法 → 实验补充 → 最终组织方案** 来给。我的总体判断是：**当前版本已经从“borderline reject”进化到“borderline / weak accept 边缘”，但还没有稳。现在最大问题不是性能，而是最强 successor rows 的证据闭环与方法定义仍不够硬。**

AAAI-27 官方页面目前列出了会议时间地点和作者时间线：AAAI-27 将于 2027 年 2 月 16–23 日在 Montréal 举办，摘要截止为 2026 年 7 月 21 日，全文截止为 7 月 28 日，补充材料和代码截止为 7 月 31 日。([AAAI][1]) 具体 main-track 审稿标准如果沿用近年 AAAI 主技术轨道，核心会看 **contribution significance/novelty、theoretical/empirical soundness、AAAI relevance、clarity、responsible/reproducible research practices**；AAAI-26 的官方 CFP 明确写到这些评价维度，并强调 critical material 应放在主文中，不能依赖 reviewer 一定看 supplement。([AAAI][2])

---

# 1. 总体评分预估

**当前版本：5.8–6.1 / 10。**

如果按 AAAI 常见口径，我会写成：

> **Recommendation: borderline / weak accept if artifacts and variant details are clarified; otherwise weak reject.**
> **Confidence: medium.**

更细一点：

| 维度                  |        当前估分 | 主要原因                                                                                             |
| ------------------- | ----------: | ------------------------------------------------------------------------------------------------ |
| Significance        |     **6.5** | IDT/no-op failure 是真实评价问题，style-ID 低成本 frontier 叙事有价值                                            |
| Novelty             | **5.5–6.0** | 方法是 latent renderer + endpoint pairing + SA-SWD + kinetic，组合有新意，但单个模块新颖性不强                       |
| Empirical soundness | **5.0–5.5** | Table 1 frontier 强，但 strongest successors 只有 CLIP-S / LPIPS closure，artifact/human/non-CLIP 还没闭环 |
| Clarity             |     **6.0** | 比早期版本清楚很多，Figure 2 也修好了；但 LBM-Knee / PS / PS-v2 定义仍偏模糊                                           |
| Reproducibility     | **4.5–5.0** | checklist 里 preprocessing code、source code、seed、hardware、hyperparameter 等仍是 No/Partial           |
| Overall             | **5.8–6.1** | 有 weak accept 潜力，但目前仍容易被 soundness/reproducibility 拉回 5 分                                        |

如果补齐我下面列的 **successor artifact + qualitative + non-CLIP / VLM + variant definition**，我会把分数上调到 **6.4–6.8**。如果再补多 seed 或第二个 split，能到 **6.8–7.0**。

---

# 2. 模拟审稿意见

## Summary

本文研究 domain-level artistic transfer，重点不是给定 reference style image 的 arbitrary style transfer，而是 **source image + target style ID** 的低成本 style-ID customization。论文指出一个评价陷阱：当 source 本身已经是 artwork 时，unchanged image 也可能有较高 target-style affinity，因此 raw CLIP-S 会高估 stylization 成功。作者提出 IDT floor，并报告 transfer-only (\Delta_{\mathrm{IDT}})。方法上，LBM 是一个 3.9M 参数的 VAE latent transport family，使用 training-side endpoint pairing、style-conditioned latent field、terminal SA-SWD 和 kinetic motion control。当前版本 Table 1 显示 LBM-K 是 closed conservative anchor，而 LBM-Knee、LBM-PS、LBM-PS-v2 分别达到 0.7102、0.7274、0.7307 transfer CLIP-S，并构成 style-content frontier。

## Strengths

**S1. 评价问题有说服力。**
IDT/no-op floor 是这篇稿子的最强贡献。当前版本明确把 Distinct5-512 定义为 IDT-calibrated operating-point benchmark，而不是 single-score leaderboard。论文指出 IDT transfer CLIP-S 已经很高，SaMAM-2250 低于 IDT，而 SaMST 虽然高于 IDT 但处于高 damage 区间，这个诊断很适合说服 reviewer：raw CLIP-style 不足以证明 target-style movement。

**S2. 性能叙事比前几版强很多。**
现在不是只靠 LBM-K 的 +0.0312 (\Delta_{\mathrm{IDT,tr}})，而是把 LBM 讲成一个可选 operating-point frontier。Table 1 里 LBM-Knee 是 content-preserving successor，LBM-PS/PS-v2 是 high-style/style-ceiling successor，这比早期“小幅超过 IDT”的故事更能打。

**S3. 方法接口边界清楚。**
Figure 2 现在把 Style Control、Inference Path、Training-side Endpoint Supervision 分开，且明确 target-style examples used only during training。这能有效避免 reviewer 误解成 reference-guided method。

**S4. 论文开始控制 claim scope。**
作者没有把 Seedream-4.5 当成 same-interface baseline，而是 external large-prior reference；也没有把 successor rows 的 artifact/human closure 伪装成已经完成。这种诚实会加分。

**S5. 低成本论点有现实价值。**
LBM-F/K 在 1.2 min RTX 3060 上过 IDT floor，而 SaMAM-2250 和 SaMST e15 的训练时间分别是 7.6h 和 5.8h；historical strict-750 还保留了 3.91M / 114 ms/image 的部署证据。这对 AAAI 的 broader AI audience 比单纯 style transfer SOTA 更容易讲清楚。

## Weaknesses

**W1. 最强结果没有完整 empirical closure。**
当前最强的 LBM-Knee / LBM-PS / LBM-PS-v2 在 Table 1 的 Closure 列都是 “frontier”，不是 closed。论文 limitations 也承认 promoted successor rows 目前只有 transfer-style / LPIPS closure，完整 artifact-sensitive 和 human-preference closure 仍是 future work。 这是最大扣分点。Reviewer 会问：这些高 CLIP-S 点是否带来 texture artifacts？是否只是 CLIP-friendly stylization？是否和 human preference 一致？

**W2. LBM-Knee / PS / PS-v2 的方法定义仍不够可复现。**
当前 Method 只说 LBM-Knee 结合 stronger successor family、queue、anisotropic/Stokes-style regularization；LBM-PS 是 balanced high-style successor；LBM-PS-v2 是 weaker viscosity。 这对主结果来说太模糊。AAAI reviewer 不会满足于“successor variant”这种描述，必须知道到底改了哪个 loss、哪个 selector、哪个 (\lambda)、哪个 viscosity schedule。

**W3. 没有主文 qualitative comparison。**
这是视觉任务的硬短板。Figure 1 是 frontier + ArtFID bar，Figure 2 是 architecture，但主文没有 Source / IDT / SaMST / Seedream / LBM-Knee / LBM-PS-v2 的视觉对比。对 style transfer 论文来说，没有 qualitative strip 会显著降低 reviewer 对自动指标的信任。

**W4. Seedream-4.5 comparison 是双刃剑。**
Seedream 加入后提高了外部参考价值，但它是 API large-prior reference，不是 same-interface style-ID model。当前文字已经说明使用 source image + fixed target-domain prompt “convert the image to <style> style”，这比上一版清楚很多。 但仍需补充：每个 style 的 prompt 是否完全固定、是否 single sample、API 日期、参数、是否有 negative prompt、是否做 resizing/post-processing。否则 reviewer 会认为外部参考不可复现。

**W5. 复现性 checklist 仍然会扣分。**
当前 checklist 中 hyperparameter range、preprocessing code、source code、random seeds、hardware/software、final hyperparameters 仍有 No/Partial。 AAAI 官方 review criteria 明确把 reproducibility practices 作为 additional considerations，包括 documenting experiments、sharing data/code 等。([AAAI][2]) 你们现在有 168 development points 和多个 successor rows，如果 config/source/eval packet 不透明，会被质疑 researcher degrees of freedom。

**W6. “Bridge Matching” 标题仍有一点风险。**
方法正文已经说 active headline rows 是 endpoint-mode OMF，(w_{\mathrm{flow}}=0)，不是 online Sinkhorn 或 random-time flow matching。 但标题还叫 “Latent Bridge Matching”，容易让理论/生成模型 reviewer 期待更强的 bridge/flow 形式化。可以保留 LBM 作为方法名，但正文和标题要更偏 “style-ID latent transport”，减少 full bridge solver 暗示。

---

# 3. 主要强项在哪里

## 强项 A：IDT calibration 是可发表的评价贡献

这篇稿子的 “hook” 不是 SA-SWD 或 kinetic，而是：**art-to-art transfer 中 no-op baseline 必须显式报告**。这点很清楚，也有实证支撑：SaMAM-2250 低于 IDT，SaMST 虽高但高 damage，LBM-K 与 successors 给出不同 style-content 区域。

建议把论文定位成：

> **Evaluation-aware method paper**
> 不是单纯 “new architecture paper”。

也就是：
**第一贡献是 IDT-calibrated protocol；第二贡献才是 LBM frontier。**

## 强项 B：frontier 叙事已经成立

Table 1 很关键：

* IDT: 0.6399 / 1.0000 / 0
* SaMAM-2250: 0.5523 / 0.6395 / -0.0877
* SaMST e15: 0.6957 / 0.3681 / +0.0558
* Seedream-4.5: 0.6920 / 0.5077 / +0.0521
* LBM-K: 0.6712 / 0.6277 / +0.0312
* LBM-Knee: 0.7102 / 0.5397 / +0.0703
* LBM-PS: 0.7274 / 0.3967 / +0.0875
* LBM-PS-v2: 0.7307 / 0.3817 / +0.0908

这说明 **LBM-Knee 同时比 Seedream transfer style 更高、1-LPIPS 也更高；LBM-PS/PS-v2 则冲到更高 style ceiling**。这个结果足够成为主文亮点。

## 强项 C：任务接口边界清楚

现在 Figure 2 的三层结构非常适合主文：style ID → tokenizer → latent field → output；target latents 只在 training-side endpoint supervision。这个图可以直接作为 Method 的 “contract figure”。

---

# 4. 主要弱点在哪里

## 弱点 A：successor rows 还像“开发点”，不是“论文点”

虽然 Table 1 里列了 Closure，但 “frontier” 这个词本身不能替代 closure。Reviewer 会觉得：

> 既然 strongest claim 来自 LBM-Knee / PS / PS-v2，那么为什么 artifact、human、seed、timing 都没有完全闭合？

你现在的策略是诚实承认 future work，这能避免 overclaim，但会压分。想冲 weak accept，至少 LBM-Knee 必须从 “frontier” 变成 “closed”。

## 弱点 B：method 中变体定义不够硬

目前 Operating-point variants 仍像论文内部说明，不像 reproducible method spec。尤其是：

* “stronger successor family”
* “anisotropic/Stokes-style regularization”
* “balanced high-style successor”
* “weaker viscosity”

这些词 reviewer 无法复现。Table 1 的主角必须有公式或参数表。

## 弱点 C：主文缺 qualitative

自动指标 + style transfer = 必须有图。尤其你们自己强调 CLIP-S 会失灵，那么只给 CLIP-S/LPIPS frontier 但没有 visual strip，会被攻击。

## 弱点 D：复现性 checklist 明显拖后腿

AAAI 竞争很强，当前 checklist 的 No/Partial 会被 reviewer 当作 soundness 风险，而不只是形式问题。

---

# 5. Writing 怎么改

## 5.1 标题建议

当前标题：

> Latent Bridge Matching: IDT-Calibrated Latent Transport for Domain-Level Artistic Transfer

可以接受，但更稳的是：

> **IDT-Calibrated Latent Transport for Style-ID Artistic Transfer**

如果必须保留 LBM：

> **Latent Bridge Matching for Style-ID Artistic Transfer: IDT-Calibrated Evaluation and Compact Latent Transport**

这样可以降低 “full bridge solver” 预期。

## 5.2 Abstract 建议改写

当前 abstract 信息够多，但太长，且 “blunt evaluation failure” 有点口语。建议改成更 AAAI 风格：

```text
Domain-level artistic transfer is difficult to evaluate when the source image is already artwork: the unchanged source can obtain high target-style affinity without performing the requested transfer. We make this failure mode explicit through an identical-image transfer (IDT) floor on a CLIP-separated WikiArt Distinct5 stress split and report signed transfer-only target-style movement above that floor.

We propose Latent Bridge Matching (LBM), a compact style-ID latent transport family in VAE space. LBM separates training-side endpoint construction, style-conditioned latent execution, kinetic motion control, and terminal SA-SWD matching. At inference time, it receives only a content image and a target style identifier.

On Distinct5-512, LBM forms a controllable style-content frontier rather than a single checkpoint. A closed conservative anchor clears the IDT floor within 1.2 minutes on an RTX 3060, while promoted LBM-Knee and Pattern+Stokes successors move transfer CLIP-S into stronger regimes under the same style-ID contract. Compared with reproduced compact baselines and an external large-prior reference, LBM exposes selectable low-damage, Pareto-knee, and style-ceiling operating points. Historical strict-750 results further show a 3.9M-parameter / 114 ms-image compact operating point. These results support IDT calibration as a useful diagnostic and compact latent transport as a practical route to low-cost style-ID customization.
```

重点是：**把 closed anchor 和 promoted successors 区分开**。不要让 abstract 暗示所有 successors 都已经 artifact/human closed。

## 5.3 Introduction 改法

现在 Intro 已经比旧版清楚，但最后一段可以再压缩。建议 introduction 只保留三件事：

1. IDT/no-op failure；
2. style-ID low-cost task boundary；
3. LBM frontier result。

Contribution 中第三条现在是：

> Frontier evidence.We show ...

这里有一个排版细节：**“Frontier evidence.We” 少了空格**。这种小错误在首页附近很伤印象。改成：

> **Frontier evidence.** We show ...

## 5.4 Method 写法要更硬

Method 的核心问题不是太长，而是 **variant spec 不够具体**。建议把 Operating-point variants 改成一个 5 行小表：

| Variant   | Endpoint selector                | Terminal/statistics          | Motion control                  | Role          |
| --------- | -------------------------------- | ---------------------------- | ------------------------------- | ------------- |
| LBM-K     | prototype queue top-k            | SA-SWD                       | (\lambda_{\mathrm{kin}}=\dots)  | conservative  |
| LBM-Knee  | structure-aware queue / StructOT | SA-SWD + Stokes term         | (\lambda_{\mathrm{visc}}=\dots) | Pareto knee   |
| LBM-PS    | Pattern+Stokes statistics        | pattern/stroke terminal term | (\lambda_{\mathrm{visc}}=\dots) | high-style    |
| LBM-PS-v2 | same as PS                       | same                         | weaker viscosity (\dots)        | style ceiling |

如果版面不够，至少写成：

```text
LBM-K uses the prototype-aware queue with λterm=..., λkin=..., and no Stokes term. 
LBM-Knee adds structure-aware endpoint selection and an anisotropic Stokes regularizer Rstokes=... with λstokes=..., λvisc=...
LBM-PS adds pattern/stroke terminal statistics P(·)=... to Lterm.
LBM-PS-v2 keeps the same endpoint selector and terminal statistics as LBM-PS but reduces λvisc from ... to ...
```

这几行会极大提升 soundness。

## 5.5 语气上要删掉的句子

建议删除或改写：

| 当前表达                                        | 问题                       | 建议                                                                                                                                    |
| ------------------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| “blunt evaluation failure”                  | 口语化                      | “systematic no-op evaluation failure”                                                                                                 |
| “This is not a marginal +0.02 style story.” | 太像 rebuttal / 宣传         | “The successor rows expand the measured style side of the frontier beyond the conservative anchor.”                                   |
| “stronger successor family”                 | 不可复现                     | 明确是哪一个 selector/loss/regularizer                                                                                                      |
| “not a trivial CLIP-only artifact”          | 证据还不够                    | “an external reference check gives consistent transfer/content trends; additional human and non-CLIP validation remains future work.” |
| “future papers should...”                   | discussion 口吻偏 editorial | “These results suggest reporting selected operating points together with IDT gain and artifact checks.”                               |

---

# 6. 图怎么画

## 6.1 Figure 1：保留，但重新分层表达 claim

当前 Figure 1 的两个 panel 是对的：左边 IDT frontier，右边 closed transfer-only tw-ArtFID。首页 caption 也已经说明 Seedream 是 external large-prior reference，而不是 style-ID model。

但我建议做 4 个小改动：

**第一，Figure 1A 用不同 marker 表示 claim status。**

* 圆点：closed baseline / closed anchor
* 星号：promoted frontier rows
* 三角：external API reference
* 灰点：development landscape

Legend 里写：

> closed rows / frontier rows / external reference / development points

这样 reviewer 一眼能看到：哪些是完全闭合，哪些是 frontier evidence。

**第二，Figure 1A 不要标太多文字。**
当前 LBM-PS、LBM-PS-v2、Seedream、LBM-Knee 都在很小区域，可能重叠。主图建议只标：

* SaMST e15
* Seedream-4.5
* LBM-K
* LBM-Knee
* LBM-PS-v2

LBM-PS 只在 Table 1 里保留。

**第三，Figure 1B 标题更精确。**
当前标题是 “Closed anchor artifact check”。建议改成：

> **Closed transfer-only artifact check**

或：

> **Artifact check for closed rows**

因为里面包括 IDT、SaMAM、LBM-K、SaMST、Seedream，不只是 anchor。

**第四，Figure 1 caption 加一句限制。**

```text
Successor rows are shown in panel (a) as frontier operating points; their full artifact-sensitive and human-preference closure is reported separately when available and is not claimed from panel (b).
```

如果你不想在首页暴露 weakness，那就不要加这句；但从审稿安全角度，Table 1 的 Closure 列已经暴露了，不如主动框定。

## 6.2 Figure 2：现在可以用，但要压低高度

Figure 2 当前内容很清楚，但在 page 3 占了很大空间。建议：

* 高度压到当前的 75–80%；
* 去掉小图标中的过多细节；
* 字体放大一些，尤其 “semantic routing / cross-attention” 和 “SA-SWD terminal matching”；
* caption 改成更有信息量：

```text
Figure 2: LBM style-ID inference and training-only supervision. At test time, only the content image and target style ID are used. Target-style examples enter only through training-side endpoint selection and terminal SA-SWD; they are not inference-time references.
```

当前 caption 太短，没有把 style-ID interface 讲出来。

## 6.3 必须加一个 Figure 3：qualitative strip

这是当前最应该补的图。建议主文放一个半栏或双栏小图，2 个 transfer case 就够：

每行：

> Source | IDT | SaMAM | SaMST | Seedream | LBM-K | LBM-Knee | LBM-PS-v2 | target-domain refs

注意 target-domain refs caption 必须写：

> target-domain images are shown only for visualization and are not provided to LBM at inference.

挑案例原则：

1. 一行展示 LBM-Knee 比 Seedream/SaMST 更稳；
2. 一行展示 LBM-PS-v2 style 更强但仍可接受；
3. appendix 放失败例子。

如果版面太紧，可以把 Figure 1B 的 bar chart 挪到 appendix，首页改成 **frontier + qualitative**。视觉任务里 qualitative 的边际收益比一个已知 ArtFID bar 更大。

## 6.4 Appendix 图表组织

appendix 必须有：

* full 168-point development landscape；
* all selected config；
* all prompt protocols；
* artifact metrics；
* human/VLM prompts；
* qualitative grid；
* failure cases。

AAAI 官方补充材料说明允许 technical appendix、multimedia、code/data，但也明确 reviewer 主要基于主文，关键材料必须在主文自洽。([AAAI][3]) 所以 appendix 只能承载细节，不能承载核心 claim。

---

# 7. 需要补什么实验

## A. 投稿前优先级最高：必须补

### A1. Successor artifact closure

至少给 LBM-Knee、LBM-PS、LBM-PS-v2 补：

| Metric                  | 作用                              |
| ----------------------- | ------------------------------- |
| tw-ArtFID transfer-only | 和 Figure 1B 对齐                  |
| MUSIQ / MANIQA          | no-reference perceptual quality |
| DISTS-content           | 内容结构保持                          |
| HF-Patch-KID            | 高频 artifact                     |
| FFT slope error         | grain / texture artifact        |
| Gram micro              | style texture sanity            |

主文不一定全放，但至少 Table 1 加一列 `tw-ArtFID` 或 `Artifact closed?`。如果 LBM-PS-v2 artifact 很差，就不要主打 PS-v2，改主打 LBM-Knee；PS-v2 作为 style ceiling。

### A2. Qualitative comparison

主文放 2 个 case，appendix 放 25 个 case：

* 5 target styles × 5 selected sources；
* Source / IDT / SaMST / Seedream / LBM-K / LBM-Knee / LBM-PS-v2；
* 同一 resize、同一 crop、同一 target style order。

### A3. Non-CLIP style verification

因为 Distinct5 是 CLIP-separated，主指标又是 CLIP-S，circularity 风险很大。最小方案：

* 用 DINOv2 或 ConvNeXt frozen features；
* train linear style classifier on WikiArt train set；
* report:

  * target style accuracy；
  * target-source margin；
  * source leakage score；
  * confusion matrix；
* 对 IDT、SaMST、Seedream、LBM-K、LBM-Knee、LBM-PS-v2 都报。

这个实验会直接回答 reviewer 的 “CLIP hack” 质疑。

### A4. Human 或 VLM preference

最小人评：

* 100 transfer cases；
* pairwise blind comparison；
* 三个问题：

  1. target style match；
  2. content preservation；
  3. artifact / visual quality；
* 比较：

  * LBM-Knee vs SaMST；
  * LBM-Knee vs Seedream；
  * LBM-PS-v2 vs SaMST；
  * LBM-K vs IDT。

如果时间不够，做 rubric-based VLM preference，但一定要 blind，prompt 固定，且不要给方法名。

### A5. Variant ablation

必须解释 LBM-K → LBM-Knee → LBM-PS → LBM-PS-v2 的增益来自哪里。最小表：

| Variant | Queue | Struct selector | Pattern stats | Stokes/viscosity | CLIP-S | 1-LPIPS | tw-ArtFID |
| ------- | ----- | --------------- | ------------- | ---------------- | -----: | ------: | --------: |

现在 Table 1 是 operating-point comparison，不是 ablation。必须有一个小表证明 PS/Knee 不是内部调参黑盒。

## B. 强烈建议补

### B1. 3 seeds for LBM-Knee

LBM-F/K 已有 bootstrap，但 strongest result 是 Knee/PS。当前 bootstrap 只对 LBM-F/K 和 SaMST 讲稳定性。 最少对 LBM-Knee 做 3 seeds，报告 mean±std。PS-v2 可以单 seed，因为它是 style ceiling，但 Knee 如果作为主结果必须稳。

### B2. 第二个 split

Distinct5 是五类 stress split。建议加：

* Distinct10；
* 或 non-CLIP-selected random WikiArt5；
* 或 photo→art split。

只要能证明 IDT diagnostic 和 LBM-Knee 不是 Distinct5-only，就会显著提高接受率。

### B3. Seedream protocol appendix

必须详细记录：

* API name/version/date；
* prompt template；
* target styles and prompts；
* image input size；
* number of samples per input；
* whether cherry-picked；
* random seed if available；
* post-processing；
* failure handling。

### B4. Reproducibility packet

至少匿名提交：

* Distinct5 split list；
* eval script；
* metrics json/csv；
* selected configs；
* checkpoint names/hash；
* prompt protocol；
* seed setting；
* environment yaml；
* model param count script。

Checklist 里 No/Partial 尽量改成 Yes/Partial+说明。

---

# 8. 实验结果怎么组织

我建议主文实验按这个顺序：

## 8.1 Protocol and metrics

保留现在结构，但压缩 20%。重点放：

* Distinct5-512；
* transfer-only (\Delta_{\mathrm{IDT}})；
* 1-LPIPS；
* tw-ArtFID；
* Seedream 是 external reference；
* strict-750 是 secondary support。

## 8.2 Main frontier table

Table 1 保留，但改成：

| Point     | Interface      | Claim status    | CLIP-S_tr | 1-LPIPS | ΔIDT_tr | tw-ArtFID_tr | Used for claim      |
| --------- | -------------- | --------------- | --------: | ------: | ------: | -----------: | ------------------- |
| IDT       | no-op          | closed          |       ... |     ... |     ... |          ... | floor               |
| SaMAM     | style-ID/repro | closed          |       ... |     ... |     ... |          ... | compact baseline    |
| SaMST     | reproduced     | closed          |       ... |     ... |     ... |          ... | high-style baseline |
| Seedream  | API prompt     | closed          |       ... |     ... |     ... |          ... | external reference  |
| LBM-K     | style-ID       | closed          |       ... |     ... |     ... |          ... | conservative anchor |
| LBM-Knee  | style-ID       | closed/frontier |       ... |     ... |     ... |          ... | main promoted point |
| LBM-PS-v2 | style-ID       | frontier        |       ... |     ... |     ... |          ... | style ceiling       |

比现在的 Closure 列更清楚。

## 8.3 Qualitative figure

紧接 Table 1 放 Figure 3。

## 8.4 Artifact and non-CLIP sanity

一段文字 + 小表：

* tw-ArtFID；
* DINO classifier；
* VLM/human if有。

## 8.5 Cost

Table 2 放后面。注意不要让 cost claim 覆盖 successor claim。当前文本已经说 compact-anchor timing 是最干净的 accessibility evidence，这个写法是对的。

## 8.6 Secondary evidence

strict-750 和 ablations 都压到 1 段。详细表进 supplement。

---

# 9. 7 页主文压缩方案

如果 AAAI-27 采用类似 AAAI-26 的 7 页 technical content + references 规则，你们现在技术内容大约到第 6 页，排版上可控；AAAI-26 官方主技术轨道明确允许最多 7 页 technical content 加 references。([AAAI][2])

推荐主文空间分配：

| 部分                                | 页数预算 | 内容                                           |
| --------------------------------- | ---: | -------------------------------------------- |
| Abstract + Figure 1               |  0.9 | frontier + artifact 或 frontier + qualitative |
| Intro + contributions             |  0.8 | IDT failure + style-ID task + LBM frontier   |
| Related work                      | 0.45 | reference-guided / style-ID / evaluation     |
| Method + Figure 2                 |  1.2 | active objective + modules + variant spec    |
| Experiments                       |  2.3 | protocol + Table 1 + Figure 3 + cost         |
| Discussion/limitations/conclusion |  0.8 | local diagnostic + limitations               |

需要删的内容：

* Discussion 中重复实验数字；
* historical strict-750 的长段；
* Related Work 的方法名列表；
* Conclusion 第二段 tokenizer 机制细节；
* “future papers should ...” 这种 editorial 句子。

不要删：

* Table 1；
* Figure 1A；
* style-ID inference boundary；
* Operating-point variants；
* limitations 中 successor closure；
* Seedream protocol 简述；
* reproducibility statement。

---

# 10. 最终建议的审稿策略

## 当前版本直接投的风险

直接投可以，但风险点很明确：**reviewer 会把最强 successor rows 视作未完全验证的 frontier points**。一个支持者会说 “interesting evaluation diagnostic + promising frontier”；一个反对者会说 “strongest results are not fully closed and variants are underspecified”。这就是 5.8–6.1 的原因。

## 最值得做的四件事

按收益排序：

1. **把 LBM-Knee 做成 fully closed main point。**
   补 tw-ArtFID、MUSIQ/MANIQA/DISTS、qualitative、3 seeds 或至少 bootstrap。

2. **把 PS/PS-v2 方法定义公式化。**
   不再说 “stronger successor family”，而是写清 endpoint selector、pattern/stroke statistics、viscosity/kinetic 权重。

3. **加一张 qualitative figure 到主文。**
   视觉任务没有 qualitative 会被扣分。哪怕 2 行也比没有强。

4. **补 non-CLIP style classifier 或 VLM/human preference。**
   这是专门防 CLIP circularity 的。

## 完成后的预期

| 改动完成度                              |         预估分 |
| ---------------------------------- | ----------: |
| 只修 writing 和 variant spec          | **6.1–6.3** |
| 加 qualitative + successor artifact | **6.3–6.6** |
| 再加 non-CLIP 或 VLM/human            | **6.6–6.8** |
| 再加 3 seeds / second split          | **6.8–7.0** |

---

# 11. 一句话总结

**这篇现在已经有 AAAI 投稿价值：评价贡献清楚，frontier 结果也够强；但要从 borderline 变成稳的 weak accept，必须把 strongest successor 从“frontier evidence”升级为“closed main result”。** 当前最优主线应是：

> **IDT calibration exposes a no-op failure in art-to-art style transfer; LBM is a compact style-ID latent transport family that offers selectable, low-cost style-content operating points; LBM-Knee is the main closed Pareto point, while PS/PS-v2 define the high-style frontier.**

[1]: https://aaai.org/conference/aaai/aaai-27/ "AAAI-27 - AAAI"
[2]: https://aaai.org/conference/aaai/aaai-26/main-technical-track-call/ "Main Technical Track: Call for Papers - AAAI"
[3]: https://aaai.org/conference/aaai/aaai-26/supplementary-material/ "AAAI-26 Supplementary Material - AAAI"
