我会把重点改成：**你们现在不是缺性能，而是缺“可审稿的性能叙事”。** 当前图里 168 个点说明 LBM 家族已经有高-style前沿，但主稿如果继续只讲 LBM-F/K，会把自己包装成“+0.02/+0.03 的小增益方法”。当前稿件 Table 1 只把 LBM-F/K 作为主结果，且自己承认 Distinct5 gains modest、缺 human/VLM validation，这会压分。 但从你给的新图看，Pattern+Stokes / XPred 系列已经能把 transfer CLIP-style 推到 0.71–0.73 附近，故事应改成 **LBM family produces a Pareto frontier, with selectable operating points for high-style, knee-point, and high-preservation regimes**。

下面给你非常具体的 writing 和画图改法。

---

## 1. 主线 writing：从“一个点赢”改成“LBM Pareto frontier”

不要再让 LBM-F/K 当唯一主角。它们应该是 **conservative / low-damage operating points**。真正的主角应该是：

| 角色         | 建议命名                             | 作用                             |
| ---------- | -------------------------------- | ------------------------------ |
| 保守点        | **LBM-Conservative / LBM-K**     | 证明在低 LPIPS 下稳定超过 IDT           |
| knee point | **LBM-Pareto-Knee**              | 主表 headline，兼顾 style 和 content |
| 高风格点       | **LBM-PS / LBM-Pattern+Strokes** | 证明 LBM 不只是小增益方法                |
| 高保真点       | **LBM-Kinetic / LBM-Preserve**   | 证明 family 能进入高 1-LPIPS 区间      |
| 探索云图       | XPred / Pattn / Stokes variants  | 放 appendix 或 Figure 1 背景，不逐个解释 |

Related/evaluation 叙事要压住 reviewer 对单指标的质疑。已有 calibrated metrics 工作把 style effectiveness 和 content coherence 分开看，并强调 style transfer 方法常在 Pareto frontier 上权衡；ArtFID 也把 stylization performance 分成 style matching 和 content preservation，并用 user study 验证自动指标；2024 的 comprehensive evaluation 继续指出 AST 评价协议不一致，需要主观和客观评价结合。你们的新图正好应该被写成“Pareto landscape”，而不是“我们某个点 CLIP-S 更高”。([CVF开放获取][1])

### 建议改摘要里的结果句

现在的 abstract 还是旧叙事：“LBM-F/K exceed IDT by +0.0244/+0.0312”。这个太弱。改成：

On Distinct5-512, the LBM family forms a controlled style-content frontier rather than a single fixed operating point. Conservative LBM checkpoints exceed the IDT floor with low content displacement, while the Pattern+Strokes and XPred variants move to a high-style regime that approaches or exceeds the strongest reproduced baseline without collapsing into the same damage profile. We therefore report three representative operating points: a low-damage point, a Pareto-knee point, and a high-style point. This presentation separates the evaluation contribution—IDT-calibrated target-style movement—from the model contribution: a compact style-ID latent renderer whose operating point can be selected according to the desired style-content budget.

这里要填具体数值时，不要从图上手抄，必须从同一个 eval json 自动生成。格式建议是：

> “LBM-PS reaches transfer CLIP-S **X.XXXX** at 1−LPIPS **Y.YYYY**, improving ΔIDT,tr by **+Z.ZZZZ**, while SaMST reaches **...** but with **...** artifact penalty.”

---

## 2. 命名必须立刻统一

现在图里的名字对 reviewer 很危险：`Pattn+Stokes`、`Pattn+Stokes002`、`XPred proximal`、`immortal`、`legacy + latent + immortal + paper-facing extras` 都像内部实验日志。主稿里不要出现这种名字。

建议统一成：

| 内部名             | 论文名                                |
| --------------- | ---------------------------------- |
| Pattn+Stokes    | **LBM-PS** 或 **LBM-PatternStroke** |
| Pattn+Stokes002 | **LBM-PS-v2**                      |
| XPred core      | **LBM-XPred-Core**                 |
| XPred proximal  | **LBM-XPred-Prox**                 |
| kinetic-only    | **LBM-Kinetic**                    |
| SaMAM-latent    | **LBM-LatentSSM** 或直接放 appendix    |
| LBM legacy      | **LBM-Base**                       |

主文里只保留 3–4 个正式变体名。其他 160+ 点统一叫 **development traces** 或 **recorded operating points**，放 appendix。

---

## 3. Figure 1 不要放现在这张 168 点大图

这张图适合 appendix，不适合 page 1。它的问题很明显：

1. **信息太多**：168 个点、10 类 legend、很多线、很多内部名，审稿人 10 秒看不懂。
2. **线条含义不清**：黑线像 Pareto frontier，但它连接的点是不是同一模型系列？如果不是，会被认为误导。
3. **label 太多且遮挡**：Pattern+Stokes 区域、LBM-F/K 区域、SaMAM 区域都有重叠。
4. **“all recorded operating points” 像开发日志**：reviewer 会担心 cherry-picking 和 researcher degrees of freedom。
5. **纵轴仍是 raw transfer CLIP-style**：既然论文主张 IDT calibration，主图最好用 **ΔIDT,tr**，而不是 absolute CLIP-S。

---

## 4. 主图建议：做成 2-panel 或 3-panel

### Figure 1A：IDT-calibrated Pareto frontier

**横轴**：Content preservation, `1 − LPIPS`
**纵轴**：`ΔIDT,tr = transfer CLIP-S − IDT transfer floor`
**虚线**：y = 0，标注 “IDT floor”
**点**：只标 7 个左右。

保留这些点：

* IDT
* SaMAM 2250 或 SaMAM 1500
* SaMST e15
* LBM-F/K conservative
* LBM-PS high-style
* LBM-XPred 或 LBM-PS-v2 best frontier
* LBM-Kinetic high-preservation

所有其他 160 个点可以作为浅灰色背景点，alpha=0.12，不进 legend。这样 reviewer 一眼能看出：**LBM 不是一个点，而是支配或逼近 baseline 的 frontier**。

图标题不要写：

> Distinct5-512 transfer landscape with all recorded operating points

改成：

> **IDT-calibrated style-content frontier on Distinct5-512**

或者：

> **LBM variants trace a controllable style-content frontier**

### Figure 1B：artifact check

只看 CLIP-S/LPIPS 还不够，因为 style transfer 评价需要同时看 style matching、content preservation 和 artifact。ArtFID 和 calibrated metrics 的相关工作都支持这个 framing。([Springer][2])

建议 Figure 1B 做：

**横轴**：targetwise ArtFID 或 artifact index
**纵轴**：ΔIDT,tr
**越左上越好**。

如果 Pattern+Stokes 高-style点 ArtFID 不差，这张图会非常强。如果 ArtFID 一般，也可以改成 bar plot：

| Method | tw-ArtFID ↓ | MUSIQ ↑ | MANIQA ↑ |
| ------ | ----------: | ------: | -------: |

Figure 1B 的目的不是证明 style 更高，而是证明高-style点不是坏图。

### Figure 1C：qualitative strip

放 2 个 transfer directions，每个方向 5 列：

`Source / IDT / SaMST / LBM-K / LBM-PS / target-domain samples`

注意：target-domain samples 不是 inference input，要在 caption 里写清楚 “shown only to visualize the target domain”。

---

## 5. 当前这张 168 点图怎么处理

把它移动到 appendix，标题改成：

> **Appendix Fig. A1: Complete recorded development landscape**

caption 里必须写清楚：

> “This figure is not used as a leaderboard. It records all retained development/evaluation points under the shared Distinct5-512 packet. The main paper selects representative operating points using the fixed criteria in Table X.”

否则 reviewer 会问：你们是不是看了所有点之后挑最好看的？

主文只需要一句：

> “The complete 168-point development landscape is provided in Appendix Fig. A1; the main paper reports fixed representative operating points selected from the non-dominated frontier.”

---

## 6. 主表要重做：从 “methods table” 改成 “operating points table”

建议 Table 1 变成这样：

| Method / point | Family   | ΔIDT,tr ↑ | CLIP-S,tr ↑ | 1−LPIPS ↑ | LPIPS ↓ | tw-ArtFID ↓ | MUSIQ ↑ | Train time | Role                |
| -------------- | -------- | --------: | ----------: | --------: | ------: | ----------: | ------: | ---------: | ------------------- |
| IDT            | control  |         0 |           — |     1.000 |       0 |           — |       — |          — | no-op floor         |
| SaMAM          | baseline |         — |           — |         — |       — |           — |       — |          — | compact baseline    |
| SaMST          | baseline |         — |           — |         — |       — |           — |       — |          — | high-style baseline |
| LBM-K          | ours     |         — |           — |         — |       — |           — |       — |          — | conservative        |
| LBM-PS         | ours     |         — |           — |         — |       — |           — |       — |          — | high-style          |
| LBM-XPred      | ours     |         — |           — |         — |       — |           — |       — |          — | Pareto frontier     |
| LBM-Kinetic    | ours     |         — |           — |         — |       — |           — |       — |          — | high-preservation   |

**最重要：Table 1 不要把 all-pairs CLIP-S 和 transfer-only CLIP-S 混用。** 你们现在图是 transfer CLIP-style，旧表是 all-pairs CLIP-S + transfer-only ΔIDT，很容易让 reviewer confused。主表建议统一成 **transfer-only**，identity/all-pairs 放 appendix。

---

## 7. Figure caption 直接改成这种风格

当前 caption 太像解释实验日志。建议主图 caption 用下面这种：

Figure 1: IDT-calibrated Distinct5-512 style-content frontier. The x-axis measures content preservation as 1−LPIPS and the y-axis reports transfer-only target-style movement above the unchanged-image floor, ΔIDT,tr. Gray points show retained development checkpoints under the same evaluation packet; labeled markers are the fixed operating points reported in Table 1. LBM variants trace a controllable frontier: conservative checkpoints remain in the low-displacement region, while Pattern+Strokes and XPred variants move into the high-style regime. The right panel checks whether the high-style points pay for this gain through targetwise ArtFID and perceptual artifact metrics.

---

## 8. Results section 重写顺序

现在建议把 Results 改成四段，而不是一段里混 SaMAM/SaMST/LBM-F/K。

### 8.1 先讲 no-op failure

> IDT is high, so raw CLIP-S alone is unsafe.

这段保留你们原来的核心评价贡献。

### 8.2 再讲 Pareto frontier

> LBM family spans multiple regimes; F/K 是 conservative，PS/XPred 是 high-style。

不要写成“we tried many variants”。写成：

> “We report three fixed operating points selected from the non-dominated frontier.”

### 8.3 再讲 artifact/content sanity

> high-style LBM 是否比 SaMST 更干净。

这里接 ArtFID/MUSIQ/MANIQA/DISTS，避免 reviewer 说你们只是在 hack CLIP。

### 8.4 最后讲 cost

> minute-scale retraining / small model / style-ID interface。

这段是加分项，但不要让它压过性能主线。

---

## 9. 写作上必须删除或降级的句子

少用这些：

| 现在的表达                                              | 问题        | 改法                                                                 |
| -------------------------------------------------- | --------- | ------------------------------------------------------------------ |
| “first trusted checkpoint”                         | 像内部筛选     | “first retained checkpoint under the fixed evaluation packet”      |
| “legacy + latent + immortal + paper-facing extras” | 内部日志感     | 只放 appendix，主文不出现                                                  |
| “doing nothing can score as target style”          | 口语        | “the unchanged source can obtain high target-style affinity”       |
| “not claimed as a full bridge solver” 反复出现         | 过度防守      | Method 一次说明，Limitations 一次说明即可                                     |
| “Pattn+Stokes002”                                  | 内部实验名     | “LBM-PS-v2”                                                        |
| “upper-right is better” 小角标                        | 可保留，但不够学术 | 轴名直接写 “Content preservation ↑” 和 “IDT-calibrated style movement ↑” |

---

## 10. 消除 cherry-picking 风险：加一个固定选点规则

你们现在有 168 点，这既是优势也是风险。一定要写清楚怎么选主点。建议定义：

> “A point is reportable if it is non-dominated under ΔIDT,tr, 1−LPIPS, and targetwise ArtFID, and if its checkpoint/config has a retained log before final table construction.”

更简单一点：

> “We select three operating points before qualitative inspection: the highest-style non-dominated LBM point, the highest-preservation positive-ΔIDT point, and the Pareto-knee point maximizing normalized ΔIDT,tr + normalized(1−LPIPS) − normalized(tw-ArtFID).”

这句话很重要。否则 reviewer 会觉得你们从 168 点里挑漂亮点。

---

## 11. 定性图不要再放巨大 5×5 grid

AAAI 主文空间很贵。建议：

**Figure 3：Qualitative comparison**

每个 transfer direction 一行：

`Source | IDT | SaMAM | SaMST | LBM-K | LBM-PS | target samples`

选 3 个方向即可：

1. 一个 LBM-PS 明显赢 style 的方向；
2. 一个 SaMST 过度破坏、LBM 更干净的方向；
3. 一个 LBM 失败或接近失败的方向。

第三个失败例子建议放主文或 appendix？如果空间允许，放 appendix，但主文 discussion 要承认 failure mode。现在这个领域评价争议很大，主动展示失败例子会增加可信度；2024 comprehensive evaluation 也强调 objective + subjective 多粒度评价比单一指标更可靠。([PubMed][3])

---

## 12. 最终推荐的论文结构

我建议按这个顺序排：

1. **Figure 1**：IDT-calibrated Pareto frontier + artifact check
2. **Table 1**：selected operating points，不超过 7 行
3. **Figure 2**：method diagram，删掉 LANCET，删掉 inactive OT/Sinkhorn 主路径
4. **Table 2**：incremental ablation：Base → Queue → XPred → Pattern → Pattern+Strokes
5. **Figure 3**：qualitative strips
6. **Appendix Fig. A1**：168-point complete landscape
7. **Appendix Table A1**：全部 config + checkpoint + seed + metrics

---

## 13. 一句话总结你们现在该怎么改

**把现在的 168 点图当作“证据矿”，不要当作主图。主图只展示 selected Pareto frontier；主文只讲 LBM-K / LBM-PS / LBM-XPred 三个正式 operating points；用 ΔIDT,tr 做纵轴，用 artifact 指标守住高-style点的可信度。**

这样写出来，reviewer 看到的不是“我们调了很多点，其中几个不错”，而是：

> **LBM is a controllable style-ID latent renderer whose variants trace a measurable IDT-calibrated style-content frontier.**

[1]: https://openaccess.thecvf.com/content_WACV_2020/papers/Yeh_Improving_Style_Transfer_with_Calibrated_Metrics_WACV_2020_paper.pdf?utm_source=chatgpt.com "Improving Style Transfer with Calibrated Metrics - CVF Open Access"
[2]: https://link.springer.com/chapter/10.1007/978-3-031-16788-1_34?utm_source=chatgpt.com "ArtFID: Quantitative Evaluation of Neural Style Transfer - Springer"
[3]: https://pubmed.ncbi.nlm.nih.gov/39320993/?utm_source=chatgpt.com "A Comprehensive Evaluation of Arbitrary Image Style Transfer Methods ..."
