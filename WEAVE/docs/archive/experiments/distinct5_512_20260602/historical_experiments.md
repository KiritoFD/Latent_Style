# Historical Experiments Separated From Distinct5-512

更新时间：2026-06-02

本文件只做历史对照隔离。下面结果不属于 Distinct5-512 新数据集，不能并入 Distinct5 主表。

## LANCET / WikiArt512 historical best

历史 750-image full eval 中，曾记录到：

| run | clip_style | content_lpips | 备注 |
|---|---:|---:|---|
| local WSL WikiArt512 hist b32 e8 | 0.792298 | 0.355038 | classifier 删除后的计时 eval 结果 |
| split_axis_geometry_derivgate_fieldbound_g005_120b | 0.801055 | 0.307672 | 早期历史最优记录，需和对应数据/配置一起引用 |

这些结果用于说明老 WikiArt512/历史分布下 LANCET 能达到的上限，但不能直接作为 Distinct5-512 的当前性能。

## SaMAM historical WikiArt512 / 512

来源：`Related_Works/baseline_pipeline/results/convergence_summary_20260601/README.md`

| step | clip_style | content_lpips |
|---:|---:|---:|
| 1000 | 0.725534 | 0.555994 |
| 3000 | 0.786911 | 0.342996 |
| 5000 | 0.791244 | 0.283292 |
| 6000 | 0.788131 | 0.264603 |
| 7000 | 0.784850 | 0.246103 |

结论：

- 这是旧 WikiArt512 / SaMAM512 收敛曲线，不是 Distinct5-512。
- style 在 5000 step 峰值，LPIPS 到 7000 step 仍下降。
- 它解释了 SaMAM 为什么能通过后期训练持续刷 LPIPS，但不能替代新 Distinct5-512 上的 SaMAM 曲线。

## SaMAM historical 256

| step | clip_style | content_lpips |
|---:|---:|---:|
| 5000 | 0.684885 | 0.534389 |
| 10000 | 0.687492 | 0.473146 |
| 14000 | 0.696867 | 0.436278 |
| 17000 | 0.695625 | 0.419127 |
| 20000 | 0.694062 | 0.409598 |
| 25000 | 0.693823 | 0.393958 |

结论：

- 256 线 style 平台明显，继续训练主要降低 LPIPS。
- 由于分辨率不同，不应拿来和 Distinct5-512 直接主表对齐。

## SaMST historical status

旧 WikiArt512 SaMST 30-epoch 复现当时未完全结束：

| target | current epoch | target epoch |
|---|---:|---:|
| Realism | 30 | 30 |
| Impressionism | 30 | 30 |
| Post_Impressionism | 21 | 30 |
| Expressionism | 15 | 30 |
| Symbolism | 15 | 30 |

另外有 strict 750 历史结果：

| clip_style | content_lpips | 推理时间 |
|---:|---:|---:|
| 0.7194 | 0.4664 | 39.826s / 750 |

这些都不是 Distinct5-512 的正式 SaMST 结果。

## 使用规则

1. 新论文主表如讨论 Distinct5-512，只能使用 `README.zh.md`, `lancet_runs.md`, `baselines_samam_samst.md` 中的新数据集结果。
2. 历史结果只能用于方法背景、收敛行为解释、或跨数据集讨论。
3. 如果要对比 SaMAM 是否“赢 LANCET”，必须在同一数据集、同一 750 all-pairs 评估口径下比较。
