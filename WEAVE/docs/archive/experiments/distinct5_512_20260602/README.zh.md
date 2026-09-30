# Distinct5-512 实验目录

更新时间：2026-06-02

本目录只记录新的 Distinct5-512 数据集相关实验。历史 WikiArt512 / SaMAM512 / SaMST 复现实验只放在 `historical_experiments.md` 里作为外部对照，不并入本数据集主表。

## 数据集结论

本地和远程使用的是同一份 Distinct5-512 图像数据：

- 类别：`Early_Renaissance`, `Impressionism`, `Minimalism`, `Rococo`, `Ukiyo_e`
- train：每类 1000 张，共 5000 张
- test：每类 30 张，共 150 张
- eval：`5 source styles x 30 images x 5 target styles = 750`

本次用统一 Python 排序后的文件名集合 hash 核过本地 `F:\wikiart_distinct5_512_images` 和远程 `/mnt/i/datasets/wikiart_distinct5_512_images`。五类 train/test 数量一致，文件名集合一致。早期文档里出现的 `/mnt/f/...` 是过期路径；远程正式路径以 `/mnt/i/...` 为准。

详细路径和 hash 见 `dataset_audit.md`。

## 当前新数据集实验状态

| 线 | 状态 | 当前结论 |
|---|---|---|
| LANCET Distinct5-512 | 已完成 A-M 表征消融 | F 是 LPIPS 最优，H 是均衡点，K 是 style 最优 |
| SaMAM Distinct5-512 | 已评估到 2250 step，后续收敛包待补齐 | 当前已评估点的 transfer CLIP-S 均低于 IDT；LPIPS 只记录发生了非零改动，不是失败定义 |
| SaMAM b8 stress | 无效 | 约 step 64 出 NaN，不作为基线 |
| SaMST Distinct5-512 | 已准备，未启动正式训练 | 等 SaMAM 收敛判断后再跑 |

## LANCET 当前 Pareto

| 选择 | run | epoch | clip_style | content_lpips | 判断 |
|---|---|---:|---:|---:|---|
| LPIPS 最优 | Variant F | 1 | 0.696915 | 0.318645 | 内容保持压力基线 |
| 均衡点 | Variant H | 1/2 | 0.699383 | 0.321333 | 当前综合基线 |
| style 最优 | Variant K | 1 | 0.700995 | 0.362294 | style-only 保留，LPIPS 代价高 |

## 文档结构

- `dataset_audit.md`：数据集身份、路径、数量和文件名 hash 核验。
- `lancet_runs.md`：LANCET 新数据集所有主要变体、远程目录、结论。
- `baselines_samam_samst.md`：SaMAM / SaMST 在 Distinct5-512 上的当前状态。
- `metric_landscape.md`：`clip_style` vs `1-content_lpips` 可视化和当前读图结论。
- `historical_experiments.md`：非 Distinct5 历史实验隔离整理，避免和新数据集混表。

## 原始来源

- `../2026-06-02-distinct5-512-lancet-representation-summary.zh.md`
- `../2026-06-02-distinct5-512-lancet-representation-speed.md`
- `../2026-06-01-samam-distinct5-b8-status.md`
- `../../../../Related_Works/baseline_pipeline/results/convergence_summary_20260601/README.md`
