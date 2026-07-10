# 2026-07-10 四指标评估审计

## 1. 结论

本次审计确认了两个独立问题：

1. 临时 DINO 评估脚本使用了错误定义；
2. 论文 D5 WEAVE 主表行混合了两个不同生成输出包的指标。

因此，旧的 `0.715 / 0.382 / 0.778 / 0.492` 不是一个真实、同源的 operating point，已经从
`aaai2027_v4/paper.tex` 撤回。

## 2. 错误 DINO 定义

错误临时脚本采用：

- `DINO-S = cos(CLS(gen), mean(CLS(style refs)))`；
- `DINO-C = 1 - patch self-similarity MSE`。

这导致本地 T11 被错误记录为 `DINO-S=0.3958, DINO-C=0.9752`。其中第二个数值并不是内容 CLS
相似度，只是 structure distance 的变形，因此与论文 DINO-C 不同量纲。

统一 canonical 定义为：

- `DINO-S = max_ref cos(CLS(gen), CLS(target-style reference))`；
- `DINO-C = cos(CLS(gen), CLS(source))`；
- `DINO-structure = MSE(SSM_patch(gen), SSM_patch(source))`，单独记录，越低越好。

实现入口为 `src/utils/compute_dino_metrics.py`。预处理固定为 224 bicubic resize、center crop、ImageNet
normalization，backbone 为 DINOv2-small，style reference 上限为每类 30 张。

## 3. 数值复现

修复后的脚本重算历史 WEAVE 输出包：

- `DINO-S=0.491726`；
- `DINO-C=0.778188`；
- `DINO-structure=0.025398`。

与历史 canonical 汇总的差异低于 `1e-5`，确认新脚本实现与论文原始 DINO 协议一致。

本地 T11 seed 42 的正确四指标为：

- `CLIP-S=0.7204`；
- `LPIPS=0.2857`；
- `DINO-S=0.4736`；
- `DINO-C=0.7759`。

## 4. 论文混表来源

旧论文行：

`CLIP-S=0.715, LPIPS=0.382, DINO-C=0.778, DINO-S=0.492`

实际来源为：

- `0.715 / 0.382`：`exp/swd_cm_sem_r8/full_eval/epoch_0005`；
- `0.778 / 0.492`：`results/D5-512/weave`。

对 `swd_cm_sem_r8` 的同一批图片运行 canonical DINO 后得到：

- `DINO-S=0.466100`；
- `DINO-C=0.688840`；
- `DINO-structure=0.027754`。

因此旧行把 semantic-region SWD 的 CLIP/LPIPS 与另一输出包的 DINO 拼在一起。该结果不可用于 baseline、消融或
论文主张。

## 5. 当前基线状态

历史 `clean_base_v2_local` 的同源 750 图输出约为：

- `CLIP-S=0.7292`；
- `LPIPS=0.3239`；
- `DINO-S=0.4874`；
- `DINO-C=0.7688`。

但对应 `epoch_0010.pt` 已不存在，只留下 config 和生成图，因此这组数值只能作为重训参考。新的可复现基线为
`configs/710_b0_weave_d5.json`，必须重新训练并闭合四指标。

## 6. Identity 注意事项

若 style reference bank 包含 source 本身，identity 输出会通过 exact/self-like reference 抬高 DINO-S。IDT 校准必须使用
`--exclude_source_from_style_refs`。正式模型报告同时保存：

- all-pairs 四指标；
- off-diagonal 四指标；
- 每个 target style 的 breakdown。

这不是增加新的主指标，而是防止数据配对结构掩盖模型行为。

## 7. 防复发协议

每个正式评估目录必须包含一个不可手工拼接的 manifest：

- checkpoint 绝对路径与 SHA256；
- checkpoint 内嵌 config hash 与外部 override hash；
- git commit；
- 数据集和 reference bank 路径；
- 750 个 `src/tgt/gen` 映射；
- 每张生成图 SHA256；
- evaluator 版本和 DINO model revision；
- num steps、solver、style strength、endpoint 参数；
- schema 默认值展开后的 effective endpoint mode；
- 是否启用后处理；
- `metrics.csv`、`dino_metrics.csv` 和 summary 的 hash。

汇总脚本必须检查：

1. `metrics.csv` 与 `dino_metrics.csv` 行数均为 750；
2. 两者的 `src/tgt/gen` key 完全一致；
3. 生成文件无缺失、无重复；
4. 四项主指标都来自同一个 manifest；
5. summary 不允许从不同目录按列拼接；
6. 论文表格只能从机器生成的单一 summary 导入。

## 8. 代码与文档变更

- canonical evaluator：`src/utils/compute_dino_metrics.py`；
- 新 WEAVE 闭环基线：`configs/710_b0_weave_d5.json`；
- style 实验计划：`docs/710/STYLE_IMPROVEMENT_PLAN.md`；
- infra 优化方案：`docs/710/INFRA_OPTIMIZATION_PLAN.md`；
- 总路线与当前正确数值：`docs/710/README.md`；
- 论文 D5 WEAVE 行已暂时置为 `--`，等待同源重训结果。
