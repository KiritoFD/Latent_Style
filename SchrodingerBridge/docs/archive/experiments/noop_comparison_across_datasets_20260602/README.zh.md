# No-op 对照下的三组数据集结果整理

Updated: 2026-06-02

## Scope

本记录把当前三组数据集分开整理，并统一加入 no-op 对照：

1. `Legacy256 / overfit50`
   - 256 分辨率旧五风格数据集。
   - styles: `Hayao`, `cezanne`, `monet`, `photo`, `vangogh`。
2. `WikiArt512-5style / 3600`
   - WikiArt 512 五风格数据集，每类约 3600 train，30 test。
   - styles: `Realism`, `Impressionism`, `Post_Impressionism`, `Expressionism`, `Symbolism`。
3. `Distinct5-512 / 1000`
   - 差异更大的 WikiArt 512 五风格数据集，每类约 1000 train，30 test。
   - styles: `Early_Renaissance`, `Impressionism`, `Minimalism`, `Rococo`, `Ukiyo_e`。

No-op 定义：按标准 5x5 推理矩阵生成结果，但每一行完全复制同一张 source 图，不做任何变化。
因此 full 为 750 张，transfer-only 为去掉对角线后的 600 张。

完整 keypoint CSV：

```text
docs/experiments/noop_comparison_across_datasets_20260602/dataset_noop_comparison_keypoints.csv
```

Distinct5 标准 no-op 5x5 图：

```text
docs/experiments/distinct5_512_20260602/visual_metric_alignment_20260602/noop_standard_5x5_grid.jpg
```

## Summary

| dataset | scope | method / point | clip_style | LPIPS | no-op clip | gain vs no-op | readout |
|---|---|---|---:|---:|---:|---:|---|
| Legacy256 | transfer | No-op | 0.616694 | 0.000000 | 0.616694 | 0.000000 | 内容保持上界 |
| Legacy256 | transfer | SaMAM 15k, best style | 0.673892 | 0.445060 | 0.616694 | +0.057198 | 有效风格增益，代价较高 |
| Legacy256 | transfer | SaMAM 25k, best LPIPS | 0.666977 | 0.402174 | 0.616694 | +0.050283 | LPIPS 改善，style 略降 |
| Legacy256 | transfer | LANCET S-add e8 | 0.692537 | 0.471155 | 0.616694 | +0.075843 | 旧 LANCET style 最强，但 LPIPS 也高 |
| WikiArt512-3600 | transfer | No-op | 0.773026 | 0.000000 | 0.773026 | 0.000000 | no-op 本身极强 |
| WikiArt512-3600 | transfer | SaMAM 5k, best style | 0.784589 | 0.283310 | 0.773026 | +0.011563 | 绝对 style 高，但相对 no-op 增益很小 |
| WikiArt512-3600 | transfer | SaMAM 10k, best LPIPS | 0.777356 | 0.164393 | 0.773026 | +0.004330 | 后期主要刷 LPIPS |
| Distinct5-1000 | transfer | No-op | 0.639921 | 0.000000 | 0.639921 | 0.000000 | 更难的风格域，no-op 不再虚高到 0.77 |
| Distinct5-1000 | transfer | SaMAM 1250, best style | 0.557183 | 0.448703 | 0.639921 | -0.082738 | 有变化但目标方向错误 |
| Distinct5-1000 | transfer | SaMAM 2250, best LPIPS | 0.552252 | 0.360452 | 0.639921 | -0.087669 | LPIPS 降，但 style 仍低于 no-op |
| Distinct5-1000 | transfer | LANCET F e1, best LPIPS | 0.664360 | 0.324528 | 0.639921 | +0.024440 | 当前较均衡点 |
| Distinct5-1000 | transfer | LANCET K e1, best style | 0.671167 | 0.372281 | 0.639921 | +0.031246 | style 最强，漂白/结构代价更大 |

## Dataset 1: Legacy256 / overfit50

| scope | method / point | clip_style | LPIPS | no-op clip | gain vs no-op |
|---|---|---:|---:|---:|---:|
| full | No-op | 0.661913 | 0.000000 | 0.661913 | 0.000000 |
| full | SaMAM 14k | 0.696867 | 0.436278 | 0.661913 | +0.034954 |
| full | SaMAM 25k | 0.693823 | 0.393958 | 0.661913 | +0.031910 |
| full | LANCET S-add e8 | 0.716724 | 0.461527 | 0.661913 | +0.054810 |
| transfer | No-op | 0.616694 | 0.000000 | 0.616694 | 0.000000 |
| transfer | SaMAM 15k | 0.673892 | 0.445060 | 0.616694 | +0.057198 |
| transfer | SaMAM 25k | 0.666977 | 0.402174 | 0.616694 | +0.050283 |
| transfer | LANCET S-add e8 | 0.692537 | 0.471155 | 0.616694 | +0.075843 |

结论：

- 旧 256 数据集上 no-op 不算特别强，模型确实能拿到明显 target-style gain。
- SaMAM 继续训练后 LPIPS 会下降，但 style 也从峰值回落。
- LANCET 旧 8 epoch 点的 no-op-adjusted style 最强，但 LPIPS 高于 SaMAM 25k。

## Dataset 2: WikiArt512-5style / 3600

| scope | method / point | clip_style | LPIPS | no-op clip | gain vs no-op |
|---|---|---:|---:|---:|---:|
| full | No-op | 0.781528 | 0.000000 | 0.781528 | 0.000000 |
| full | SaMAM 5k | 0.791244 | 0.283292 | 0.781528 | +0.009716 |
| full | SaMAM 10k | 0.785089 | 0.164336 | 0.781528 | +0.003561 |
| transfer | No-op | 0.773026 | 0.000000 | 0.773026 | 0.000000 |
| transfer | SaMAM 5k | 0.784589 | 0.283310 | 0.773026 | +0.011563 |
| transfer | SaMAM 10k | 0.777356 | 0.164393 | 0.773026 | +0.004330 |

结论：

- 这组 512/3600 数据集的 no-op clip_style 极高，是最明显的“强 no-op 基线”。
- SaMAM 绝对 `clip_style ~= 0.79` 看起来很好，但相对 no-op 的增益只有约 `+0.01`。
- 10k 后 LPIPS 明显更好，但 target-style gain 基本被压到 `+0.004` 级别。
- 当前没有同口径 LANCET 结果进入这个表；若要在论文主表使用，应补同一 split 的 LANCET 评估。

## Dataset 3: Distinct5-512 / 1000

| scope | method / point | clip_style | LPIPS | no-op clip | gain vs no-op |
|---|---|---:|---:|---:|---:|
| full | No-op | 0.680123 | 0.000000 | 0.680123 | 0.000000 |
| full | SaMAM 2000 | 0.583346 | 0.362153 | 0.680123 | -0.096777 |
| full | SaMAM 2250 | 0.581097 | 0.353820 | 0.680123 | -0.099026 |
| full | LANCET F e1 | 0.696915 | 0.318645 | 0.680123 | +0.016792 |
| full | LANCET K e1 | 0.700995 | 0.362294 | 0.680123 | +0.020872 |
| transfer | No-op | 0.639921 | 0.000000 | 0.639921 | 0.000000 |
| transfer | SaMAM 1250 | 0.557183 | 0.448703 | 0.639921 | -0.082738 |
| transfer | SaMAM 2250 | 0.552252 | 0.360452 | 0.639921 | -0.087669 |
| transfer | LANCET F e1 | 0.664360 | 0.324528 | 0.639921 | +0.024440 |
| transfer | LANCET K e1 | 0.671167 | 0.372281 | 0.639921 | +0.031246 |

结论：

- Distinct5 的风格差异更大，no-op 不再像 WikiArt512-3600 那样接近目标域。
- SaMAM 在这个 split 上实际低于 no-op：说明它有图像变化，但没有稳定朝目标风格移动。
- LANCET 是当前唯一在 transfer-only 上明显高于 no-op 的方法。
- F e1 是更均衡的点；K e1 style 更强，但视觉上更容易出现漂白、平涂和结构损伤。

## Cross-dataset Readout

1. `clip_style` 必须和 no-op 同表报告。绝对值在 512/3600 上会显著高估有效风格迁移。
2. `style_gain_vs_noop` 比绝对 `clip_style` 更能说明是否朝 target style 移动。
3. 256 和 Distinct5 上模型可以产生正向 target-style gain；512/3600 的 no-op 太强，SaMAM 的有效增益很小。
4. Distinct5 是目前更适合做表征探索的数据集，因为它能区分“保持艺术域”和“真正迁移到目标风格”。
5. 后续 ArtFID 必须按 target-wise/per-target 口径重算；之前的 aggregate ArtFID 混合 5 个 target，不能用于判断 no-op 是否靠近每个目标域。
