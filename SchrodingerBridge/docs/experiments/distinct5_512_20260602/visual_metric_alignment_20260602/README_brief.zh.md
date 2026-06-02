# Distinct5-512 指标与视觉现象简记

Updated: 2026-06-02

## Visual Panel

![Distinct5 visual alignment](distinct5_visual_alignment_grid.jpg)

## Key Data

全量数据口径：transfer-only 600 张。下表只列和图中现象最相关的聚合结果。

| method | clip_style | LPIPS | aggregate ArtFID | observation |
|---|---:|---:|---:|---|
| No-op | 0.639921 | 0.000000 | 1.001099 | 原图不动，内容和艺术域保持最强 |
| SaMAM-2250 | 0.552252 | 0.360452 | 148.205852 | 明显改图，但目标风格方向不稳定 |
| LANCET-F e1 | 0.664360 | 0.324528 | 126.825714 | 有目标风格增益，代价相对较低 |
| LANCET-K e1 | 0.671167 | 0.372281 | 161.957657 | style 更高，但漂白和平涂更重 |

图中局部样例：

| pair | no-op clip | SaMAM clip / LPIPS | LANCET-F clip / LPIPS | LANCET-K clip / LPIPS |
|---|---:|---:|---:|---:|
| Early_Renaissance -> Minimalism | 0.607 | 0.566 / 0.387 | 0.710 / 0.498 | 0.734 / 0.640 |
| Early_Renaissance -> Ukiyo_e | 0.663 | 0.571 / 0.321 | 0.724 / 0.396 | 0.712 / 0.450 |
| Impressionism -> Minimalism | 0.602 | 0.623 / 0.585 | 0.823 / 0.483 | 0.809 / 0.627 |
| Minimalism -> Rococo | 0.558 | 0.468 / 0.580 | 0.588 / 0.342 | 0.590 / 0.383 |
| Rococo -> Ukiyo_e | 0.515 | 0.539 / 0.542 | 0.599 / 0.295 | 0.610 / 0.343 |
| Ukiyo_e -> Early_Renaissance | 0.685 | 0.567 / 0.315 | 0.675 / 0.250 | 0.641 / 0.267 |

## Phenomenon

1. `No-op` 是强基线。它没有做风格迁移，但因为 source 本身已经是真实 WikiArt 图像，所以 LPIPS 和 aggregate ArtFID 极好。
2. `SaMAM` 并不是没有变化。视觉上有调色、对比度和纹理变化，但很多样例的 `clip_style` 低于 no-op，说明变化没有稳定指向目标风格。
3. `LANCET-F/K` 的 `clip_style` 提升和视觉变化一致。它们确实把图像推离 source，但主要表现为漂白、雾化和平涂化。
4. `LANCET-F` 当前更均衡；`LANCET-K` style 更高，但结构损伤和 ArtFID 代价更大。

## Likely Cause

Distinct5 的目标类差异很大，尤其包含 `Minimalism`。当前 LANCET 的最容易优化路径是低频测度漂移：降低局部纹理、压平颜色、提高画面亮度和均匀度。这会快速靠近某些目标风格原型，因此 `clip_style` 上升；但它不是稳定的笔触/结构迁移，所以 LPIPS 和 ArtFID 代价明显。

SaMAM 的问题相反：它会改变图像，但缺少稳定的目标风格表征约束，变化常落在“泛艺术增强”或错误色彩方向上，因此在 Distinct5 上甚至低于 no-op。

## Takeaway

这组结果不是单纯的 metric hacking。更准确地说：

- LPIPS / aggregate ArtFID 说明 no-op 是强内容和艺术域基线。
- `clip_style - no_op_clip_style` 才能说明目标风格移动。
- 视觉面板显示 LANCET 的目标移动真实存在，但目前过度依赖低频漂白和平涂化。

下一步应减少低频平涂捷径，增强可控的中高频 stroke / edge / texture 表征。
