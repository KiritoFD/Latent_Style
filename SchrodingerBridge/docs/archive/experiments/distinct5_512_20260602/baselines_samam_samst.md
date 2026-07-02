# Distinct5-512 Baselines: SaMAM / SaMST

更新时间：2026-06-02

## SaMAM 数据口径

SaMAM 使用同一 Distinct5-512 图像 split：

- 远程图片根目录：`/mnt/i/datasets/wikiart_distinct5_samam_512`
- train flat：`train_flat/content`, `train_flat/style`
- test flat：`test_flat/content`, `test_flat/style`
- 正式评估：同一 150 张 test image 扩展到 all 5x5 / 750 generated images

## SaMAM 当前有效 run

```text
/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag
```

状态：

- WSL venv：`/home/xy/venvs/samam312`
- 当前有效 batch：`6`
- 2000 已完成：`step-step=002000.ckpt` 已生成并完成 750 eval
- 继续训练：从 `step-step=002000.ckpt` 续到 2250
- 当前进程仍在跑，日志：`direct_continue_step_2250_20260602_124746.log`
- 本次状态检查时 GPU：约 7475 MiB，util 约 61%，power 约 106W
- 当前日志约在 3/250 steps；尚未生成 `step-step=002250.ckpt`

## SaMAM 已恢复评估曲线

来源：

```text
.../eval_curve/curve_metrics_recovered.csv
.../eval_curve/convergence_recovered.md
```

| step | count | clip_style | content_lpips | clip_content |
|---:|---:|---:|---:|---:|
| 250 | 750 | 0.547991 | 0.600625 | 0.656202 |
| 500 | 750 | 0.562997 | 0.542445 | 0.704642 |
| 750 | 750 | 0.565732 | 0.535965 | 0.711661 |
| 1000 | 750 | 0.565910 | 0.460542 | 0.757051 |
| 1250 | 750 | 0.578039 | 0.442945 | 0.796373 |
| 1500 | 750 | 0.580525 | 0.412826 | 0.841042 |
| 1750 | 750 | 0.577616 | 0.377244 | 0.857463 |
| 2000 | 750 | 0.583346 | 0.362153 | 0.866976 |

阶段判断：

- SaMAM Distinct5-512 的 `clip_style` 在 2000 step 反弹并成为当前最高。
- `content_lpips` 到 2000 step 仍明显下降，因此还不能说内容指标收敛。
- 继续到 2250 的目的，是确认 LPIPS 是否继续下降，以及 2000 的 style 反弹是否稳定。

## 无效 SaMAM run

```text
/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_mamba_b8_seg250_remote_wsl_20260601_1935
```

结论：

- b8 显存接近目标区间，但约 step 64 出 NaN。
- 没有 250-step 有效 checkpoint。
- 不作为 baseline 曲线，只能作为 batch-size stress failure。

## SaMST 状态

准备目录：

```text
/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samst_distinct5_512_prepared
```

启动脚本：

```text
/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samst_distinct5_512_prepared/run_samst_distinct5_512.sh
```

源码目录：

```text
/mnt/i/Github/Latent_Style/Related_Works/repos/SaMST-main
```

状态：

- 脚本已准备。
- 尚未启动正式 Distinct5-512 SaMST。
- 按当前优先级，应等 SaMAM 2000/后续收敛判断完成后再启动，避免两个 baseline 状态互相抢占和混淆。
