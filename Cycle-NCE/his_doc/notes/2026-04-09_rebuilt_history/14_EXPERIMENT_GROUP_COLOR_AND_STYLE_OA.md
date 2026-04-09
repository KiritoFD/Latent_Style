# 实验组专档：Color 组与 Style-OA / 联合优化组

## 1. Color 组的价值

Color 相关实验在这个项目里不是边缘问题，而是直接回答：

- 风格迁移为什么总会亮度漂移、发灰、偏色
- color loss 应该在 latent 空间里怎样定义

## 2. Color 组代表目录

从 `RESULTS_INDEX_20260330.csv` 可见，比较关键的有：

- `color_01_adain_wc2_tv05_r16_e60`
- `color_ablation_exp1_anchor_pseudo_adain_wc2_tv05_r16_e60`
- `color_ablation_exp2_tv_off_pseudo_adain_wc2_tv00_r16_e60`
- `color_ablation_exp3_stress_pseudo_adain_wc5_tv05_r16_e60`
- `color_ablation_exp4_dimtest_latent_adain_wc2_tv05_r16_e60`
- `color_mode_01_pseudo_rgb_adain_r16_e40`
- `color_mode_02_pseudo_rgb_hist_r16_e40`
- `color_mode_03_latent_decoupled_adain_r16_e40`

## 3. 能提炼出的主结论

### 3.1 直接在 latent 通道上做颜色统计不够稳

否则不会有后续：

- pseudo RGB adain
- pseudo RGB hist
- latent decoupled adain

这些分化路线。

### 3.2 伪 RGB + 低频统计是后期主胜法

`ed596c0` 已经把方向写死了：

- 通道映射回 RGB 的缩略图 color loss 大赢

所以正式结论可以写成：

- color supervision 应该尽量接近可感知颜色空间，并且聚焦低频颜色一致性

## 4. Style-OA 组

这是后期最像“参数联合寻优”的一组。

旧分析提炼出的结论：

- `w_swd = 60` 通常优于 `90`
- `w_identity = 3.0` 通常优于 `1.5`
- `w_color = 2.0` 比 `5.0` 更均衡
- `lr = 5e-4` 能把 style 顶上去，但更容易损伤 LPIPS

## 5. 代表实验：`style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10`

旧分析给出的关键 epoch：

- 30 epoch：style `0.6981`, lpips `0.4250`
- 60 epoch：style `0.7209`, lpips `0.4939`
- 100 epoch：style `0.7297`, lpips `0.5095`
- 120 epoch：style `0.7265`, lpips `0.5217`

这组结果特别有价值，因为它告诉我们：

- 不是简单“训越久越好”
- 后期 style 可能继续冲高，但内容/分布代价也持续上升
- 因此 early stopping 和 checkpoint 选择是这个项目的重要组成部分

## 6. 这组材料支持的正式结论

1. color 约束必须设计成可感知的低频统计任务。
2. 联合优化里存在明显甜点区，尤其是 SWD / identity / color 三者配比。
3. epoch 选择不是附属问题，而是结果定义的一部分。

