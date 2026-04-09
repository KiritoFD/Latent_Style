# 实验榜单与均衡性附录

数据源：`Y:\experiments\EXPERIMENT_RECORD_FULL_DATA.csv`

## 1. 为什么要单独做榜单

如果只看一个指标，这个项目很容易得出错误结论。

最典型的例子就是：

- `exp_S1_zero_id` style 全表最高
- 但 FID / art-FID / LPIPS 全面恶化

所以榜单必须拆开看。

## 2. 按 style 排名前列

前几名大致是：

1. `exp_S1_zero_id`
2. `exp_S2_color_blind`
3. `G0-Base-Gain0.5`
4. `final_demodulation`
5. `ablate_A3_p5_id030_tv005`

### 2.1 结论

这份榜单最有价值的地方不是“谁赢了”，而是说明：

- style 高分经常来自弱化 identity、弱化 color、放大 skip 或放大胆量更大的结构
- 它不是可靠的最终优胜依据

## 3. 按 FID 排名前列

前几名大致是：

1. `exp_G1_edge_rush`
2. `exp_3_macro_strokes`
3. `exp_2_zero_id`
4. `exp_1_control`
5. `exp_4_zero_tv`

### 3.1 结论

主线 EXP 组在 FID 维度反而相当强，尤其：

- `exp_G1_edge_rush`
- `exp_3_macro_strokes`

这说明：

- “最稳的结果”往往还在主线调参，而不一定在后期更激进的结构试验里

## 4. 按 art-FID 排名前列

前几名大致是：

1. `exp_3_macro_strokes`
2. `exp_G1_edge_rush`
3. `exp_1_control`
4. `exp_4_zero_tv`
5. `exp_2_zero_id`

### 4.1 结论

`exp_3_macro_strokes` 在 art-FID 上非常值得重视。  
它虽然不是 style 冠军，但在“更接近艺术域分布”这件事上表现稳定。

## 5. 一个临时的“均衡性”启发式榜单

我额外做了一个仅供资料库内部使用的启发式分数：

`balance_score = style - 0.2 * lpips - 0.0005 * fid`

注意：

- 这不是正式指标
- 只是为了快速筛出“style 不低，同时 LPIPS/FID 也没太坏”的实验

按这个启发式排前列的包括：

1. `G0-Base-Gain0.5`
2. `exp_G1_edge_rush`
3. `ablate_A2_p11_id045_tv005`
4. `ablate_A1_p7_id045_tv005`
5. `master_sweep_01_cap_64`
6. `master_sweep_06_patch_std`
7. `master_sweep_14_narrow_micro`

## 6. 这个榜单说明了什么

### 6.1 A 系列和 Master Sweep 的价值被低估了

如果只看单项 style，它们不算最亮眼。  
但如果看均衡性，它们非常靠前。

### 6.2 `exp_G1_edge_rush` 是非常强的均衡候选

它同时具备：

- 较高 style
- 很强 FID
- 很强 art-FID

这类实验在正式历史总结里应该被重点标记。

### 6.3 `G0-Base-Gain0.5` 很像“意外强者”

它不是最常被提到的目录，但在平衡角度非常能打。

## 7. 后续怎么用

正式文档里可以把实验分成三类：

1. style 冠军型
2. 分布最优型
3. 均衡候选型

而不是只给一张总排行榜。

