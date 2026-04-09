# Loss 演化专档

## 1. 总览

项目的 loss 演化基本经历了五代：

1. Gram / Moment / Code / Cycle 组合
2. SWD 上位
3. NCE 作为内容保持补强
4. Color / Brightness 约束系统化
5. 简化成 SWD + Identity + Color 为主、其余为辅助

## 2. 第一代：Gram / Moment / Code / Cycle

代表版本：`ae596d1` 及更早  
代表实现：早期 `Cycle-NCE/src/losses.py`

### 2.1 `calc_moment_loss`

实现：

- 对每通道求 mean / std
- 用 MSE 对齐

作用：

- 把风格当作通道一级统计量

问题：

- 太粗
- 对局部纹理和笔触无能为力

### 2.2 `calc_gram_loss`

实现：

- 展平空间
- 做通道相关性矩阵
- MSE 对齐 Gram matrix

作用：

- 继承经典 style transfer 的二阶统计思路

历史结论：

- 在这个项目里效果有限
- 后面被 SWD 路线压制

### 2.3 `loss_code`

实现：

- `encode_style(pred)` 对齐 `encode_style(target_style)`

作用：

- 强行让输出落到目标风格编码空间

优点：

- 给风格一个显式判别锚点

缺点：

- 容易与别的风格约束重复
- 依赖 style encoder 的质量

### 2.4 `loss_cycle`

实现：

- 把 `pred` 再映回 content style
- 对 content 做低频 L1 重建

作用：

- 防止模型把内容结构彻底冲掉

问题：

- 容易鼓励保守解

## 3. 第二代：SWD 上位

代表提交：`1f818cc`, `adb274a`

### 3.1 为什么 SWD 上位

已有结论非常明确：

- Domain > Instance
- 1x1 / 3x3 / 平滑变体各有对照
- 在 5 风格联合任务上，SWD 给出的可分性结论强于 Gram

### 3.2 `calc_swd_loss`

当前和中后期实现大体思路一致：

- 对 patch 做随机投影
- 比较投影后分布
- 支持多 patch size
- 支持 chunk 化投影
- 支持排序版或 soft-CDF 版距离

### 3.3 为什么它比 Gram 更适合这里

推断：

- Gram 只看二阶相关
- SWD 更接近直接比较局部分布形状
- 对纹理、笔触、局部统计的表达更自然

这与项目“latent 局部纹理迁移”的目标更一致。

## 4. 第三代：NCE 作为内容结构补强

代表提交：`4992e06`

### 4.1 `calc_patch_nce_loss` / 早期 `calc_nce_loss`

核心思路：

- 从 source / target feature 中采 patch 或 token
- 做 InfoNCE
- 保持内容结构一致性

### 4.2 历史定位

NCE 在这个项目里更像：

- 内容结构保护器
- 防止纯风格 loss 把映射推向不可控漂移

不是主风格损失。

### 4.3 为什么没有成为最后唯一主线

推断：

- NCE 对“保持内容”有帮助
- 但对“真正把风格纹理学出来”不够直接
- 所以后来它更多作为辅助或被阶段性裁剪

## 5. 第四代：Color / Brightness 系统化

代表提交：`ed52ecd`, `ed596c0`, `c8577e0`

### 5.1 早期 color loss 的问题

历史信息直接写了：

- color loss 有大问题

这通常意味着：

- 颜色统计和视觉效果不一致
- latent 通道上的直接约束不可靠
- 亮度、对比度、色相可能被混淆

### 5.2 伪 RGB / 低频统计方案

后来版本里的 `calc_spatial_agnostic_color_loss` 体现了比较成熟的方案：

- 先用固定矩阵把 4 通道 latent 映射成伪 RGB
- 再转到 YUV 或类似亮度/色度空间
- 对均值和方差做约束
- 聚焦低频颜色一致性，而不是逐像素硬对齐

### 5.3 亮度约束

`c8577e0` 把亮度约束和 cross-attn 放在一起，非常合理：

- attention 会增强改写能力
- brightness constraint 防止整体曝光漂移

## 6. 第五代：简化后的核心组合

当前 `Cycle-NCE/src/losses.py` 体现的是一套明显收敛后的思路：

- `w_swd_unified`
- `w_swd_micro`
- `w_swd_macro`
- `w_identity`
- `w_repulsive`
- `w_color`
- `w_aux_delta_variance`

对应计算项有：

- `calc_swd_loss`
- 高频版本 / fused HF feature
- `calc_spatial_agnostic_color_loss`
- masked identity
- repulsive loss

### 6.1 这说明什么

最后留下来的主要是三类损失：

1. 风格分布匹配：SWD
2. 内容 / 保真约束：identity
3. 色彩稳定：color / brightness

其它大量早期损失，大多不是完全错误，而是被证明：

- 不够稳定
- 不够直接
- 或者在最终系统里边际收益有限

## 7. 当前 loss 文件的工程特征

当前实现还有几个很明显的工程化特征：

- projection bank 缓存
- CDF / tree 两类距离模式
- 高频 Sobel 融合
- batch 级 SWD 采样
- NVTX / profiler 友好结构

这说明 loss 在后期不只是“理论尝试”，而是重度性能优化对象。

## 8. 总结

如果只保留一句话：

- 这个项目的 loss 历史，本质上是从“经典风格统计 + 多辅助项”收敛到“SWD 做风格，identity 做保真，color 做稳定”的过程。

