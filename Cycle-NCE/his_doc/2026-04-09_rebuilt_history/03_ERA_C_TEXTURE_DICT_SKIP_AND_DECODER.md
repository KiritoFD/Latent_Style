# Era C：TextureDict、Skip Routing 与 Decoder 配方

时间范围：2026-02-26 到 2026-03-22  
代表提交：`c619fda`, `4992e06`, `4699637`, `ed52ecd`, `ed596c0`

## 1. 这一阶段的主题

如果说 Era B 解决的是“风格怎么进模型”，那么 Era C 解决的是：

1. 风格进入后，怎样不把内容毁掉。
2. skip 怎样防止把原图高频直接漏回输出。
3. decoder 是不是应该继续做 norm-heavy 调制。
4. color loss 到底应该怎样定义才不会误导模型。

这是工程味最重、也最像“真正产品化调参”的一个阶段。

## 2. 核心结构：`TextureDictAdaGN`

以 `c619fda` 附近版本为代表，`model.py` 的核心调制器已经不是最早的简单 `AdaGN`，而是：

- `TextureDictAdaGN`

其结构包含：

- `GroupNorm`
- 全局 `style_dim -> 2 * dim` 的 scale / shift
- 低秩 `style_V`, `style_U`
- `spatial_attn`
- 坐标网格输入
- 对 feature 做 read/write 式混合

### 2.1 设计意图

这类实现背后的思想可以概括为：

- 不是只做全局颜色/对比度调制
- 而是让 style code 通过低秩“纹理字典”改写空间特征

也就是说，风格被视为一种“可读写纹理基底”，而不是纯通道仿射变换。

### 2.2 为什么重要

这解释了为什么单纯调 loss 还不够。

项目后来很多结论都在说明：

- style 成败高度依赖调制器设计
- 尤其是纹理、笔触、局部风格，一定要通过 block 内部机制显式处理

## 3. `NormFreeModulation` 与 no-norm decoder

`c619fda` 明确写了：

- `modified decoder block to no norm`

对应模块是：

- `NormFreeModulation`

其思想非常清楚：

- decoder 侧不再依赖强归一化
- 避免把局部对比度和细纹理一并抹平
- 只通过 style-conditioned gamma / beta 做轻量调制

这是一个很关键的工程判断：

- 主干里可以强约束
- decoder 里如果约束过重，输出容易发灰、发糊、发雾

## 4. `StyleAdaptiveSkip` 与 skip 过滤

这是这段历史里另一个核心模块。

目标：

- 不是简单保留 skip
- 而是让 skip 成为“可过滤、可改写、可控制泄漏”的路径

从实现看，它包含：

- `gate_mapper`
- `rewrite_mapper`
- `content_retention_boost`

这说明 skip 在项目里被重新定义成：

- 内容保真通道
- 也是 source texture leakage 通道

后面的消融结论和这完全一致。

## 5. 重要消融结论

`DEEP_EXPERIMENT_ANALYSIS_REPORT_20260326.md` 对应这一时代给出了非常强的信号。

### 5.1 `abl_no_adagn`

现象：

- style 不低
- 但内容和分布明显恶化

结论：

- AdaGN 类调制不是锦上添花，而是核心约束器

### 5.2 `abl_naive_skip`

现象：

- style 指标继续上升
- LPIPS / FID / art_fid 全面恶化

结论：

- 未过滤的 skip 会让原图高频和伪纹理一起泄漏
- “style 高”不代表“真正学会了风格迁移”

### 5.3 `abl_no_residual`

现象：

- 数值上更干净
- 但 style 明显下滑

结论：

- 残差不只是优化技巧，也是风格变换强度的重要承载路径

## 6. NCE 在这段时期的角色

`4992e06` 写得很明确：

> 分类准确率有提升，NCE loss 是有效的

这说明 NCE 在这里承担的是“内容结构保真”的职责，而不是主要风格信号。

也可以把它理解成：

- SWD 管风格分布
- NCE 管内容可辨识结构

这也是为什么两者后来会并存一段时间。

## 7. tokenizer / style embedding 的简化

`4699637` 的信息是：

- 单独蒸馏 tokenizer
- 优化 style_embedding
- 指标明显提升

这件事说明项目这时已经意识到：

- 风格编码部分本身也可能是瓶颈
- 不是每个功能都该塞进主模型同步学

从后续 trainer 和 model 简化来看，这一步也推动了系统解耦。

## 8. color loss 的阶段性结论

这段时间围绕 color loss 出现两次很关键的拐点：

- `ed52ecd`：color loss 有大问题，增加实现和消融
- `ed596c0`：通道映射回 RGB 的缩略图 color loss 大赢

所以这一阶段对 color 的认知从：

- “直接在 latent 通道上做统计约束”

变成：

- “先构造伪 RGB / 亮度色彩空间，再做低频统计或直方图约束”

这个思路后来被保留到了更新的 `losses.py` 里。

## 9. 对应实验组

这一阶段主要对应：

- decoder 结构与配方组
- NCE / SWD 路线组
- 空间调制与颜色策略组
- 部分快速扫描与微型回归组

典型实验名包括：

- `decoder-*`
- `Decoder_D*`
- `nce_*`
- `color_*`
- `micro*`

## 10. 阶段总结

Era C 是“把风格做得更像风格，同时不让模型走捷径”的阶段。

这一代留下来的最重要资产不是某个单一冠军实验，而是三套成熟机制：

1. style-conditioned normalization / modulation
2. skip routing / leakage control
3. color loss 的伪 RGB / 亮度一致性思路

