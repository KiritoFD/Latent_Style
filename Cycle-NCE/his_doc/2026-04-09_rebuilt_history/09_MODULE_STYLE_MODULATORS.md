# 模块专档：Style Modulator 家族

## 1. 为什么单独给这一类模块立档

从整个历史看，最稳定的一条技术主线不是具体 backbone，而是：

- 风格一定要通过 style-conditioned modulation 进入网络

这条线先后出现过：

- 简单 `AdaGN`
- `TextureDictAdaGN`
- `CrossAttnAdaGN`

## 2. 第一代：`AdaGN`

代表时期：

- Era A
- 代表提交 `ae596d1`

实现特征：

- `GroupNorm(affine=False)`
- `Linear(style_code -> 2 * dim)`
- scale / shift 仿射调制
- identity init

优点：

- 简单
- 稳定
- 训练初期不容易炸

局限：

- 风格调制能力偏全局
- 难以表达复杂局部纹理

## 3. 第二代：`TextureDictAdaGN`

代表时期：

- Era C
- 代表提交 `c619fda` 附近

实现特征：

- 全局 `scale/shift`
- 低秩 `style_V` / `style_U`
- `spatial_attn`
- 坐标网格
- read/write 式纹理混合

我对它的理解：

- 把 style 当作一组可读写的纹理基底
- 每个空间位置通过 attention 风格化地读取这些基底

这比简单仿射更强，也更贴近“笔触迁移”目标。

## 4. 第三代：`CrossAttnAdaGN`

代表时期：

- Era D
- 代表提交 `c8577e0`

实现特征：

- learnable style tokens
- style-code-conditioned token bias
- positional projection
- `scaled_dot_product_attention`
- FFN 后处理

它解决的问题是：

- 让不同空间位置从不同 style token 中读取信息
- 让风格调制具备更显式的 token-空间匹配结构

## 5. 三代对比

| 模块 | 风格表达方式 | 空间性 | 复杂度 | 历史定位 |
| --- | --- | --- | --- | --- |
| `AdaGN` | 全局仿射 | 弱 | 低 | 起点、基座 |
| `TextureDictAdaGN` | 低秩纹理读写 | 中 | 中 | 主要工作马 |
| `CrossAttnAdaGN` | token 注意力 | 强 | 高 | 后期增强版 |

## 6. 历史结论

### 6.1 AdaGN 思路从未被彻底抛弃

哪怕具体实现变化很大，“归一化后再用 style 条件调制”这件事一直在。

### 6.2 调制器是画风的核心，不只是稳定器

`f7b328c` 明确说明改 AdaGN 会导致笔触明显变化。  
所以 modulator 直接决定风格外观，不是简单训练辅助。

### 6.3 越强的调制器，越需要额外稳定约束

从 TextureDict 到 CrossAttn，模型改写能力越来越强，同时：

- color / brightness 问题更突出
- skip leakage 更危险
- identity shortcut 更容易出现

