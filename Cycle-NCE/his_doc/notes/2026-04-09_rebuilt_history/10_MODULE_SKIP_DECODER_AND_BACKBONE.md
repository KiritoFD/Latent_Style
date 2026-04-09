# 模块专档：Skip、Decoder 与 Backbone

## 1. Skip 为什么是核心问题

在这个项目里，skip 不是普通 U-Net 的“保真捷径”，而是：

- 内容结构救命通道
- 也是 source 高频泄漏入口

所以后面大量消融都在围绕 skip 做。

## 2. `StyleAdaptiveSkip` / `StyleRoutingSkip`

代表实现位置：

- 当前 `Cycle-NCE/src/model.py`

主要机制：

- `gate_mapper`
- `rewrite_mapper`
- `content_retention_boost`
- 可切到 naive / normalized / adaptive 路由

含义：

- skip 不再直接拼回去
- 而是先判断该保留多少、重写多少、压掉多少

## 3. `NormFreeModulation` 与 decoder

decoder 的主要历史矛盾是：

- 调制太弱，风格纹理不出来
- 调制太重，画面发雾发糊

`NormFreeModulation` 的设计就是在两者中找平衡：

- 不做强归一化
- 只做 style-conditioned gamma/beta
- 保持局部对比度

## 4. 为什么 no-norm decoder 是一个转折

`c619fda` 之所以关键，是因为它让 decoder 从“继续归一化地改”转成“少动统计，多动局部改写”。

这和项目后来对 color / brightness 的重视是一致的：

- decoder 太强地改统计量，会直接把色彩和亮度做坏

## 5. Backbone 从 Conv 到 Attention 到 CGW

当前代码已经支持：

- `conv`
- `global_attn`
- `window_attn`

这说明 backbone 的演化主线是：

1. 卷积骨架
2. 插入全局 attention
3. 插入窗口 attention
4. 组合成 c-g-w 混合骨架

## 6. 对应历史结论

### 6.1 `abl_naive_skip`

说明：

- 不过滤 skip 会让 style 分数升高
- 但 FID / art_fid / LPIPS 同时变差

解释：

- 模型借 source 高频和伪纹理走捷径

### 6.2 `abl_heavy_decoder`

说明：

- 不能轻易下结论说“decoder 更深就更好”

原因：

- 很多 heavy decoder 实验同时混入了 rank、HF-SWD、trainer 变化

### 6.3 `cfdbaba` + `c405b9d`

说明：

- c-g-w 不是只改结构名字，而是把 conv/global/window 三类 block 正式纳入搜索空间

