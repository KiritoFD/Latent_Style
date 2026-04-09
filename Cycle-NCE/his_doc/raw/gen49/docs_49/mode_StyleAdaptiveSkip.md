# mode_StyleAdaptiveSkip.md

> 首次明确出现: 2026-03-26 `61fb4578`  
> 历史位置: `StyleRoutingSkip` 之前的一代 skip 风格过滤模块  
> 当前状态: 已被后续 skip 版本替代，但其思想直接延续到了 4 月的 skip 战线

---

## 1. 模块定位

`StyleAdaptiveSkip` 是项目里非常关键的一代 skip 模块。  
它的目标不是简单地“把 encoder 的浅层特征接回 decoder”，而是先回答一个更难的问题：

**skip 里到底哪些内容应该保留，哪些高频泄漏应该压掉？**

这就是它和普通 U-Net skip 的根本区别。

---

## 2. 代码结构

核心实现非常直白：

```python
self.gate_mapper = nn.Sequential(
    nn.Linear(style_dim, channels),
    nn.Sigmoid(),
)
self.rewrite_mapper = nn.Linear(style_dim, channels)
```

然后在 `forward()` 里：

1. 用 `gate_mapper` 预测每个通道的保留/抑制系数
2. 用 `rewrite_mapper` 预测替代偏置
3. 根据 style 决定 skip_feat 保留多少、重写多少

最终形式：

```python
return skip_feat * effective_gate + rewrite_bias * (1.0 - effective_gate)
```

所以它本质上是在做一件事：

**让 skip 不再是被动拷贝，而是主动风格筛选。**

---

## 3. 这个模块为什么会出现

从 3 月下旬到 4 月上旬的实验主题看，项目已经意识到一个核心矛盾：

- skip 能强力保内容和结构
- 但 skip 也可能把源图的高频噪声、错误纹理、颜色残留原封不动带回去

这会造成几类问题：

1. 风格被 skip 冲淡
2. 结构虽然保住了，但画面脏
3. decoder 明明做了风格生成，却被 skip 抢权

`StyleAdaptiveSkip` 的设计，就是为了把这个矛盾显式化。

---

## 4. 它和 4 月上旬实验的直接关系

虽然当前版本已经换成了 `StyleRoutingSkip`，但 4 月那一串实验名字几乎都能看出它的思想延续：

- `dirty_skip`
- `no_skip`
- `restore_skip_shortcut`
- `macro_no_skip`
- `skip clean`
- `skip blur`

这些问题其实都在围绕 `StyleAdaptiveSkip` 这类模块打转：

- skip 要不要过滤
- 过滤得多强
- 过滤是固定规则还是由 style 驱动
- 如果完全关掉 skip 会怎样

所以它虽然不是最新类名，但它定义了 4 月研究问题的坐标系。

---

## 5. 两个关键设计点

### 5.1 content retention boost

```python
self.content_retention_boost
```

这个参数非常有意思，因为它说明项目并不想把 skip 纯粹变成风格通道，而是想在“保内容”和“去泄漏”之间找一个中间点。

### 5.2 naive skip ablation

```python
self.ablation_naive_skip = False
self.ablation_naive_skip_gain = 1.5
```

这说明作者非常在意一个问题：

如果把一切精细过滤都拿掉，直接把 skip 粗暴放大，会发生什么？

这就是非常典型的研究型代码思路。  
模块不是只负责跑通，还要天然带着“如何反证自己”的开关。

---

## 6. 它为什么后来被替换

后续代码里出现了 `StyleRoutingSkip`，说明项目继续往前走了两步：

1. 从“自适应过滤 skip”进一步走向“skip 路由”
2. 把 skip 与 decoder / style modulator / gate 的耦合做得更细

但这不是否定 `StyleAdaptiveSkip`，而是因为研究问题更具体了：

- 不只是过滤
- 还要决定 skip 怎么融合、在哪一级融合、是否走不同分支

---

## 7. 在模型史中的结论

`StyleAdaptiveSkip` 的意义不在于它是否是最终答案，而在于它第一次把下面这件事写成了明确的模块：

**skip 不是天然无害的，它本身就是风格迁移中的主要矛盾源之一。**

4 月所有围绕 skip 的实验，几乎都可以看作是在继续回答它提出的问题。  
因此它必须单独立档，不能只在总报告里一笔带过。

