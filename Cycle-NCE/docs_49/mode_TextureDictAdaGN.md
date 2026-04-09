# mode_TextureDictAdaGN.md

> 首次明确出现: 2026-03-26 前后作为主风格调制器定型  
> 历史前身: 更早期的 `AdaGN`、空间风格图注入思路  
> 当前角色: 历史上最关键的主力调制器之一

---

## 1. 模块定位

`TextureDictAdaGN` 可以理解为 `AdaGN` 的空间化、字典化、低秩化升级版。

它仍然保留了 `AdaGN` 的核心骨架：

- 先对 feature 做 `GroupNorm`
- 再让 style 控制 scale / shift

但它额外做了三件 `AdaGN` 做不到的事：

1. 用低秩读写矩阵表示纹理字典
2. 用空间注意分配不同位置该读哪个纹理槽
3. 用坐标网格把位置显式送进去

所以它不再只是“全局风格 affine”，而是开始接近“空间纹理路由器”。

---

## 2. 代码结构

核心组成可以拆成四部分。

### 2.1 归一化底座

```python
self.norm = nn.GroupNorm(groups, dim, affine=False)
```

这保证进入风格调制前，特征统计相对稳定。

### 2.2 全局颜色/对比度映射

```python
self.global_proj = nn.Linear(style_dim, dim * 2)
```

这部分保留了 `AdaGN` 传统的 `scale + shift` 机制，用来处理全局层面的颜色、对比度、整体偏向。

### 2.3 低秩纹理字典

```python
self.style_V = nn.Linear(style_dim, self.rank * dim)
self.style_U = nn.Linear(style_dim, dim * self.rank)
```

这相当于把风格调制拆成：

- 一个“读入纹理模式”的低秩表示
- 一个“写回通道空间”的低秩表示

它的关键假设是：

- 风格纹理其实不需要很高维就能描述；
- 真正重要的是如何把少数模式分布到不同空间位置。

### 2.4 空间分配器

```python
self.spatial_attn = nn.Sequential(
    nn.Conv2d(dim + 2, hidden_dim, kernel_size=7, padding=3, bias=False),
    nn.SiLU(inplace=True),
    nn.Conv2d(hidden_dim, self.rank, kernel_size=1, bias=True),
)
```

这里把 `normalized feature + (x,y) 坐标` 拼在一起，输出每个位置对各个 rank 槽位的注意分配。

这一步是 `TextureDictAdaGN` 真正超越早期 `AdaGN` 的地方。

---

## 3. forward 逻辑的真实含义

`forward()` 大致是这样走的：

1. `x -> GroupNorm`
2. `style_code -> global scale/shift`
3. 构造坐标网格
4. 用 `spatial_attn` 预测每个位置该激活哪些纹理槽
5. 用 `style_V`、`style_U` 做低秩纹理读写
6. 把“全局 affine”和“局部纹理混合”叠加起来
7. 通过 `gate` 决定调制强度

从效果上讲，这个模块试图同时负责：

- 大范围风格色调
- 局部笔触分布
- 各空间区域不同的风格激活方式

---

## 4. 它为什么会成为主力模块

因为它恰好踩中了项目当时最需要解决的几个问题：

### 4.1 纯全局调制不够

`AdaGN` 只会全局乘加，难以表达：

- 哪一块应该粗笔触
- 哪一块应该更细
- 哪一块应该带更强局部风格

`TextureDictAdaGN` 通过空间路由补上了这一点。

### 4.2 全 attention 太重、太不稳

后续虽然又尝试了 `CrossAttnAdaGN`，但 3 月底到 4 月初的历史已经反复证明：

- attention 能带来表达力；
- 但也容易让结构、温度、位置编码一起变复杂。

相较之下，`TextureDictAdaGN` 复杂度更可控，也更像一条“稳态主线”。

### 4.3 低秩假设很符合纹理问题

项目的很多后续结论都隐含同一个经验：

- 纹理模式本身维度没那么高
- 真正困难的是如何路由、如何分布、如何和内容结构对齐

`TextureDictAdaGN` 的低秩设计正好契合这一点。

---

## 5. 在模型史中的阶段意义

### 5.1 它是从“风格注入”走向“风格纹理建模”的转折点

有了它之后，模型不再只是问“有没有把 style 打进去”，而开始问：

- 打到了哪里
- 不同位置打的是不是同一种纹理
- 一个 style code 如何展开为空间结构

### 5.2 它是后续 `CrossAttnAdaGN` 的对照基线

`CrossAttnAdaGN` 的 docstring 里都明确写了：

- 它保持旧的 AdaGN API
- 目的是可替换 `TextureDictAdaGN`

也就是说，在 3 月 26 日以后，`TextureDictAdaGN` 已经是主 baseline，而不是试验品。

### 5.3 它也是很多结构消融的受力点

4 月上旬围绕 skip、decoder、color highway、temperature 的实验，虽然不总是直接改这个模块本身，但实际都在改变它与其他子系统的耦合关系。

---

## 6. 与实验现象的关系

为什么很多实验会围绕：

- highpass
- skip clean
- color highway
- patch spectrum

这些看起来分散的点展开？

因为 `TextureDictAdaGN` 虽然能产出空间纹理，但它并不是孤立工作的。  
它的风格输出要经过：

- skip 路由
- decoder 重建
- SWD patch 监督
- color/identity 约束

所以后续大量实验其实是在问：

- 这个模块生成出来的纹理，最终是被谁放大了，还是被谁污染了。

---

## 7. 结论

`TextureDictAdaGN` 是当前这段模型史里最值得被视作“主角模块”之一的东西。

它的重要性不只是因为它复杂，而是因为它第一次把下列三件事揉到了一起：

1. 全局风格 affine
2. 局部空间纹理分配
3. 低秩风格字典读写

如果说 `AdaGN` 是“风格可以进模型”，那 `TextureDictAdaGN` 就是“风格终于开始像纹理那样在模型里展开”。

