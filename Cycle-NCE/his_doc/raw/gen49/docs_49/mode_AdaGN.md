# mode_AdaGN.md

> 首次明确出现: 2026-02-08 `38e00fc5`  
> 典型阶段: AdaGN 基线期、StyleMaps 时期、45/46 之前的大多数 U-Net 版本  
> 当前状态: 已被更复杂的 `TextureDictAdaGN`、`CrossAttnAdaGN` 等调制器取代，但它是整条模型线的起点模块

---

## 1. 模块定位

`AdaGN` 是这条模型史里最基础的风格调制单元。  
它解决的问题非常朴素：

- 先用 `GroupNorm` 把 feature 归一化；
- 再用 style code 预测出逐通道 `scale / shift`；
- 最后把风格信号注入到 feature map 中。

从架构史的角度看，它是“全局风格调制”时代的代表。

---

## 2. 代码证据

在 2026-02-11 `c19371e0` 对应的 `model.py` 中，`AdaGN` 的核心形式已经很明确：

```python
class AdaGN(nn.Module):
    def __init__(self, dim: int, style_dim: int, num_groups: int = 8) -> None:
        self.norm = nn.GroupNorm(groups, dim, affine=False, eps=1e-6)
        self.proj = nn.Linear(style_dim, dim * 2)

    def forward(self, x: torch.Tensor, style_code: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        params = self.proj(style_code).unsqueeze(-1).unsqueeze(-1)
        scale, shift = params.chunk(2, dim=1)
        return h * scale + shift
```

这里最重要的点有两个：

1. 它只看一个全局 `style_code`，没有空间分布。
2. 它的初始化刻意接近恒等映射，避免训练一开始就把内容结构冲坏。

---

## 3. 在历史上的角色

### 3.1 它是最早“风格进入模型体内”的标准接口

早期版本里，`ResBlock` 的两层归一化都是 `AdaGN`：

- `norm1 = AdaGN(...)`
- `norm2 = AdaGN(...)`

这意味着风格不是只在输入端或输出端做一次 affine，而是深入残差块内部。

### 3.2 它代表的是“全局风格假设”

`AdaGN` 默认认为：

- 一个 style 向量可以控制整张特征图；
- 空间上的细粒度笔触、局部纹理、区域差异，不需要独立路由。

这在项目前期是合理的，因为先要验证“风格注入本身有没有用”。

### 3.3 它后来不够用了

随着实验推进，项目开始需要：

- 低频和高频分离
- 空间位置敏感的纹理注入
- skip 与 decoder 分工
- 局部颜色/笔触控制

这时单纯的 `AdaGN` 就显得太“全局”、太“平”。

---

## 4. 为什么它后来被替换

后续新模块的出现，本质上都是在补 `AdaGN` 的短板：

| 后续模块 | 相比 `AdaGN` 新增了什么 |
|----------|-------------------------|
| `TextureDictAdaGN` | 引入低秩纹理字典、空间注意分配、位置坐标 |
| `CrossAttnAdaGN` | 用 style tokens 和 cross-attention 取代单向量调制 |
| `NormFreeModulation` | 在 decoder 里尝试摆脱显式归一化的束缚 |

换句话说，后续模块不是在否定 `AdaGN`，而是在给它加上：

- 空间感知
- 多 token 表达
- 更细的风格路由

---

## 5. 它和实验结果的关系

虽然 4 月的高阶实验已经很少直接把模块名写成 `AdaGN`，但大量结论都从它出发：

1. 早期能稳定训练，说明“归一化后再做 style affine”这条路是对的。
2. 后续引入 `TextureDictAdaGN` 时，仍然保留了“先 norm，再调制”的骨架。
3. 连 `CrossAttnAdaGN` 也保留了同样的 API 形态，本质上是在沿用 `AdaGN` 的接口契约。

所以从模型演化史看，`AdaGN` 不是被推翻，而是被“升级封装”。

---

## 6. 输入输出与接口语义

### 输入

- `x`: `[B, C, H, W]`
- `style_code`: `[B, style_dim]`

### 输出

- 与 `x` 同形状的调制后特征图

### 语义

- `scale` 控制每个通道的放大/抑制
- `shift` 控制每个通道的偏移

它不改变空间尺寸，不引入卷积，不直接做混合，只负责“把 style 乘加到归一化特征上”。

---

## 7. 在模型史中的结论

如果只用一句话概括：

`AdaGN` 是这条模型线从“外部风格条件”走向“内部风格调制”的第一块地基。

后面所有更复杂的调制器，无论名字怎么换，都还在继承它留下的三个原则：

1. 风格调制应该发生在 feature 域内部。
2. 调制前最好先把统计量归整到稳定范围。
3. 初始化应该尽量接近恒等映射，先保训练稳定，再逐步放大风格作用。

