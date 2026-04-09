# mode_GlobalDemodulatedAdaMixGN.md

> 典型出现阶段: 2026-03-26 附近  
> 当前角色: 向后兼容别名类  
> 重要性: 它本身不是新结构，但它记录了模型命名和 checkpoint 兼容链

---

## 1. 这不是“无关紧要的小别名”

`GlobalDemodulatedAdaMixGN` 在当前代码里看起来只是：

```python
class GlobalDemodulatedAdaMixGN(TextureDictAdaGN):
    def __init__(...):
        super().__init__(...)
```

如果只看运行逻辑，确实它没有引入新算子。  
但从考古角度，这个类很重要，因为它说明：

1. 模块曾经有过另一套命名体系。
2. 老 checkpoint、老 config、老代码路径都在用旧名字。
3. 项目在架构迭代时，已经开始认真处理历史兼容问题。

---

## 2. 它暴露出的历史信息

当前代码紧接着写了：

```python
SpatiallyAdaptiveAdaMixGN = GlobalDemodulatedAdaMixGN
SpatiallyAdaptiveAdaGN = GlobalDemodulatedAdaMixGN
CoordSPADE = TextureDictAdaGN
```

这说明至少有几条旧命名支线曾经真实存在过：

- `GlobalDemodulatedAdaMixGN`
- `SpatiallyAdaptiveAdaMixGN`
- `SpatiallyAdaptiveAdaGN`
- `CoordSPADE`

这些名字本身就反映了作者曾经如何理解这个模块：

- 有时强调“global demodulation”
- 有时强调“spatially adaptive”
- 有时把它类比为 `SPADE`

也就是说，模块语义在历史上是逐渐收敛到 `TextureDictAdaGN` 的。

---

## 3. 为什么最终收敛到 `TextureDictAdaGN`

旧名字虽然各有强调点，但都不够完整：

- `GlobalDemodulatedAdaMixGN`
  - 强调了 demodulation / AdaMix / GN
  - 但没有直接点出“纹理字典”这一最关键的设计意图
- `SpatiallyAdaptiveAdaGN`
  - 强调了空间自适应
  - 但没体现低秩字典读写
- `CoordSPADE`
  - 强调坐标与空间调制
  - 但又会把人带向标准 SPADE 语境

而 `TextureDictAdaGN` 这个名字最准确地概括了设计核心：

- texture
- dictionary
- AdaGN

---

## 4. 它在实验史中的作用

这个类的存在，意味着后续实验没有因为重命名而断档。

否则会出现两个严重问题：

1. 旧配置文件里指定的模块类型失效
2. 旧 checkpoint 反序列化后找不到类定义

对一个高频改结构的项目来说，这类兼容层非常关键。  
否则许多历史实验就无法被重新加载、重新验证、重新比较。

---

## 5. 结论

`GlobalDemodulatedAdaMixGN` 不是性能模块，而是历史接口模块。

它告诉我们两件事：

1. 这条模型线对“空间风格调制器”的理解经历过多次命名重构。
2. 到 3 月 26 日前后，团队已经开始认真维护向后兼容，而不是只顾着往前试新结构。

如果说 `TextureDictAdaGN` 代表结构收敛，  
那么 `GlobalDemodulatedAdaMixGN` 代表的是**命名与 checkpoint 兼容层的收敛**。

