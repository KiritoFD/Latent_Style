# mode_attn_gate_mode.md

> 日期: 2026-04-09
> 主题: `attn_gate_mode` 在当前模型里到底控制什么
> 直接证据:
> - `src/model.py` 中 `SemanticCrossAttn`
> - 当前 `src/ablate.py`
> - `config_04_attn_gate_fixed.json`
> - `config_05_attn_gate_learned.json`
> - `config_09/10/11/12` 等 Gate 系列 config

---

## 1. 这不是独立类，但它已经是结构级变量

`attn_gate_mode` 不是一个单独的 Python class。

但在 2026-04-09 这个时点，它已经不只是“某个小配置项”，而是会实打实改变 `SemanticCrossAttn` 输出如何回写到主干特征里的结构级变量。

也正因为如此，虽然它不是独立模块类，我还是把它单独立档。

否则后面看 `Gate` 系列时，很容易只记住“fixed / learned / none” 这几个名字，却说不清它们到底在改什么。

---

## 2. 它控制的是哪一步

`SemanticCrossAttn.forward()` 的大体流程是:

1. 用内容特征 `x_c` 和 style map 做 cross-attention
2. 得到 attention 加权后的 `painted`
3. 再做一次平滑，得到 `painted_smoothed`
4. 通过 `proj_out` 得到一个待注入增量 `delta`
5. 最后把 `delta` 混回原始特征 `x_c`

`attn_gate_mode` 控制的就是第 5 步。

所以它不是在控制:

- attention 要不要算
- q/k/v 要不要生成

它控制的是:

- **attention 生成出来以后，应该以什么方式影响主干特征**

---

## 3. 当前实现是什么

在 `SemanticCrossAttn` 里，相关逻辑是:

```python
delta = final_gate * self.proj_out(painted_smoothed)
gate_mode = str(getattr(self, "attn_gate_mode", "none")).strip().lower()
if gate_mode == "learned":
    mix = torch.sigmoid(self.gate_conv(x_c))
    return x_c * (1.0 - mix) + delta * mix
if gate_mode == "fixed":
    return x_c * 0.5 + delta * 0.5
return x_c + delta
```

这里有三种模式:

- `none`
- `fixed`
- `learned`

---

## 4. 三种模式分别在干什么

### 4.1 `none`

行为:

```python
return x_c + delta
```

含义:

- attention 产物被当作标准 residual 增量
- 原特征保留完整，再额外叠加 style delta

为什么这样写:

- 最简单
- 最接近常规 residual 注入
- 不额外引入新的门控参数

它的优点是朴素直接，缺点是：

- 如果 `delta` 太强，内容会被硬推走
- 如果 `delta` 分布不稳，主干会直接吃到这种不稳

### 4.2 `fixed`

行为:

```python
return x_c * 0.5 + delta * 0.5
```

含义:

- 把原特征和 attention delta 强行五五开

为什么会出现这种设计:

- 当开发者怀疑 “直接 residual 相加太猛或太不受控” 时，一个很自然的第一步就是硬编码比例
- 它比 learned gate 更容易分析，因为没有额外学习自由度

它更像一种工程探针:

- 先看看“硬门控”本身有没有方向性收益
- 如果有，再决定要不要继续上学习式门控

### 4.3 `learned`

行为:

```python
mix = torch.sigmoid(self.gate_conv(x_c))
return x_c * (1.0 - mix) + delta * mix
```

含义:

- 不是全局一个系数
- 而是由 `gate_conv(x_c)` 预测出逐位置、逐通道的混合比例

为什么这样写:

- 因为风格注入不太可能在所有空间位置都应该同样强
- 有的区域该保内容
- 有的区域该更大胆地吃 style delta

这套写法的直觉是:

- 内容主干自己最知道哪里该守、哪里可以放

---

## 5. `gate_conv` 为什么初始化成 0

在 `SemanticCrossAttn.__init__()` 里:

- `gate_conv.weight` 被初始化为 0
- `gate_conv.bias` 也被初始化为 0

这意味着刚开始训练时:

- `sigmoid(0) = 0.5`

也就是说，`learned` 模式在训练初期并不是乱门控，而是先退化成接近 `fixed` 的 0.5 混合。

这个设计非常合理，因为它避免了两种常见坏情况:

1. 一开始 gate 全开，attention delta 直接把主干冲坏
2. 一开始 gate 全关，学习不到风格注入路径

它本质上是在说:

- 先给你一个中性的中点
- 再让训练慢慢把门控往更适合的方向推

---

## 6. 它解决的历史问题是什么

`attn_gate_mode` 出现，本质上是在回应一个越来越具体的问题:

- 不是 “attention 有没有帮助”
- 而是 “attention 明明能生成风格信息，但它一旦太直接地写回主干，就可能带来污染、冲突、内容走形”

更早阶段里，项目主要在研究:

- skip 怎么接
- macro / micro patch 怎么配
- decoder 和 skip 谁更该承担风格表达

到了 `Gate` 这一段，问题又往前推进了一层:

- 如果 attention 已经被证明值得存在，那么它对主干的写入力度本身就值得被实验化

所以 `attn_gate_mode` 解决的不是“有没有 style”，而是“style 往内容主干里灌的时候，如何更可控”。

---

## 7. 为什么它和 `Gate` 系列绑定得这么紧

`Gate` 系列里最核心的几组 config 正是:

- `04_attn_gate_fixed`
- `05_attn_gate_learned`
- `06_gate_learned_idt_energy`
- `09_gate_and_bipolar`
- `10_gate_and_low_color`
- `11_gate_bipolar_low_color`
- `12_gate_energy_bipolar_low_color`

也就是说，`Gate` 不是泛泛地研究很多东西，而是围绕 `attn_gate_mode` 展开了一串组合试验:

- gate 本身有无收益
- gate 和 energy identity 能否配合
- gate 和 bipolar patch 是否互相加强
- gate 和低 color / 高 SWD 是否更匹配

所以 `attn_gate_mode` 可以视为这一整条实验线的核心结构旋钮。

---

## 8. 它和别的开关有什么区别

这一点很容易混淆。

### 8.1 它不同于 `ablation_direct_qk`

`ablation_direct_qk` 改的是:

- q / k 是否还过投影层

这是 attention 内部构造问题。

### 8.2 它不同于 `ablation_raw_v`

`ablation_raw_v` 改的是:

- value 用原始 style_map 还是投影后的 value

这也是 attention 内部构造问题。

### 8.3 它不同于 `ablation_no_smooth`

`ablation_no_smooth` 改的是:

- `painted` 是否还要经过平滑

这属于 attention 输出后处理问题。

### 8.4 `attn_gate_mode` 改的是最后写回主干的方式

也就是说，它是在 attention 链路的最后一跳上做控制。

从职责上说，它更接近“注入控制器”，而不是“attention 算法内部超参”。

---

## 9. 当前可以下的理解

### 理解 1

`none` 是最朴素的 residual 注入基线。

### 理解 2

`fixed` 是一个可分析、可诊断的硬门控探针。

### 理解 3

`learned` 是真正想要的形态，因为它允许不同位置、不同通道有不同的风格注入强度。

### 理解 4

它之所以单独值得成文，是因为它代表了这段时间模型研究重心从“有没有 attention”转向“attention 如何安全、有效地写回主干”。

