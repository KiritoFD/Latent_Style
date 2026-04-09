# loss_aux_delta_variance.md

> 日期: 2026-04-09
> 主题: `w_aux_delta_variance` 这个辅助项在惩罚什么
> 直接证据:
> - `src/losses.py`
> - `src/model.py`
> - `src/trainer.py`
> - `config_07_aux_loss_weak.json`
> - `config_08_aux_loss_strong.json`

---

## 1. 它不是传统主损失，而是辅助约束

`w_aux_delta_variance` 不是像 SWD、identity、color 那样长期存在的主损失项。

从当前代码和 `Gate` 系列配置来看，它更像一种实验型辅助约束:

- 用来约束模型最近一次预测出的 `delta`
- 目的不是直接对齐风格或内容
- 而是调节 `delta` 自身的空间统计特性

这也是为什么它主要出现在:

- `07_aux_loss_weak`
- `08_aux_loss_strong`

这样的配置名里。

---

## 2. 这个 loss 从哪来

在 `src/model.py` 里，`LatentAdaCUT.forward()` 的结尾会做:

```python
delta = self._predict_delta_from_context(...)
self.last_delta = delta
anchor = self._perturb_anchor_if_needed(x)
pred = anchor + delta * float(step_size) * step_scale
```

也就是说，模型会把最近一次前向预测出的 `delta` 存在 `self.last_delta`。

然后在 `src/losses.py` 的 `AdaCUTObjective.compute()` 中:

```python
last_delta = getattr(model, "last_delta", None)
if self.w_aux_delta_variance > 0.0 and torch.is_tensor(last_delta):
    delta_variance = torch.var(last_delta.float(), dim=(2, 3), unbiased=False).mean()
    l_aux = pred.new_tensor(self.w_aux_delta_variance) * (1.0 / (delta_variance + 1e-6))
    total = total + l_aux
    metrics["aux_delta"] = l_aux.detach()
    metrics["_aux_delta_raw"] = delta_variance.detach()
```

这就是它的全部来源链。

---

## 3. 数学上它到底在做什么

这个辅助项先计算:

- `delta_variance = var(last_delta, spatial_dims).mean()`

也就是:

- 先对 `last_delta` 在空间维度 `(H, W)` 上算方差
- 再对 batch / channel 结果求平均

然后真正加到总损失里的是:

- `w_aux_delta_variance * (1 / (delta_variance + 1e-6))`

这意味着什么？

意味着这个 loss **不喜欢方差太小**。

因为:

- 当 `delta_variance` 很小的时候，`1 / variance` 会很大，惩罚会变强
- 当 `delta_variance` 足够大时，这个项就会变小

所以它本质上是在鼓励:

- `delta` 不要过于平
- `delta` 在空间上要有一定变化量

---

## 4. 为什么会想加这个项

把这条辅助项放回 `Gate` 语境里看，就很好理解了。

`Gate` 在研究的是:

- attention 生成出来的 `delta` 应该如何写回主干

这时候一个很自然的担心是:

- gate 机制可能把 `delta` 压得太弱、太平
- 最后模型虽然“更稳定”，但也变得没什么有效风格注入

所以 `w_aux_delta_variance` 的直觉可以理解成:

- 如果 `delta` 的空间变化量太低，就给它一点惩罚
- 逼模型不要把 style delta 学成一张过于平庸、近似常数的图

换句话说，它不是直接奖励 style，而是在防止“attention delta 没有生命力”。

---

## 5. 它和别的 loss 有什么不同

### 5.1 它不同于 SWD

SWD 在对齐的是:

- 生成结果和目标风格在 patch 投影分布上的差异

它是结果层面的风格统计对齐。

### 5.2 它不同于 color loss

color loss 在对齐的是:

- 颜色均值、对比度、色调、饱和度

它关心的是颜色统计。

### 5.3 它不同于 identity / topology

identity / topology 关心的是:

- 内容结构别跑飞

### 5.4 `aux_delta_variance` 关心的是内部增量本身

它根本不直接比较 `pred` 和 `target_style`。

它只看:

- 模型内部刚刚预测出的 `delta` 有没有足够的空间变化

所以它是一个非常典型的“内部机制约束”，不是输出对齐损失。

---

## 6. 为什么记录名叫 `aux_delta`

在训练器里，这个量会被记成:

- `metrics["aux_delta"]`

并且进入 `trainer.py` 的日志列与 tqdm 展示。

这说明作者有意把它当成一个单独可观测的辅助机制，而不是把它悄悄混进总 loss 里。

这很重要，因为实验时要回答的不只是:

- 总 loss 有没有降

还包括:

- 这个辅助项是不是一直很大
- 它到底有没有真的改变 `delta` 的统计性质

---

## 7. 弱版和强版在测什么

`Gate` 系列中:

- `config_07_aux_loss_weak.json` 使用 `w_aux_delta_variance = 0.1`
- `config_08_aux_loss_strong.json` 使用 `w_aux_delta_variance = 1.0`

这两个配置在问的问题很直接:

- 如果只轻轻鼓励 `delta` 有空间变化，会不会已经足够
- 如果把这个约束开得更强，是否会让风格注入更有力，还是反而变得噪声更大、更不稳

也就是说，这不是“有无”实验，而是“这个内部约束应该多强”的标定实验。

---

## 8. 这个写法有什么潜在风险

虽然这个辅助项很聪明，但也要注意它的天然风险。

### 8.1 它只奖励方差，不奖励正确方向

它会鼓励 `delta` 有变化，但不会保证这种变化一定是“有用的风格变化”。

所以如果单独使用得太强，它可能鼓励的是:

- 更有纹理
- 更有起伏
- 但不一定更正确

### 8.2 它可能和稳定性目标冲突

如果门控本来就是为了让写回更稳、更克制，那么这个项又在鼓励 `delta` 更有空间起伏，二者就可能形成拉扯。

### 8.3 它更适合作为配角

从代码位置和命名看，它本来也不是被当主损失设计的。

更合理的理解是:

- SWD / identity / color 决定大方向
- `aux_delta_variance` 负责防止内部增量退化得太平

---

## 9. 它解决的到底是什么问题

它解决的不是“结果不像风格图”，而是另一个更内部的问题:

- 模型可能学会一个过于保守、空间变化过低的 `delta`
- 这种 `delta` 让训练看起来稳定，但风格注入不够有力

所以它要解决的是:

- **内部风格增量的塌平问题**

这是非常 04 月这段风格实验才会出现的问题，因为只有当模型已经开始显式建模 `delta`、attention、gate 时，才会需要对内部增量本身加约束。

---

## 10. 当前可下的理解

### 理解 1

`w_aux_delta_variance` 是一个鼓励 `last_delta` 保持空间变化量的辅助项。

### 理解 2

它惩罚的不是“风格不准”，而是“delta 太平”。

### 理解 3

它最适合放在 `Gate` 这样的内部机制实验里，因为那时研究重点已经转向“风格增量怎么产生、怎么写回”。

### 理解 4

如果后续看到 `aux_loss_weak / aux_loss_strong` 的结果差异，这份文档就是理解它们的必要前置材料。

