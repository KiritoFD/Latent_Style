# exp_Gate.md

> 日期: 2026-04-09
> 主题: 当前工作树里的 `Gate` 系列计划矩阵
> 直接证据:
> - 当前 `src/ablate.py`
> - `src/Gate.bat`
> - `src/Layer-Norm.json`
> - `src/config_01_baseline.json` 到 `src/config_12_gate_energy_bipolar_low_color.json`

---

## 1. 先说结论

`Gate` 目前在当前工作树里更像一条“已经编排完成、但结果尚未完整落回仓库”的实验线。

也就是说:

- 设计矩阵很完整
- config 已经全部生成
- 运行脚本也写好了
- 但根目录 `Gate/` 聚合结果目录目前还没看到

所以这份文档的任务不是报结果，而是把这条线到底想测什么、每个 config 在问什么、它和前面 `45/46/chess` 的关系写清楚。

---

## 2. 它在研究什么

`Gate` 这条线的研究问题非常集中:

1. `SemanticCrossAttn` 的输出应该直接 residual 相加，还是要门控混合
2. 门控是固定比例好，还是让网络自己学好
3. 在开门控后，是否还需要同时改 identity loss 形态
4. SWD patch 组合和 color loss 是否要一起重调

这和 `45/46` 的差别很明显:

- `45/46` 主要在问结构职责怎么分
- `Gate` 主要在问“已经有这套职责分工以后，attention 注入到底怎么控强度”

---

## 3. 基线来源

当前 `src/ablate.py` 指定:

- `SERIES_NAME = "Gate"`
- `DEFAULT_BASE_CONFIG = "Layer-Norm.json"`

所以整条 `Gate` 线都不是从空白 config 起步，而是从 `Layer-Norm.json` 这个基线分叉。

`Layer-Norm.json` 的核心公共骨架:

- `body_block_type="global_attn"`
- `decoder_block_type="conv"`
- `skip_routing_mode="none"`
- `color_highway_gain=0.3`
- `attn_gate_mode` 默认为 `none`
- `w_swd_unified=40.0`
- `w_repulsive=15.0`
- `w_color=50.0`
- `w_identity=5`
- `swd_patch_sizes=[3,5,15,19]`

而 `ablate.py` 又强制把训练侧覆盖为:

- `batch_size=256`
- `num_epochs=120`
- `save_interval=30`
- `full_eval_interval=30`
- `full_eval_on_last_epoch=true`
- `warmup_ratio=0.1`
- `learning_rate=8e-5`

这说明 `Gate` 比 `45` 更像认真跑长线的矩阵。

---

## 4. `attn_gate_mode` 到底在模型里做什么

当前 `src/model.py` 中，`SemanticCrossAttn.forward()` 会先得到:

- `painted`
- `painted_smoothed`
- `delta = gate * proj_out(painted_smoothed)`

然后根据 `attn_gate_mode` 决定怎么把 `delta` 混回原特征 `x_c`:

- `none`: `x_c + delta`
- `fixed`: `x_c * 0.5 + delta * 0.5`
- `learned`: `mix = sigmoid(gate_conv(x_c))`，再做逐位置混合

所以 `Gate` 线研究的不是“有没有 attention”，而是“attention 产物该以什么强度、什么形式进入主干”。

---

## 5. 配置矩阵总表

| 编号 | 配置名 | 主要问题 | `attn_gate_mode` | identity | color | SWD | patch |
|------|--------|----------|------------------|----------|-------|-----|-------|
| 01 | `baseline` | Layer-Norm 基线 | `none` | 5 | 50 | 40 | `[3,5,15,19]` |
| 02 | `weak_decoder` | 少一个 decoder block 会怎样 | `none` | 5 | 50 | 40 | `[3,5,15,19]` |
| 03 | `restore_skip_shortcut` | 恢复 skip 路由是否更稳 | `none` | 5 | 50 | 40 | `[3,5,15,19]` |
| 04 | `attn_gate_fixed` | 固定 0.5/0.5 混合是否优于直接相加 | `fixed` | 5 | 50 | 40 | `[3,5,15,19]` |
| 05 | `attn_gate_learned` | 学习式门控是否优于固定门控 | `learned` | 5 | 50 | 40 | `[3,5,15,19]` |
| 06 | `gate_learned_idt_energy` | 学习门控 + 更强 identity 约束 | `learned` | 200, `idt_mode=energy` | 50 | 40 | `[3,5,15,19]` |
| 07 | `aux_loss_weak` | 弱辅助 delta variance loss 是否有用 | `none` | 5 | 50 | 40 | `[3,5,15,19]` |
| 08 | `aux_loss_strong` | 强辅助 delta variance loss 是否有用 | `none` | 5 | 50 | 40 | `[3,5,15,19]` |
| 09 | `gate_and_bipolar` | 学习门控 + 双极 patch 组合 | `learned` | 5 | 50 | 40 | `[3,25]` |
| 10 | `gate_and_low_color` | 学习门控 + 低 color + 更强 SWD | `learned` | 5 | 10 | 60 | `[3,5,15,19]` |
| 11 | `gate_bipolar_low_color` | 学习门控 + 低 color + bipolar patch | `learned` | 5 | 10 | 60 | `[3,25]` |
| 12 | `gate_energy_bipolar_low_color` | 学习门控 + energy idt + bipolar patch + low color | `learned` | 200, `idt_mode=energy` | 10 | 60 | `[3,25]` |

---

## 6. 可以把它拆成哪几组问题

### 6.1 第一组: 门控本身有没有用

- `01_baseline`
- `04_attn_gate_fixed`
- `05_attn_gate_learned`

它们几乎只改 `attn_gate_mode`，是最干净的核心对照。

### 6.2 第二组: 门控和 identity 约束会不会互相干扰

- `05_attn_gate_learned`
- `06_gate_learned_idt_energy`
- `12_gate_energy_bipolar_low_color`

这组在问:

- 学习门控会不会把风格注入推得过猛
- 如果会，是不是需要 `energy` 版 identity loss 把内容重新拉住

### 6.3 第三组: 门控是不是要和 patch 频带策略一起调

- `05_attn_gate_learned`
- `09_gate_and_bipolar`
- `11_gate_bipolar_low_color`

`[3,25]` 这种 patch 组合很像双极实验:

- 一个很小 patch
- 一个很大 patch

等于是在问门控能否同时处理局部纹理和大尺度风格结构。

### 6.4 第四组: color loss 要不要给 attention 让路

- `05_attn_gate_learned`
- `10_gate_and_low_color`
- `11_gate_bipolar_low_color`
- `12_gate_energy_bipolar_low_color`

这里把 `w_color` 从 `50 -> 10`，同时 `w_swd_unified` 从 `40 -> 60`。

这说明作者在试探:

- attention + SWD 是否已经足够承担风格表达
- color loss 会不会反而把输出拉回保守状态

---

## 7. 和前面系列的关系

### 7.1 和 `45`

`45` 关注的是:

- skip 怎么接
- macro / micro 怎么分工

`Gate` 则继续追问:

- attention 输出到底怎么混回主干

### 7.2 和 `46`

`46` 讨论的是:

- color highway
- dirty skip
- decoder usurpation

`Gate` 则把“谁负责风格”推进成“attention 负责时，注入阀门怎么设计”。

### 7.3 和 `chess`

`chess` 更多在查病理纹理、位置编码、patch 分布。

`Gate` 可以看成在查另一类潜在病理:

- attention delta 注入太硬、太猛、太不稳定，会不会导致输出污染或塌陷

---

## 8. 当前没有结果时，应该怎么记

### 8.1 已经确定的事实

- `Gate` 系列已经在 `ablate.py` 和 `Gate.bat` 中被完整定义
- `config_01` 到 `config_12` 已经真实存在
- 这些 config 的变量改动是清晰可追的

### 8.2 还没有闭环的部分

- 还没有在当前工作树里看到完整 `Gate_*` 实验目录
- 还没有看到统一整理后的 `summary.json` 结果链
- 因此现在还不能对“门控是否赢了”下结论

---

## 9. 现在就能写下来的历史判断

### 判断 1

`Gate` 是一条非常明确的“attention 注入控制”实验线，不是泛泛的结构微调。

### 判断 2

它以 `Layer-Norm.json` 为母体，说明那时主线已经相对稳定，研究重心转向局部机制优化。

### 判断 3

这条线把三个此前分散的问题拧在了一起:

- gate 模式
- identity 形态
- patch / color / SWD 再配平

### 判断 4

即使现在没有完整结果，`Gate` 依然值得单独立档，因为它不是零散 config，而是一条结构清楚的问题链。

