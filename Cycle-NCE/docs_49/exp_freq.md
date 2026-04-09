# exp_freq.md

> 生成时间: 2026-04-09  
> 数据来源: `freq/*/config.json` + `full_eval/epoch_0080_tokenized_distill_epochs200/summary.json`  
> 说明: 当前先按实验目录事实整理，尚未把它和某一版 `ablate.py` 提交完全钉死

---

## 1. 这条线在研究什么

`freq` 系列名字已经把研究意图写得很直白：

- `conservative_baseline`
- `brush_frenzy`
- `large_view_awareness`
- `no_idt_abyss`
- `idt_iron_fist`
- `yuv_dictatorship`
- `extreme_asymmetry`

这说明它主要不是结构主干竞争，而是在研究：

1. patch 频谱如何影响风格和内容
2. identity 权重如何改变频域监督的效果
3. color 约束加重以后会不会主导结果

所以它是一条很标准的“频谱/损失关系研究线”。

---

## 2. 实验总表

| 目录 | patch | `w_color` | `w_id` | style | content | clip_dir | p2a_style | p2a_content |
|------|-------|-----------|--------|-------|---------|----------|-----------|-------------|
| `freq_01_conservative_baseline` | `1,3,5,7` | 50 | 5 | 0.6356 | 0.8681 | 0.3347 | 0.5923 | 0.8651 |
| `freq_02_brush_frenzy` | `1,3,5` | 20 | 5 | 0.6338 | 0.8694 | 0.3290 | 0.5901 | 0.8654 |
| `freq_03_large_view_awareness` | `1,3,7,15,25` | 30 | 5 | 0.6347 | 0.8724 | 0.3291 | 0.5881 | 0.8691 |
| `freq_04_no_idt_abyss` | `1,3,5,11` | 40 | 0 | 0.6454 | 0.8751 | 0.3514 | 0.6071 | 0.8522 |
| `freq_05_idt_iron_fist` | `1,3,5,11` | 40 | 25 | 0.6392 | 0.8614 | 0.3479 | 0.5945 | 0.8459 |
| `freq_06_yuv_dictatorship` | `1,3` | 150 | 5 | 0.6324 | 0.8736 | 0.3230 | 0.5828 | 0.8766 |
| `freq_07_remove_blast_wall` | `1,3,5,9` | 60 | 5 | 0.6337 | 0.8675 | 0.3306 | 0.5865 | 0.8671 |
| `freq_08_extreme_asymmetry` | `1,15,25` | 30 | 5 | 0.6366 | 0.8771 | 0.3314 | 0.5860 | 0.8790 |
| `freq_09_lancet` | `1,3,5,11,25` | 40 | 8 | 0.6353 | 0.8707 | 0.3313 | 0.5888 | 0.8636 |

---

## 3. 关键观察

### 3.1 这条线整体偏 content-first

先不看单点，整体范围已经说明问题：

- style 大多在 `0.632 - 0.645`
- content 大多在 `0.861 - 0.877`

这不是冲 style 上限的路线，而是一条明显保守的频谱研究线。

### 3.2 `no_idt_abyss` 很值得重点标记

`freq_04_no_idt_abyss` 是这一批里最值得反复引用的点之一：

- style 最高: `0.6454`
- content 也不低: `0.8751`
- `clip_dir` 也最高之一: `0.3514`

它说明在这条频谱线里，拿掉 IDT 并没有让模型崩掉，反而在多个指标上更好。

### 3.3 `idt_iron_fist` 证明强 IDT 会同时压风格和内容

`freq_05_idt_iron_fist` 把 `w_id` 拉到 `25` 后：

- style 下降
- content 下降
- p2a 也下降

这意味着在这条线里，强 IDT 并没有换来更强的内容保持，反而可能让整体训练变得更僵硬。

### 3.4 `yuv_dictatorship` 说明高 color 权重不是万能解

`freq_06_yuv_dictatorship` 把 `w_color` 提到 `150`：

- style 并没有提高
- content 倒是很高

也就是说，只靠加强颜色约束，并不能自动换来更强的风格表达。

### 3.5 `extreme_asymmetry` 说明不对称 patch 组合有潜力

`freq_08_extreme_asymmetry` 用的是 `1,15,25`：

- style 不突出
- 但 content 和 p2a_content 都是这一批最强区间

这说明“极不对称 patch 组合”至少不是坏主意，它可能更适合作为保内容方案。

---

## 4. 这条线在大历史里的位置

和其它线比较：

- `Aline120` 是 style 上限探针
- `46` 是主结构 holy grail 搜索
- `chess` 是病理排障
- `freq` 是频谱-identity-color 关系研究

所以它不一定会直接产出主线最优结构，但它提供的是一种更底层的规律：

**不同 patch 频谱和 identity/color 权重到底怎样共同塑造结果。**

---

## 5. 当前结论

从当前数据看，`freq` 最重要的几条结论是：

1. 这条线整体偏保内容，不是冲 style 上限。
2. `w_id=0` 的 `freq_04_no_idt_abyss` 非常值得反复引用。
3. 高 color 权重不能替代真正的风格驱动。
4. patch 频谱的组合方式会改变结果，但在这条线上更明显地影响的是“保守程度”，不是绝对 style 上限。

