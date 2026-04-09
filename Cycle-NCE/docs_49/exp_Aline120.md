# exp_Aline120.md

> 生成时间: 2026-04-09  
> 数据来源: `Aline120/*/config.json` + `full_eval/epoch_0030_tokenized_distill_epochs200/summary.json`  
> 关联提交: `0928420200`、`6b34f2209`

---

## 1. 这条线在研究什么

`Aline120` 不是一条保守的均衡线，而是一条明显偏“高风格冲击”的并行支线。  
它的配置特征非常统一：

- `w_identity` 很低，甚至为 0
- `w_swd` 很高
- `w_color` 极高
- patch 组合偏向直接推纹理
- `residual_gain` 被明确当作主变量

从结果看，这条线更像是在问：

1. 如果几乎不顾内容锚点，style 上限能冲到哪。
2. naive skip 和残差恢复会不会在极端配置下反而有帮助。
3. patch 频谱与 residual_gain 的组合，能不能直接塑造一条更“暴力”的风格路径。

---

## 2. 实验总表

| 目录 | 配置摘要 | style | content | clip_dir | p2a_style | p2a_content |
|------|----------|-------|---------|----------|-----------|-------------|
| `Aline120_aline_01_oracle` | `res=2.0`, `no_res=True`, `w_id=0`, `w_swd=250`, `w_color=150`, patch=`1,3,5` | 0.6579 | 0.5474 | 0.5834 | 0.6495 | 0.5572 |
| `Aline120_aline_02_texture_maniac` | `res=2.5`, `no_res=True`, `w_id=0`, `w_swd=300`, `w_color=150`, patch=`3,5` | 0.6730 | 0.5611 | 0.5960 | 0.6637 | 0.5302 |
| `Aline120_aline_03_ghost_wireframe` | `res=1.5`, `no_res=False`, `naive_skip=True`, `w_id=0`, `w_swd=250`, `w_color=150`, patch=`3,5,7` | 0.7114 | 0.6364 | 0.6067 | 0.7090 | 0.6279 |
| `Aline120_aline_04_macro_trap` | `res=1.5`, `no_res=True`, `w_id=0`, `w_swd=200`, `w_color=150`, patch=`5,7,9` | 0.6571 | 0.5437 | 0.5848 | 0.6536 | 0.5413 |
| `Aline120_aline_05_idt_poison` | `res=1.5`, `no_res=True`, `w_id=20`, `w_swd=150`, `w_color=150`, patch=`1,3,5` | 0.6782 | 0.5626 | 0.6002 | 0.6738 | 0.5310 |

---

## 3. 最值得记录的现象

### 3.1 `ghost_wireframe` 是这条线的真正强点

`Aline120_aline_03_ghost_wireframe` 明显高于其它同线实验：

- `style = 0.7114`
- `content = 0.6364`
- `p2a_style = 0.7090`

这说明两件事：

1. 即便在极端高 style 配置下，`naive_skip` 也不一定只会带来负收益。
2. 完全去 residual 并不总是最优；在这条线里，保留 residual 反而更有助于把风格打上去同时不至于让内容彻底崩掉。

### 3.2 `macro_trap` 这个名字是有道理的

`Aline120_aline_04_macro_trap` 的 patch 更偏宏观，但结果没有变强：

- style 没比 `oracle` 或 `texture_maniac` 好
- content 仍然很低

也就是说，在 Aline 这条极端线里，偏宏观 patch 不是决定性的捷径。

### 3.3 强 IDT 在这里没有完全杀死 style

`Aline120_aline_05_idt_poison` 虽然名字叫 `idt_poison`，但它的 style 仍然有 `0.6782`。  
这表明 Aline 支线的 style 驱动力很强，强到即使把 `w_identity` 拉到 `20`，也没有像某些主线实验那样被明显压垮。

---

## 4. 这条线在大历史里的位置

如果把它和 `Ablate43`、`46`、`freq` 比：

- `Ablate43` 更像结构均衡性实验
- `46` 更像 holy grail 周边结构定位
- `freq` 更像频谱/identity 的保守研究
- `Aline120` 更像极端风格推进器

所以这条线很适合作为“上限探针”，不太像最终主线解。

---

## 5. 当前结论

`Aline120` 最重要的历史价值，不是它是否成为最终模型，而是它告诉我们：

1. 这段时期团队并没有只做保守的均衡优化，还在并行探索极端高 style 方向。
2. `naive_skip`、残差恢复、patch 频谱在极端配置下会呈现和主线不一样的相互作用。
3. `ghost_wireframe` 这种结果值得在后续主线文档里反复引用，因为它证明某些看似“危险”的结构在高 style 场景下可能反而有效。

