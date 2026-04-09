# exp_46.md

> 生成时间: 2026-04-09  
> 数据来源: `46/*/config.json` + `full_eval/epoch_0080_tokenized_distill_epochs200/summary.json`  
> 关联提交: `28e6d0719`、`486f9cc09`

---

## 1. 这条线在研究什么

`46` 系列是 4 月上旬最重要的一条主结构线之一。  
它的研究目标不是简单提分，而是把主模型里的三条关键路径拆开来查：

1. `color_highway`
2. `skip_clean / skip_blur`
3. `decoder_highpass`

再加上一个额外控制项：

4. `semantic_attn_temperature`

这意味着 `46` 系列的真正主题不是“调参”，而是：

**谁在负责颜色，谁在负责结构，谁在负责高频风格。**

---

## 2. 实验总表

| 目录 | 关键配置 | style | content | clip_dir | p2a_style | p2a_content |
|------|----------|-------|---------|----------|-----------|-------------|
| `46_00_holy_grail` | `no_res=True`, `skip_clean=True`, `skip_blur=True`, `dec_highpass=True`, `color_highway=0.5`, `sem_temp=0.08`, patch=`1,3,11,15,25`, `w_micro=5`, `w_macro=80` | 0.6766 | 0.7613 | 0.4941 | 0.6574 | 0.7527 |
| `46_01_highway_cut` | 在 baseline 上把 `color_highway=0.0` | 0.6837 | 0.7356 | 0.5200 | 0.6708 | 0.7273 |
| `46_02_dirty_skip` | `skip_clean=False`, `skip_blur=False` | 0.6785 | 0.7543 | 0.4999 | 0.6608 | 0.7468 |
| `46_03_decoder_usurpation` | `dec_highpass=False` | 0.6662 | 0.7829 | 0.4615 | 0.6482 | 0.7787 |
| `46_04_muddy_routing` | `sem_temp=0.5` | 0.6813 | 0.7561 | 0.5052 | 0.6660 | 0.7500 |

---

## 3. 核心观察

### 3.1 `holy_grail` 真的是一个中心点，不是偶然命名

`46_00_holy_grail` 的价值不在它是不是单项冠军，而在它是最适合做中心对照组的版本：

- style 不低
- content 也不算差
- clip_dir 与 p2a 也处在可接受区间

后面几个实验几乎都像是在问：

- 如果拿掉某一条路，会偏向哪边？

所以它是结构研究的中心点，而不是简单 baseline。

### 3.2 `highway_cut` 非常清楚地说明了 color highway 的职责

当 `color_highway_gain` 从 `0.5` 降到 `0.0`：

- style 从 `0.6766` 升到 `0.6837`
- content 从 `0.7613` 掉到 `0.7356`
- clip_dir 反而升到 `0.5200`

这说明 `color_highway` 的存在更像是在约束模型别太激进，而不是帮它冲更强风格。  
它牺牲了一点风格推进，换来了更稳的内容和颜色结构。

### 3.3 `dirty_skip` 证明 skip 清洁不是心理安慰

只把 `skip_clean / skip_blur` 关掉后：

- style 没有明显抬高
- content 也没有更优

这说明干净 skip 不是无意义复杂化，它确实是在帮助模型隔离“有用结构”和“脏纹理泄漏”。

### 3.4 `decoder_usurpation` 非常像一把秤

把 `decoder_highpass` 关掉后：

- style 掉到 `0.6662`
- content 升到 `0.7829`
- p2a_content 也明显更高

这就是最典型的“decoder 高频通路在拿 content 换 style”的证据。  
所以 `decoder_highpass` 不是副角，而是主平衡杆之一。

### 3.5 `muddy_routing` 说明高温 attention 不是结构性突破

把 `semantic_attn_temperature` 提到 `0.5`：

- style 略升
- content 略降
- 没有出现压倒性改善

这意味着 attention 温度确实在影响模型性格，但它不是这条线里的头号决定因素。

---

## 4. 这条线在大历史里的意义

`46` 是 4 月上旬最像“结构因果实验”的一条线。  
它的价值不只是有结果，而是变量拆得非常干净：

- 颜色主通道
- skip 清洁
- decoder 高频
- attention 温度

因此它在后续写模型史时应该被反复引用，因为它是少数能直接把“结构职责分工”写清楚的系列。

---

## 5. 当前结论

如果只保留一句话：

`46` 系列证明了 4 月上旬的主矛盾不是有没有模块，而是颜色、结构、高频纹理这三条路径该怎么分权。

