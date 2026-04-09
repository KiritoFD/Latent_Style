# exp_series_scoreboard.md

> 生成时间: 2026-04-09  
> 用途: 汇总 `Ablate43`、`46`、`Aline120`、`chess`、`freq` 等已能直接读取到 `summary.json` 的实验支线，形成跨系列对照表  
> 说明: 这里只记录当前可验证的一手结果，不替代各支线详细文档

---

## 1. 使用方式

这张表解决的是一个很实际的问题：

- 文档越来越多以后，很容易知道每条线研究了什么，
- 但反而不容易一眼看出“哪条线在 style 更猛，哪条线在 content 更稳”。

所以这里把当前能稳定读取到的关键实验，拉成一个横向排行榜。

---

## 2. 关键代表点总表

| 系列 | 实验 | style | content | clip_dir | 这条结果最能说明什么 |
|------|------|-------|---------|----------|----------------------|
| `Ablate43` | `Ablate43_A01_ResOn_TheFilter` | 0.6799 | 0.7425 | 0.5092 左右 | 开 residual 与过滤结构后，style 可被明显拉高，但内容会掉 |
| `Ablate43` | `Ablate43_S01_Baseline_Gold` | 0.6481 | 0.8595 | 0.4154 左右 | 04-03 的均衡基线 |
| `46` | `46_01_highway_cut` | 0.6837 | 0.7356 | 0.5200 | 去掉 color highway 后 style 更高，但 content 掉得更明显 |
| `46` | `46_00_holy_grail` | 0.6766 | 0.7613 | 0.4941 | 46 系列的结构中心点 |
| `Aline120` | `Aline120_aline_03_ghost_wireframe` | 0.7114 | 0.6364 | 0.6067 | 极端高 style 支线的高位结果 |
| `chess` | `chess_07_patch_macro_only` | 0.7024 | 0.6682 | 0.5586 | 宏 patch 能强推 style，但很伤 content |
| `chess` | `chess_08_swd_sort_mode` | 0.6796 | 0.7588 | 0.4715 | 距离模式切换会显著把模型推向 content-first |
| `freq` | `freq_04_no_idt_abyss` | 0.6454 | 0.8751 | 0.3514 | 去掉 IDT 后，在这条线里 style/content 都更强 |
| `freq` | `freq_08_extreme_asymmetry` | 0.6366 | 0.8771 | 0.3314 | 极不对称 patch 组合更像内容守恒方案 |
| `repulse` | `Layer-Norm-repulse` | 0.6881 | 0.6529 | 未单列 | repulsive loss 可以明显把平衡推向更高 style、更低内容 |
| `repulse` | `46_repulse` | 0.6474 | 0.8251 | 未单列 | 同样是 repulse，接在不同基线上结果方向完全不同 |

---

## 3. 按 style 看

### 当前高位组

| 排名感受 | 实验 | style | 备注 |
|----------|------|-------|------|
| 很高 | `Aline120_aline_03_ghost_wireframe` | 0.7114 | 当前这批里非常接近极限推进型 |
| 很高 | `chess_07_patch_macro_only` | 0.7024 | 宏 patch 单压 style |
| 高 | `Layer-Norm-repulse` | 0.6881 | repulsive 支线有效抬高 style |
| 高 | `chess_03_high_temp_attn` | 0.6889 | 高温 attention 也能抬一点 style |
| 高 | `chess_02_no_pos_emb` | 0.6887 | 去掉位置编码并未削弱 style |
| 高 | `46_01_highway_cut` | 0.6837 | 典型的“拿 content 换 style”结构点 |

### 解释

从这些点可以看出，style 冲高通常有几条路径：

1. 极端弱化内容约束，像 `Aline120`
2. 偏宏观 patch，像 `chess_07`
3. 用 repulsive 或 decoder/skip 结构重新分权
4. 拿掉某些保守稳定通道，比如 `color_highway`

---

## 4. 按 content 看

### 当前高位组

| 排名感受 | 实验 | content | 备注 |
|----------|------|---------|------|
| 很高 | `freq_08_extreme_asymmetry` | 0.8771 | 极偏 patch 组合，但更偏内容守恒 |
| 很高 | `freq_04_no_idt_abyss` | 0.8751 | 去掉 IDT 在这条线里反而两头都不差 |
| 很高 | `freq_06_yuv_dictatorship` | 0.8736 | 高 color 权重更偏内容稳定 |
| 很高 | `freq_03_large_view_awareness` | 0.8724 | 大视野 patch 组合更保内容 |
| 高 | `Ablate43_L01_SWD_TurnOff` | 0.8921 | 极端 content-first 例子，但 style 掉得厉害 |
| 高 | `Ablate43_I04_Gain_Vanilla` | 0.8833 | 降低 residual gain 的代价是 style 回落 |

### 解释

高 content 的方案不一定是“主线最佳”，因为它们常常意味着：

- 风格被压住
- 模型更接近保守重建
- 或者训练目标对风格改写不够强

所以 content 高并不自动等于好，必须和 style 一起看。

---

## 5. 按“均衡性”看

如果不追求单项极值，而是看 style/content 是否同时站得住，当前比较像“均衡样本”的有：

| 实验 | style | content | 为什么算均衡 |
|------|-------|---------|--------------|
| `46_00_holy_grail` | 0.6766 | 0.7613 | 46 系列中心点，不极端，但结构解释力强 |
| `Ablate43_A01_ResOn_TheFilter` | 0.6799 | 0.7425 | style 不低，content 还能接受 |
| `chess_02_no_pos_emb` | 0.6887 | 0.7185 | 排障线里比较健康的折中点 |
| `chess_03_high_temp_attn` | 0.6889 | 0.7208 | 和上项相似，稍微更偏内容 |

这些点的重要性在于：

- 它们不一定是单项冠军；
- 但更容易告诉我们“结构往哪个方向改是健康的”。

---

## 6. 当前最值得反复引用的几个点

### `Aline120_aline_03_ghost_wireframe`

意义：

- 证明极端高 style 路线是真能打到 `0.71+` 的；
- 说明 `naive_skip + residual` 在极端配置下不一定是坏事。

### `46_00_holy_grail`

意义：

- 是主线结构讨论的中心点；
- 很多后续 `highway / skip / decoder highpass` 实验都是围着它转。

### `freq_04_no_idt_abyss`

意义：

- 对“IDT 一定必要吗”提出非常强的反例；
- 说明在某些频谱配方里，去掉 IDT 反而可以让两头都不差。

### `chess_07_patch_macro_only`

意义：

- 很清楚地证明宏 patch 可以强推 style；
- 也很清楚地展示了 content 代价。

---

## 7. 当前结论

如果只看 04-03 到 04-09 这一段，最重要的事实不是“谁是总冠军”，而是不同系列已经开始承担不同研究职责：

- `Ablate43`: 稳定体拆解
- `46`: 主结构分权
- `Aline120`: 极端高 style 探针
- `chess`: 病理排障
- `freq`: 频谱/identity/color 关系研究
- `repulse`: 新损失形态对平衡的重新拉扯

也正因为如此，这张表不能只拿来找第一名，更要拿来判断：

**当我们想推高 style、保住 content、排查伪影、或者验证某种损失时，应该先回看哪条线。**

