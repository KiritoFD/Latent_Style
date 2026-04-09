# exp_chess.md

> 生成时间: 2026-04-09  
> 数据来源: `src/chess/*/config.json` + `full_eval/epoch_0060_tokenized_distill_epochs50/summary.json`  
> 关联提交: `62f2f3833`

---

## 1. 这条线在研究什么

`chess` 系列不是常规提分实验，而是一条非常明显的病理排查线。  
提交信息已经点明了问题：

- `base01` 出现棋盘格
- 画面有黑框
- 怀疑是数值问题
- 重点怀疑位置编码

所以这条线的价值不在“绝对最高分”，而在于它清晰地拆开了几个可能的病灶：

1. 位置编码
2. attention 温度
3. color highway
4. skip
5. patch 频谱
6. SWD 距离模式

---

## 2. 实验总表

| 目录 | 关键配置 | style | content | clip_dir | p2a_style | p2a_content |
|------|----------|-------|---------|----------|-----------|-------------|
| `chess_01_baseline` | patch=`3,5,15,19`, `cdf`, `color_gain=0.3`, temp=`0.08/0.08` | 0.6855 | 0.7178 | 0.5053 | 0.6462 | 0.7347 |
| `chess_02_no_pos_emb` | 去 `pos_emb` | 0.6887 | 0.7185 | 0.5109 | 0.6516 | 0.7340 |
| `chess_03_high_temp_attn` | `semantic/style temp=0.5` | 0.6889 | 0.7208 | 0.5099 | 0.6517 | 0.7317 |
| `chess_04_no_color_highway` | `color_gain=0.0` | 0.6851 | 0.6669 | 0.5393 | 0.6631 | 0.6977 |
| `chess_05_no_skip` | 目录命名为 `no_skip`，配置仍显示 `skip=none` | 0.6841 | 0.7201 | 0.5020 | 0.6458 | 0.7360 |
| `chess_06_patch_micro_only` | patch=`3` | 0.6899 | 0.7544 | 0.4849 | 0.6479 | 0.7619 |
| `chess_07_patch_macro_only` | patch=`25` | 0.7024 | 0.6682 | 0.5586 | 0.6709 | 0.6879 |
| `chess_08_swd_sort_mode` | `swd_distance_mode=sort` | 0.6796 | 0.7588 | 0.4715 | 0.6422 | 0.7620 |
| `chess_09_no_pos_high_temp` | 去 `pos_emb` + 高温 attention | 0.6853 | 0.7178 | 0.5040 | 0.6470 | 0.7372 |

---

## 3. 关键观察

### 3.1 去掉位置编码不是坏事

`chess_02_no_pos_emb` 相比 baseline：

- style 小升
- content 小升
- `clip_dir` 也略高

这至少说明一件事：

如果当时真的怀疑 `pos_emb` 和棋盘格/黑框有关，这个怀疑不是空想。

### 3.2 高温 attention 也不是灾难

`chess_03_high_temp_attn` 结果和 `no_pos_emb` 很接近，甚至 content 还更高一点。  
这说明当时的 attention 温度至少没有被证明是必须锁死在低温的一项。

### 3.3 `color_highway` 是真实有效结构

`chess_04_no_color_highway` 的 content 掉得非常明显：

- content 直接掉到 `0.6669`
- p2a_content 也明显变差

所以在这条排障线里，`color_highway` 不是装饰项，而是一个真正在稳定输出结构和内容的部件。

### 3.4 微 patch 和宏 patch 分别代表两种方向

`chess_06_patch_micro_only`

- style 没有特别高
- content 很高

`chess_07_patch_macro_only`

- style 最高
- content 明显掉

这和这段时期别的实验线很一致：  
微 patch 更像保守内容锚，宏 patch 更像风格推进器。

### 3.5 `sort` 距离模式明显转向 content-first

`chess_08_swd_sort_mode`：

- style 回落
- content 升到 `0.7588`

这说明 SWD 的距离定义本身也会改变 style/content 平衡，不只是 patch 组合在起作用。

---

## 4. 这条线的实际价值

`chess` 最大的价值是：  
它把“画面伪影排查”从模糊感觉，变成了一套可以重跑的结构排查矩阵。

这类文档特别重要，因为后续如果又看到：

- 棋盘格
- 黑框
- 条纹
- 不稳定 attention

就可以直接回看这条线，而不是重新从零开始猜。

---

## 5. 当前结论

从当前结果看，`chess` 系列最值得保留的结论是：

1. `pos_emb` 至少不是明显正收益，甚至可能是问题源之一。
2. `color_highway` 不能轻易拿掉。
3. 宏 patch 会推 style，但代价是 content 下滑。
4. `sort mode` 更偏保内容。

也就是说，这条线已经具备“故障诊断手册”的价值，而不只是一个临时实验目录。

