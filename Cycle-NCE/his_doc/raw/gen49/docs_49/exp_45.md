# exp_45.md

> 日期: 2026-04-09
> 主题: 把 `45` 这条短周期结构探索线单独立账
> 直接证据:
> - `src/ablate.py` 的 2026-04-05 `92d7012c`
> - `src/45.bat`
> - `src/45/45/45_*/config.json`
> - `src/45/45/45_*/full_eval/epoch_0030/summary.json`

---

## 1. 这条线在干什么

`45` 不是一条“冲最终 SOTA”的长线，而是一条非常典型的开路实验线。

它的任务不是把所有变量一起推满，而是先回答一个更早、更朴素的问题:

1. skip 到底该怎么接
2. 风格信息到底应该更多走 macro 还是 micro
3. cross-attn + skip funnel 这种结构分工值不值得往后继续押

从这个角度看，`45` 的历史地位很清楚:

- `Ablate43` 更像 04-03 时点的“全盘摸底”
- `45` 是第一次把“结构职责分工”明确拉成一条独立主线
- `46` 则是在 `45` 的基础上把 color highway / dirty skip / decoder patch 继续细化

也就是说，`45` 更接近 `46` 的前置原型，而不是一条孤立支线。

---

## 2. 运行编排证据

### 2.1 `ablate.py` 层面的定义

2026-04-05 的 `ablate.py` 把这一批实验命名为 `SERIES_NAME="45"`，并固定了:

- `batch_size=256`
- `num_epochs=30`
- `save_interval=30`
- `full_eval_interval=30`

这说明它就是一组“30 epoch 快速比结构”的短跑实验，不是那种打磨到 80/120 epoch 的长线版本。

### 2.2 `45.bat` 透露出的执行方式

`src/45.bat` 做的事情非常完整:

1. 依次运行 4 个 config
2. 把 `../45_*` 目录统一搬到根目录 `45/`
3. 跑 `batch_distill_full_eval.py --exp_dir 45`
4. 导出 `45.csv`
5. 继续补 `probe_ma.py` 和 `probe_ma_sweep.py`

这很重要，因为它说明 `45` 从设计上就不只是“训练看看”，而是带着一整套评估与 probe 流程去的。

### 2.3 当前工作树里真实能看到的落盘位置

当前仓库里能直接看到的不是根目录 `45/45_*`，而是一个更绕一点的路径:

- `src/45/45/45_01_golden_funnel`
- `src/45/45/45_02_naked_fusion`
- `src/45/45/45_03_macro_dictator`
- `src/45/45/45_04_micro_rebel`

这说明至少在当前工作树中，`45` 的结果被保留在一个嵌套目录里，而不是后来脚本预期的“根目录 `45/` 聚合态”。

这个路径错位本身也应该被记下来，因为它是后面做历史对照时最容易误判成“没有实验结果”的地方。

---

## 3. 四个实验到底改了什么

这一组实验的公共骨架几乎一致:

- `base_dim=96`
- `lift_channels=128`
- `style_dim=160`
- `num_hires_blocks=2`
- `num_res_blocks=1`
- `num_decoder_blocks=2`
- `residual_gain=1.5`
- `style_modulator_type="cross_attn"`
- `body_block_type="global_attn"`
- `skip_fusion_mode="add_proj"`
- `skip_routing_mode="naive"`
- `skip_naive_gain=0.15`
- `inject_gate_hires=0`
- `inject_gate_decoder=1`

所以 `45` 的变量并不分散，核心就是:

- funnel/skip 逻辑先固定成一个“弱 skip + decoder 侧接管”的结构
- 然后只改 spatial prior 与 patch/loss 分布

### 3.1 `45_01_golden_funnel`

这是 45 线自己的基准点。

模型侧:

- `skip_routing_mode="naive"`
- `skip_naive_gain=0.15`
- `style_attn_temperature=0.08`
- `inject_gate_decoder=1`

loss 侧:

- `swd_patch_sizes=[1,3,5,11,21]`
- `w_swd_micro=1.0`
- `w_swd_macro=10.0`
- `w_identity=8.0`
- `w_color=50.0`
- `w_oob=10.0`

它的含义不是“平均用力”，而是:

- macro 比 micro 强很多
- 但 micro 还没有被完全关掉
- skip 只留一个很弱的朴素通道
- decoder 端明确承担一部分风格表达职责

### 3.2 `45_02_naked_fusion`

在 `golden_funnel` 基础上只额外打开:

- `ablation_disable_spatial_prior=true`

它不是改整个结构，而是在问:

- 如果去掉 spatial prior，当前这套 funnel 结构还能不能稳定工作

### 3.3 `45_03_macro_dictator`

结构不变，loss 明显向 macro 倾斜:

- `swd_patch_sizes=[7,15,25]`
- `w_swd_micro=0.0`
- `w_swd_macro=20.0`
- `w_color=40.0`

这条线就是在非常直接地问:

- 如果把风格理解几乎完全交给大 patch / 大纹理，会发生什么

### 3.4 `45_04_micro_rebel`

和 `macro_dictator` 正好形成镜像:

- `swd_patch_sizes=[1,3]`
- `w_swd_micro=15.0`
- `w_swd_macro=0.0`
- `w_color=40.0`

它问的是另一个极端:

- 如果更信小 patch、小纹理、小局部风格，能不能在不完全毁掉结构的前提下把 style 拉起来

---

## 4. 结果对照表

注意: `45` 的 `summary.json` 不是后面常见的 `overall / style_transfer_avg / identity_avg` 结构，而是更老一点的:

- `analysis.all_pairs_overview`
- `analysis.style_transfer_ability`
- `analysis.identity_reconstruction`
- `analysis.photo_to_art_performance`

### 4.1 主表

| 实验 | 结构/配置重点 | style_transfer_style | style_transfer_content | style_transfer_dir | identity_style | identity_content | photo_to_art_style | photo_to_art_content |
|------|---------------|----------------------|------------------------|-------------------|----------------|------------------|--------------------|----------------------|
| `45_01_golden_funnel` | 弱 naive skip + macro 主导 + mixed patch | 0.6674 | 0.5905 | 0.5752 | 0.6821 | 0.5885 | 0.6596 | 0.6209 |
| `45_02_naked_fusion` | 去 spatial prior | 0.6669 | 0.5761 | 0.5813 | 0.6799 | 0.5788 | 0.6532 | 0.5946 |
| `45_03_macro_dictator` | macro-only 倾向 | 0.6137 | 0.5288 | 0.5445 | 0.6264 | 0.5332 | 0.6166 | 0.5318 |
| `45_04_micro_rebel` | micro-only 倾向 | 0.6704 | 0.5994 | 0.5745 | 0.6905 | 0.5989 | 0.6592 | 0.6212 |

### 4.2 all-pairs 总览

| 实验 | all_pairs_style | all_pairs_content | all_pairs_dir |
|------|-----------------|-------------------|---------------|
| `45_01_golden_funnel` | 0.6704 | 0.5901 | 0.5580 |
| `45_02_naked_fusion` | 0.6695 | 0.5767 | 0.5648 |
| `45_03_macro_dictator` | 0.6162 | 0.5297 | 0.5277 |
| `45_04_micro_rebel` | 0.6745 | 0.5993 | 0.5580 |

---

## 5. 这些结果说明了什么

### 5.1 `macro_dictator` 是明显失败的

- style_transfer_style 掉到 `0.6137`
- style_transfer_content 掉到 `0.5288`
- photo-to-art 也一起掉

这说明在 04-05 这个结构阶段里，macro-only 不是“更艺术”，而是“两头都不占便宜”。

### 5.2 `micro_rebel` 反而是这一组里更像赢家的

- style_transfer_style 最高: `0.6704`
- style_transfer_content 最高: `0.5994`
- photo_to_art_content 也最高: `0.6212`

这意味着在当时那版 funnel 结构下，局部纹理监督和弱 skip 更搭。

### 5.3 去掉 spatial prior 会伤内容，但不会立刻毁掉 style

`45_02_naked_fusion` 相比 baseline:

- style_transfer_style 几乎不变
- style_transfer_content 明显下滑
- photo_to_art_content 也下滑

它更像是在证明 spatial prior 是稳定器，而不是直接拉 style 的主引擎。

### 5.4 `45` 的真正贡献是方向判断

它给后面留下的不是最高分，而是几个很关键的判断:

1. macro-only 不是答案
2. micro 分支值得继续保留
3. spatial prior 不能随便扔
4. 弱 skip + decoder 接管风格表达，是一条能继续往下挖的路

---

## 6. 它和后续系列怎么接上

### 6.1 到 `46`

`46` 基本是在 `45` 的框架上继续追问“职责分工”:

- `46_01_highway_cut`
- `46_02_dirty_skip`
- `46_03_decoder_usurpation`
- `46_04_muddy_routing`

### 6.2 到 `chess`

`45` 的 macro / micro 分流，在 `chess` 里还能看到更病理化的再验证:

- `chess_06_patch_micro_only`
- `chess_07_patch_macro_only`

### 6.3 到 `Gate`

`45` 还没有显式测 `attn_gate_mode`，但已经把“谁控制风格注入强度”这个问题摆上台面。

后面的 `Gate` 系列，本质上就是把这件事从“结构分工”继续推进到“显式门控机制”。

---

## 7. 当前可下的历史结论

### 结论 1

`45` 是从 `Ablate43` 走向 `46` 的真正桥梁。

### 结论 2

这条线最重要的负面证据是: macro-only 思路在这里明显不成立。

### 结论 3

这条线最重要的正面证据是: micro 倾向 + 弱 skip + decoder 接管，是一条可继续推进的方向。

### 结论 4

`45` 的实验结果是存在的，只是保存在较老格式的 `summary.json` 和嵌套目录里，不能再把它写成“目录暂不可见”。

