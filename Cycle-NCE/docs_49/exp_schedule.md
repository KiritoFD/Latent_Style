# exp_schedule.md

> 生成时间: 2026-04-09  
> 工作目标: 以 `git` 历史中的 `Cycle-NCE/src/ablate.py` 为主索引，建立实验编排时间线，并把对应实验目录中的 `config.json` 与 `summary.json` 挂到同一张对照表里。  
> 参考材料: `docs_49/plan_49.md`、`docs_49/exp_49.md`、`his_doc/History_Report.md`、`git log -- Cycle-NCE/src/ablate.py`、各实验目录的 `config.json` / `summary.json`

---

## 1. 使用说明

这份文档不是只罗列实验目录，而是把三层信息并排放在一起：

1. `ablate.py` 当时定义了什么实验矩阵。
2. 实验目录里实际落盘的 `config.json` 是什么。
3. `summary.json` 给出的结果大致落在什么水平。

其中有两种证据类型：

- **直接证据**: `ablate.py` 提交内容、实验目录中的 `config.json`、`summary.json`
- **推断**: 当目录命名与 `ablate.py` 命名不完全一致时，根据提交时间、配置模式、目录结构做的对应判断。凡是推断我都会显式写出。

---

## 2. ablate.py 历史断点总表

| 时间 | Commit | SERIES_NAME | 阶段主题 | 证据状态 |
|------|--------|-------------|----------|----------|
| 2026-02-27 | `52b158ad` | 未显式分组，`master_sweep_*` | 20 组大规模容量/patch/LR 曲面扫描 | 有 `ablate.py`，当前未看到同名聚合目录 |
| 2026-03-05 | `3dae1763` | 未显式分组，decoder ablate | 6 组 decoder 正交消融 | 有 `ablate.py`，未在根目录看到同名聚合目录 |
| 2026-03-19 | `c8be1f3c` | 未显式分组，full ablation | 12 组全栈消融矩阵 | 有 `ablate.py`，与 `Ablate43/` 存在主题相近但命名不同的落地结果 |
| 2026-04-02 | `58831eb6` | `micro` | micro-batch 后的小 patch / 高频 / 残差组合实验 | 有 `ablate.py`，当前未看到 `micro/` 根目录 |
| 2026-04-04 01:03 | `09284202` | `Aline120` | color hard align 之前后的 Aline 支线 | 有 `ablate.py`，有 `Aline120/` 根目录待继续挖结果 |
| 2026-04-04 04:19 | `6b34f220` | `Aline120` | Aline 支线延长到 120 epoch | 有 `ablate.py` |
| 2026-04-05 | `92d7012c` | `45` | 结构调优后的 45 系列首轮实验 | 有 `ablate.py`，根目录 `45/` 不在当前工作树可见 |
| 2026-04-06 01:35 | `28e6d071` | `46` | holy grail / highway / skip / decoder highpass 结构消融 | 有 `ablate.py`，有 `46/` 真实目录与结果 |
| 2026-04-06 11:20 | `486f9cc0` | `46` | 46 系列扩展，加入 `07_rebuild_holy_grail` 等后续项 | `ablate.py` 有增补，当前 `46/` 目录只稳定看到 00-04 |
| 2026-04-06 18:42 | `3ad9977e` | `repuls` | repulsive loss + skip/structure 联合消融 | 有 `ablate.py`，根目录存在 `repuls/`，但当前缺少完整总结材料 |
| 2026-04-07 19:13 | `97de1653` | 未单独读取 | Layer-Norm-idt_schedule 方向 | 需后续继续深挖 |
| 2026-04-07 23:50 | `62f2f383` | `chess` | 针对棋盘格/黑框/位置编码问题的 10 组实验 | 有 `ablate.py`，`src/chess/` 已有目录 |
| 2026-04-08 21:59 | 当前工作树 | `Gate` | 新一轮 gate/aux/idt-energy 组合实验 | `src/ablate.py` 现行版本，目录尚未看到完整跑完结果 |

---

## 3. 逐阶段对照

### 3.1 2026-02-27 `52b158ad`：master sweep 曲面扫描

**代码直接证据**

- `ablate.py` 用 `calculate_precise_vram_bs_and_lr()` 做四元响应面求解。
- 目标不是单一结构消融，而是把一批容量/patch/LR 组合都咬在约 10.5GB 显存甜点位。
- 输出目录命名为 `../master_sweep_{name}`。

**核心配置机制**

- 基线来源: `config.json`
- 强制覆盖:
  - `w_identity = 2.0`
  - `w_delta_tv = 0.1`
  - `w_color = 15.0`
  - `w_swd = 30.0`
  - `num_epochs = 100`
  - `full_eval_interval = 100`
  - `save_interval = 50`
- 动态求解:
  - `batch_size`
  - `learning_rate`
  - `min_learning_rate`

**实验矩阵**

| 实验名 | 主要改动 |
|--------|----------|
| `01_cap_64` | `base_dim=64`, `ada_mix_rank=16`, patch=`[5,7,9]` |
| `02_cap_128` | `base_dim=128`, `ada_mix_rank=32` |
| `03_cap_192` | `base_dim=192`, `ada_mix_rank=48` |
| `04_cap_256` | `base_dim=256`, `ada_mix_rank=64` |
| `05_patch_micro` | patch=`[3,5]` |
| `06_patch_std` | patch=`[3,5,7]` |
| `07_patch_xmax` | patch=`[7,11,15]` |
| `08_lr_fast` | `learning_rate=5e-4` |
| `09_lr_slow` | `learning_rate=5e-5` |
| `10_wide_xmax` | 宽模型 + 大 patch |
| `11_narrow_xmax` | 窄模型 + 大 patch |
| `12_mid_xmax` | 中等容量 + 大 patch |
| `13_wide_micro` | 宽模型 + 微 patch |
| `14_narrow_micro` | 窄模型 + 微 patch |
| `15_split_brain_128` | patch=`[3,15]` |
| `16_split_brain_256` | patch=`[3,7,15]` |
| `17_extreme_underpowered` | `base_dim=32`, `rank=8` |
| `18_extreme_overpowered` | `base_dim=384`, `rank=64` |
| `19_the_abyss` | 大 patch + `w_swd=50` |
| `20_golden_balance` | `base_dim=192`, `rank=32`, patch=`[5,7,11]` |

**这阶段的历史意义**

- 这不是“某一个模型结论”，而是给后面 3 月、4 月大量实验提供了参数先验。
- 它更像前史中的参数地图，而不是最终结构报告。

---

### 3.2 2026-03-05 `3dae1763`：decoder orthogonal sweep

**代码直接证据**

- `ablate.py` 改成 6 组 decoder-focused orthogonal sweep。
- 基线文件: `config.json`
- 输出脚本: `run_decoder_ablate_6.bat`
- 汇总脚本: `collect_ablation_results.py`

**固定训练参数**

- `num_epochs=80`
- `full_eval_interval=40`
- `save_interval=20`

**实验矩阵**

| 实验名 | HF | HF 比例 | `w_identity` | `w_delta_tv` | 实验意图 |
|--------|----|---------|--------------|--------------|----------|
| `decoder-A-anchor-nohf` | 否 | 2.0 | 1.2 | 0.005 | 强 identity、去掉 HF |
| `decoder-B-hf-strict-id` | 是 | 2.0 | 1.2 | 0.005 | 保持强 identity，恢复 HF |
| `decoder-C-relaxed-id-nohf` | 否 | 2.0 | 0.25 | 0.005 | 降低 identity，看 style 是否抬升 |
| `decoder-D-sweetspot` | 是 | 2.0 | 0.3 | 0.005 | 预期甜点位 |
| `decoder-E-extreme-brush` | 是 | 5.0 | 0.05 | 0.005 | 极端强化笔触 |
| `decoder-F-tv-off` | 是 | 2.0 | 0.3 | 0.0 | 验证 TV 是否必要 |

**历史意义**

- 这是后面“TV 可以扔”“decoder sweetspot”的前置证据链。
- 3 月 22 日 `weight系列实验，TV可以扔了` 与这一阶段高度连续。

---

### 3.3 2026-03-19 `c8be1f3c`：full ablation matrix

**代码直接证据**

- 基线文件: `config_decoder-D-sweetspot.json`
- 统一默认:
  - `lift_channels=128`
  - `base_dim=96`
  - `ada_mix_rank=32`
  - `num_decoder_blocks=1`
  - `residual_gain=1.0`
  - `swd_distance_mode='cdf'`
  - `swd_use_high_freq=True`
  - `w_color=0.5`
  - `w_identity=0.3`
  - `w_delta_tv=0.005`
  - `batch_size=320`
  - `learning_rate=1.4e-4`
  - `num_epochs=60`

**代码中定义的 12 组实验**

| 实验名 | 主要改动 |
|--------|----------|
| `abl_heavy_decoder` | `num_decoder_blocks=6`, `batch_size=256` |
| `abl_no_residual` | `residual_gain=0.0` |
| `abl_vanilla_gn` | `ada_mix_rank=1` |
| `abl_no_skip_filter` | `style_skip_content_retention_boost=1.0` |
| `abl_no_id` | `w_identity=0.0` |
| `abl_hard_sort` | `swd_distance_mode='sort'` |
| `abl_no_hf_swd` | `swd_use_high_freq=False` |
| `abl_no_color` | `w_color=0.0` |
| `abl_no_tv` | `w_delta_tv=0.0` |
| `scale_c64` | `lift=64`, `base_dim=48`, `rank=16`, `batch=512` |
| `scale_c256` | `lift=256`, `base_dim=192`, `rank=64`, `batch=128` |
| `baseline` | 基线恢复项 |

**与 `Ablate43/` 的关系**

- 当前根目录里真实存在的是 `Ablate43/`，且命名不是 `abl_*`，而是 `S/A/I/L/P` 五个子主题。
- 我认为两者属于**同一研究母题**，但不能在没有更多提交证据的情况下强行说一一等价。
- 下面先把 **真实落地的 `Ablate43` 结果** 单独挂出来，作为 04-03 时点的可靠一手材料。

#### 3.3.1 `Ablate43/` 真实落地目录对照

以下数据来自目录中的 `config.json` 与 `summary.json`，不是从 `ablate.py` 反推。

| 目录 | 配置摘要 | style | content | 判断 |
|------|----------|-------|---------|------|
| `Ablate43_S01_Baseline_Gold` | `base_dim=96`, `lift=128`, `dec=2`, `res=1.5`, patch=`1,3,5`, `w_id=0`, `w_color=50` | 0.6481 | 0.8595 | 04-03 基线 |
| `Ablate43_S02_DeepConv3` | decoder 更深 (`dec=3`) | 0.6502 | 0.8540 | style 微增，content 微降 |
| `Ablate43_A01_ResOn_TheFilter` | 保留 residual 的 A 组 | 0.6799 | 0.7425 | style 拉高，content 明显掉 |
| `Ablate43_A02_Capacity_Conv1` | 容量/卷积调整 | 0.6520 | 0.8601 | 更像保守修正 |
| `Ablate43_A03_WindowAttn_Size8` | 引入 window attention 尺寸 8 | 0.6548 | 0.8584 | attention 没有立刻带来质变 |
| `Ablate43_A04_Modulator_GlobalOnly` | modulator 退化为更全局形式 | 0.6522 | 0.8598 | 与基线接近 |
| `Ablate43_I01_Skip_TotalBlind` | `skip_mode=none` | 0.6533 | 0.8551 | 完全砍 skip 不优 |
| `Ablate43_I02_Skip_ConcatFusion` | concat 融合 skip | 0.6503 | 0.8599 | 更像结构替代而非提升 |
| `Ablate43_I03_Gate_Hires_Only` | 只开高分辨率 gate | 0.6520 | 0.8608 | 几乎持平 |
| `Ablate43_I04_Gain_Vanilla` | `residual_gain=1.0` | 0.6387 | 0.8833 | content 很强，style 回落 |
| `Ablate43_L01_SWD_TurnOff` | 关闭/弱化 SWD | 0.6365 | 0.8921 | content 最高，style 退化，反证 SWD 仍关键 |
| `Ablate43_L02_Color_TurnOff` | `w_color=0` | 0.6502 | 0.8630 | color 不是 style 主引擎，但拿掉后无增益 |
| `Ablate43_L03_IDT_MassiveReturn` | `w_identity=20` | 0.6394 | 0.8770 | 强 identity 明显压制 style |
| `Ablate43_L04_SWD_Nuke` | SWD 核打穿式修改 | 0.6467 | 0.8630 | 仍不如合理 baseline |
| `Ablate43_P01_Patch_LargeOnly` | patch=`19,25,31` | 0.6577 | 0.8507 | 大 patch 单独跑不差，但不稳 |
| `Ablate43_P02_Patch_FullSpectrum` | patch=`1,3,5,15,25` | 0.6520 | 0.8550 | 全频谱比 baseline 更均衡但不显著更强 |
| `Ablate43_P03_Patch_NanoClash` | patch=`1,3` | 0.6499 | 0.8596 | 微 patch 过窄也没有明显赢 |

**阶段结论**

- 04-03 时点最清楚的结论不是“某个单项大赢”，而是：
  - **强 SWD 不能随便砍**
  - **强 IDT 会压 style**
  - **patch 需要宽窄搭配，单押大或单押小都不是决定性答案**
  - **attention/skip 的替换在这个版本线里还没有形成压倒性收益**

---

### 3.4 2026-04-02 `58831eb6`：micro 系列

**代码直接证据**

- `SERIES_NAME="micro"`
- 基线自动从 `config.json` 或 `config_p_1_5_9_15_hf_1p0.json` 等候选中选择
- 实验全部围绕小 patch、高 `w_swd`、零 identity、两层 decoder、强 residual gain 展开

**实验矩阵**

| 实验名 | 关键配置 |
|--------|----------|
| `E01_Patch3_Gain4_LR2e4` | `ablation_no_residual=True`, `dec=2`, `residual_gain=4.0`, `w_identity=0`, patch=`[3]`, `w_swd=250` |
| `E02_Patch3_5_Gain4_LR2e4` | 在 E01 上加入 patch=`[3,5]` + `HF ratio=10` |
| `E03_Patch3_5_7_Gain4_LR2e4` | patch 扩到 `[3,5,7]` |
| `E04_Patch1_3_5_Gain4_LR2e4` | patch 改为 `[1,3,5]` |

**历史意义**

- 对应提交信息“micro batch效果大好”。
- 这条线把注意力从“大而全结构”重新拉回“小 patch + 大 style push + 更紧训练回路”。

---

### 3.5 2026-04-05 `92d7012c`：45 系列

**代码直接证据**

- `SERIES_NAME="45"`
- 训练固定:
  - `batch_size=256`
  - `num_epochs=30`
  - `full_eval_interval=30`
- `45` 更像一个短周期开路实验，目的在于快速比较结构方向。

**实验矩阵**

| 实验名 | 模型配置 | loss 配置 | 备注 |
|--------|----------|-----------|------|
| `01_golden_funnel` | `skip_routing_mode='naive'`, `skip_naive_gain=0.15`, `style_attn_temperature=0.08` | `w_identity=8`, patch=`[1,3,5,11,21]`, `w_swd_micro=1`, `w_swd_macro=10`, `w_color=50`, `w_oob=10` | 45 系列基线 |
| `02_naked_fusion` | 在上项基础上 `ablation_disable_spatial_prior=True` | 同上 | 去掉空间先验 |
| `03_macro_dictator` | 同样 funnel，但偏向宏观 | patch=`[7,15,25]`, `w_swd_micro=0`, `w_swd_macro=20`, `w_color=40` | 宏观纹理主导 |
| `04_micro_rebel` | 同样 funnel，但偏向微观 | patch=`[1,3]`, `w_swd_micro=15`, `w_swd_macro=0`, `w_color=40` | 微观纹理主导 |

**与目录的关系**

- 当前工作树中没有直接可读的 `45/` 根目录结果。
- `History_Report.md` 与 `commits_0403_0409_detail.md` 都把 `45_01_golden_funnel` 视为这一波结构转折的起点。
- 因此这部分目前主要以 `ablate.py` 为直接证据，实验结果要靠后续继续挖历史目录或压缩包。

---

### 3.6 2026-04-04 `09284202` / `6b34f220`：Aline120 支线

**代码直接证据**

- 两个相邻版本的 `ablate.py` 都使用 `SERIES_NAME="Aline120"`。
- 这说明 04-04 当天主线之外还并行开了一条 Aline 支线。

**第一版（`09284202`）实验矩阵**

| 实验名 | 关键配置 |
|--------|----------|
| `aline_01_oracle` | `aline=True`, `ablation_no_residual=True`, `residual_gain=2.0`, `w_identity=0`, patch=`[1,3,5]`, `w_swd=250`, `w_color=150` |
| `aline_02_texture_maniac` | `residual_gain=2.5`, patch=`[3,5]`, `w_swd=300`, `w_color=150` |
| `aline_03_ghost_wireframe` | `ablation_naive_skip=True`, `ablation_naive_skip_gain=0.15`, `residual_gain=1.5`, patch=`[3,5,7]` |
| `aline_04_macro_trap` | patch=`[5,7,9]`, `w_swd=200` |
| `aline_05_idt_poison` | `w_identity=20`, patch=`[1,3,5]`, `w_swd=150` |

**第二版（`6b34f220`）变化**

- 仍然是同样的 5 个实验名。
- 最大变化在训练时长与 color loss：
  - `num_epochs` 从 `30` 拉长到 `120`
  - `w_color` 从 `150` 改成 `50`
- 这说明 04-04 当天并不是简单修 bug，而是在把 Aline 从“短跑极端试验”改成更可训练、可观察的长期版本。

**这条线的意义**

- 提交信息 `色彩硬对齐` 和 `paper&ppt` 表面看起来不像系统实验，但 `ablate.py` 其实在同步推进一条独立支线。
- `Aline120` 很像一个“极端强 style 方案”的平行验证场，不是当时主线 holy grail 的直接替代。

**真实目录结果**

`Aline120/` 目录中已经有完整落盘结果，而且目前看到的是 `epoch_0030_tokenized_distill_epochs200` 这一层最稳定。

| 目录 | 配置摘要 | style | content | 观察 |
|------|----------|-------|---------|------|
| `Aline120_aline_01_oracle` | `no_res=True`, `res=2.0`, `w_id=0`, `w_swd=250`, `w_color=150`, patch=`1,3,5` | 0.6579 | 0.5474 | 强 style，但内容保持明显不足 |
| `Aline120_aline_02_texture_maniac` | `no_res=True`, `res=2.5`, `w_swd=300`, patch=`3,5` | 0.6730 | 0.5611 | 更激进的纹理推进，style 继续上升 |
| `Aline120_aline_03_ghost_wireframe` | `no_res=False`, `naive_skip=True`, `skip_gain=0.15`, patch=`3,5,7` | 0.7114 | 0.6364 | 这条线最强，说明 naive skip 与残差恢复并非一定坏事 |
| `Aline120_aline_04_macro_trap` | `no_res=True`, patch=`5,7,9`, `w_swd=200` | 0.6571 | 0.5437 | 偏宏观 patch 没带来优势 |
| `Aline120_aline_05_idt_poison` | `w_id=20`, `w_swd=150`, patch=`1,3,5` | 0.6782 | 0.5626 | 名字叫 poison，但在这条线里强 IDT 并没有把 style 完全压死 |

这组结果最值得记的一点是：

- `Aline120_aline_03_ghost_wireframe` 的 `style=0.7114` 已经进入这段时期的高位区间；
- 但 content 仍明显弱于 46 / Ablate43 这些更均衡的主线版本。

所以 Aline 更像“上限探针”，不是主线均衡解。

---

### 3.7 2026-04-06 `28e6d071`：46 系列首轮 holy grail 结构消融

这是目前 `ablate.py` 与实验目录对得最稳的一段。

**代码直接证据**

- `SERIES_NAME="46"`
- 训练强制覆盖:
  - `batch_size=256`
  - `num_epochs=80`
  - `learning_rate=1e-4`
  - `warmup_ratio=0.125`
- loss 强制覆盖:
  - `swd_num_projections=384`

**首轮实验矩阵**

| 实验名 | ablate.py 中的关键 patch |
|--------|--------------------------|
| `00_holy_grail` | `ablation_no_residual=True`, `ablation_skip_clean=True`, `ablation_skip_blur=True`, `ablation_decoder_highpass=True`, `color_highway_gain=0.5`, `semantic_attn_temperature=0.08`, patch=`[1,3,11,15,25]`, `w_swd_micro=5`, `w_swd_macro=80` |
| `01_highway_cut` | 相比 baseline，把 `color_highway_gain=0.0` |
| `02_dirty_skip` | 把 `ablation_skip_clean=False`, `ablation_skip_blur=False` |
| `03_decoder_usurpation` | 把 `ablation_decoder_highpass=False` |
| `04_muddy_routing` | 把 `semantic_attn_temperature=0.5` |
| `05_micro_dictatorship` | patch 改成 `[1,3]`，`w_swd_micro=80`, `w_swd_macro=0` |
| `06_hard_anchor` | 把 `ablation_no_residual=False`，恢复 residual anchor |

**真实目录对照**

当前 `46/` 目录中稳定能看到并有结果的是 00-04。05/06 在当前工作树中未见稳定结果目录。

| 目录 | config 摘要 | tokenized epoch80 style | tokenized epoch80 content | 判断 |
|------|-------------|-------------------------|---------------------------|------|
| `46_00_holy_grail` | `base_dim=96`, `lift=128`, `dec=2`, `skip_mode=none`, `fusion=add_proj`, `w_id=5`, `w_color=40` | 0.6766 | 0.7613 | 结构总方案成立 |
| `46_01_highway_cut` | 去掉 `color_highway_gain` 主通道的对应版本 | 0.6837 | 0.7356 | style 更高，但 content 掉得更明显 |
| `46_02_dirty_skip` | skip 清洁/模糊机制退掉 | 0.6785 | 0.7543 | 比 baseline 略差，说明 skip 清洁有作用 |
| `46_03_decoder_usurpation` | 关 decoder highpass | 0.6662 | 0.7829 | content 上去，style 下去，说明 decoder 高频通道确实在推风格 |
| `46_04_muddy_routing` | 提高 attention temperature | 0.6813 | 0.7561 | 路由更“浑”后没有明显赢 baseline |

**这段的直接结论**

1. `46_01_highway_cut` 的 style 最高，但它不是最均衡方案。
2. `46_03_decoder_usurpation` 明显呈现“拿 style 换 content”的倾向。
3. 这条线支撑了 04-06 提交说明中的那几个结构判断：
   - color highway 不是装饰项
   - clean skip 不是白加
   - decoder highpass 直接关系到风格强度

---

### 3.8 2026-04-06 `486f9cc0`：46 系列继续扩展

**代码直接证据**

- 仍然是 `SERIES_NAME="46"`，但在上一个版本基础上往后加实验。
- 当前片段能直接看到新增的：
  - `07_rebuild_holy_grail`
- 其余新增项由于抓取片段被截断，后续还要继续翻完整版本。

**已确认的新增方向**

| 实验名 | 关键改动 |
|--------|----------|
| `07_rebuild_holy_grail` | `ablation_decoder_highpass=False`, `color_highway_gain=1.0`, `semantic_attn_temperature=0.1`, `w_identity=3.0`, 试图重建一个更强 style 的 holy grail |

**阶段判断**

- 这说明 04-06 上午那轮不是收口，而是开始进入“在 holy grail 周围做二次逼近”的阶段。
- 也就是说，`46_00` 不是终点，而是一个可被继续重构的中心点。

---

### 3.9 2026-04-06 `3ad9977e` 与 2026-04-07 `97de1653`：repuls 系列延续

**代码直接证据**

- `SERIES_NAME="repuls"`
- 默认基线优先取:
  - `config_repulse.json`
  - `config_in-idt.json`
  - `config.json`
- 强制模型覆盖:
  - `ablation_no_residual=False`
  - `ablation_direct_delta_blend=True`
- 强制 loss 覆盖:
  - `swd_num_projections=384`
  - `swd_distance_mode='sort'`

**实验矩阵**

| 实验名 | 关键配置 |
|--------|----------|
| `00_l1_mean_filter` | `w_repulsive=25`, `margin=0.8`, `temperature=0.2`, `repulsive_mode='l1'`, `w_swd_unified=50` |
| `01_mse_local_tear` | `w_repulsive=30`, `margin=0.3`, `repulsive_mode='mse'` |
| `02_micro_mesh` | `skip_bottleneck_channels=2`, `skip_spatial_dropout_p=0.3`, `ablation_skip_blur=True` |
| `03_zero_skip_isolation` | `skip_routing_mode='none'` |
| `04_subzero_one_hot` | `semantic_attn_temperature=0.05`, `color_highway_gain=0.2` |
| `05_highway_override` | `color_highway_gain=0.9`, `w_repulsive=20`, patch=`[3,5]` |
| `06_structure_sacrifice` | `w_identity=1`, `w_swd_unified=60`, `w_repulsive=30` |
| `07_edge_rebel` | `w_repulsive=25`, `swd_use_high_freq=True`, `hf_ratio=0.25`, `w_identity=4` |

**关于 2026-04-07 `97de1653`**

- 这次提交的 `ablate.py` 仍然是 `SERIES_NAME="repuls"`，不是新的 `idt_schedule` 生成器。
- 也就是说，提交消息里提到的“开始学到东西，低idt导致视觉效果不好，但是确实有区别了”，是在 repuls 这条代码框架里继续推进的。
- 这提醒我们：**提交消息与 `ablate.py` 的系列名并不总是一一对应**，后面写报告必须把“消息主题”和“脚本系列”分开记。

**与目录的关系**

- 根目录有 `repuls/` 与 `46_repulse/`，但目前还没把这一套完整指标表拉平。
- 这部分后续应继续补 `config.json` 与 `summary.json` 对照。

**当前已确认的真实结果**

虽然 `repuls/` 根目录本身还没直接抓到完整 `summary.json`，但相关落地目录已经能看到两条很重要的结果：

| 目录 | 配置摘要 | style | content | 观察 |
|------|----------|-------|---------|------|
| `46_repulse` | `w_repulsive=25`, `w_id=5`, `w_color=25`, patch=`3,5,7,11,15` | 0.6474 | 0.8251 | 更像“保内容的 repulse 接枝” |
| `Layer-Norm-repulse` | `repulsive_mode='l1'`, `w_repulsive=15`, `w_id=5`, `w_color=50`, patch=`3,5,15,19` | 0.6881 | 0.6529 | style 更高，但均衡性明显改变 |

这两条结果说明：

1. repulsive loss 并不是单向把 style 拉高，它会把 style/content 平衡重新推向两侧。
2. 它和基础结构、color loss、patch 组合强相关，不能单独看一个 `w_repulsive`。

---

### 3.10 2026-04-07 `62f2f383`：chess 系列

**代码直接证据**

- `SERIES_NAME="chess"`
- 基线文件: `Layer-Norm.json`
- 训练参数:
  - `batch_size=256`
  - `num_epochs=60`
  - `learning_rate=8e-5`
  - `warmup_ratio=0.1`

**实验矩阵**

| 实验名 | 关键改动 | 解释 |
|--------|----------|------|
| `01_baseline` | 不改 | 基线 |
| `02_no_pos_emb` | `ablation_disable_pos_emb=True` | 检查位置编码是否导致棋盘格/黑框 |
| `03_high_temp_attn` | `semantic_attn_temperature=0.5`, `style_attn_temperature=0.5` | 放松 attention |
| `04_no_color_highway` | `color_highway_gain=0` | 检查 color highway 是否是问题源 |
| `05_no_skip` | `skip_routing_mode='none'` | 验证 skip 是否制造伪影 |
| `06_patch_micro_only` | patch=`[3]` | 只保留微 patch |
| `07_patch_macro_only` | patch=`[25]` | 只保留宏 patch |
| `08_swd_sort_mode` | `swd_distance_mode='sort'` | 换 SWD 距离模式 |
| `09_no_pos_high_temp` | 同时去位置编码并升温 | 联合排查 |
| `10_macro_no_skip` | `skip_routing_mode='none'`, patch=`[25]` | 最激进排查项 |

**目录状态**

- 当前 `src/chess/` 下已能看到 `chess_01_baseline` 以及多个后续实验目录。
- 这说明 62f2 的 `ablate.py` 不只是纸面生成，而是已经开始实际执行。

**真实目录结果**

当前最稳定能读到的是 `epoch_0060_tokenized_distill_epochs50` 这一层。

| 目录 | 配置摘要 | style | content | 观察 |
|------|----------|-------|---------|------|
| `chess_01_baseline` | `skip=none`, `sem/sty temp=0.08`, `color_gain=0.3`, patch=`3,5,15,19`, `cdf` | 0.6855 | 0.7178 | baseline |
| `chess_02_no_pos_emb` | 去掉位置编码 | 0.6887 | 0.7185 | style/content 都略升，说明 pos emb 至少不是明显正收益 |
| `chess_03_high_temp_attn` | attention 温度升到 `0.5` | 0.6889 | 0.7208 | 与 no-pos 接近，偏正向但不剧烈 |
| `chess_04_no_color_highway` | `color_gain=0` | 0.6851 | 0.6669 | content 明显掉，说明 color highway 不是可有可无 |
| `chess_05_no_skip` | 目录存在，但配置里仍显示 `skip=none` | 0.6841 | 0.7201 | 需要继续核对是否基线本就无 skip |
| `chess_06_patch_micro_only` | patch=`3` | 0.6899 | 0.7544 | content 提升很大，说明微 patch 很保守 |
| `chess_07_patch_macro_only` | patch=`25` | 0.7024 | 0.6682 | style 最强，但 content 掉得明显 |
| `chess_08_swd_sort_mode` | `sort` 距离 | 0.6796 | 0.7588 | 明显转向 content 保持 |
| `chess_09_no_pos_high_temp` | 去 pos + 高温 | 0.6853 | 0.7178 | 联合修改没有进一步增益 |

**这一段目前最可信的结论**

1. `macro-only patch` 确实会推高 style，但副作用是 content 明显下滑。
2. `micro-only patch` 与 `sort mode` 都更偏内容保守。
3. `no_pos_emb` 没有造成灾难，甚至小幅改善，这很值得保留。
4. `no_color_highway` 的 content 掉得最明显之一，说明 color highway 在这条排障线里并不是装饰项。

**与提交说明的直接对应**

- 提交信息里写得非常清楚：`base01都出现了棋盘格和黑框，还是数值问题，改chess系列10个实验，看看是不是位置编码的问题`
- 所以这一版 `ablate.py` 的性质不是常规优化，而是**病理定位脚本**。

---

### 3.11 2026-04-08 当前工作树：Gate 系列

**代码直接证据**

- `SERIES_NAME="Gate"`
- 基线文件: `Layer-Norm.json`
- 训练固定:
  - `batch_size=256`
  - `num_epochs=120`
  - `learning_rate=8e-5`

**实验矩阵**

| 实验名 | patch / gate / loss 变化 |
|--------|--------------------------|
| `01_baseline` | 基线 |
| `02_weak_decoder` | `num_decoder_blocks=1` |
| `03_restore_skip_shortcut` | `skip_routing_mode='adaptive'`, `skip_fusion_mode='add_proj'` |
| `04_attn_gate_fixed` | `attn_gate_mode='fixed'` |
| `05_attn_gate_learned` | `attn_gate_mode='learned'` |
| `06_gate_learned_idt_energy` | learned gate + `idt_mode='energy'`, `w_identity=200` |
| `07_aux_loss_weak` | `w_aux_delta_variance=0.1` |
| `08_aux_loss_strong` | `w_aux_delta_variance=1.0` |
| `09_gate_and_bipolar` | learned gate + patch=`[3,25]` |
| `10_gate_and_low_color` | learned gate + `w_color=10`, `w_swd_unified=60` |
| `11_gate_bipolar_low_color` | gate + bipolar patch + low color |
| `12_gate_energy_bipolar_low_color` | gate + energy IDT + bipolar patch + low color |

**目录状态**

- 当前只看到 `src/Gate.bat` 与一批 `config_*.json`，还没有完整汇总后的 `Gate/` 根目录结果。
- 因此这部分目前是**待执行或执行中的计划矩阵**，不能和 `46/`、`Ablate43/` 一样视为已稳定产出的实验事实。

**可见的配置痕迹**

当前 `src/` 目录中已经存在这些与 Gate / attention / aux / bipolar 相关的配置文件：

- `config_04_attn_gate_fixed.json`
- `config_05_attn_gate_learned.json`
- `config_06_gate_learned_idt_energy.json`
- `config_07_aux_loss_weak.json`
- `config_08_aux_loss_strong.json`
- `config_09_gate_and_bipolar.json`
- `config_10_gate_and_low_color.json`
- `config_11_gate_bipolar_low_color.json`
- `config_12_gate_energy_bipolar_low_color.json`

这说明 Gate 线并不是空想阶段，至少配置层已经完整落地了，只是当前还缺统一整理后的结果目录。

---

## 4. 目录侧补充：freq 系列

`freq/` 这条线目前还没有完全钉死到某一版 `ablate.py`，所以这里先把它作为“实验目录事实”记录，而不是强行说它对应哪次脚本提交。

### 4.1 已确认的配置与结果

以下结果都来自 `freq/*/config.json` 与 `epoch_0080_tokenized_distill_epochs200/summary.json`：

| 目录 | patch | `w_color` | `w_id` | style | content | 观察 |
|------|-------|-----------|--------|-------|---------|------|
| `freq_01_conservative_baseline` | `1,3,5,7` | 50 | 5 | 0.6356 | 0.8681 | 保守基线 |
| `freq_02_brush_frenzy` | `1,3,5` | 20 | 5 | 0.6338 | 0.8694 | 降 color 没换来 style 提升 |
| `freq_03_large_view_awareness` | `1,3,7,15,25` | 30 | 5 | 0.6347 | 0.8724 | 更大视野主要推高 content |
| `freq_04_no_idt_abyss` | `1,3,5,11` | 40 | 0 | 0.6454 | 0.8751 | 去掉 IDT 反而 style/content 都高，值得重点标记 |
| `freq_05_idt_iron_fist` | `1,3,5,11` | 40 | 25 | 0.6392 | 0.8614 | 强 IDT 压 style 也压 content |
| `freq_06_yuv_dictatorship` | `1,3` | 150 | 5 | 0.6324 | 0.8736 | 极高 color 权重没带来 style 回报 |
| `freq_07_remove_blast_wall` | `1,3,5,9` | 60 | 5 | 0.6337 | 0.8675 | 调 color/patch 仍偏保守 |
| `freq_08_extreme_asymmetry` | `1,15,25` | 30 | 5 | 0.6366 | 0.8771 | 最偏不对称 patch，content 反而最高之一 |
| `freq_09_lancet` | `1,3,5,11,25` | 40 | 8 | 0.6353 | 0.8707 | 更像综合折中版 |

### 4.2 当前能读出的结论

1. `freq` 系列整体明显偏 content-first，不是冲 style 上限的路线。
2. `freq_04_no_idt_abyss` 很特别，`w_id=0` 后 style 和 content 都比强 IDT 版本好。
3. `freq_06_yuv_dictatorship` 说明只靠抬高 color 权重，并不能把 style 强拉起来。
4. 这条线更像是“频谱/patch/identity 关系研究”，而不是最终主线架构竞争者。

---

## 5. 目前可以确认的经验规律

这些规律是从 `ablate.py` 设计意图和已落盘结果共同支持的。

### 4.1 IDT 权重是最典型的 style/content 杠杆

- `Ablate43_L03_IDT_MassiveReturn` 中，`w_identity=20` 明显提高 content，压低 style。
- `46_03_decoder_usurpation` 与 `46_01_highway_cut` 也呈现类似“结构项偏向某一侧”的摆动。
- 所以 4 月的很多实验，本质是在寻找更细粒度的替代锚点，而不是继续无脑抬高 IDT。

### 4.2 patch 不是越大越好，也不是越小越好

- `Ablate43_P01_Patch_LargeOnly` 高于 baseline 一点，但不构成碾压。
- `Ablate43_P03_Patch_NanoClash` 也没有胜出。
- 45、46、Gate、chess 多条线都保留了宽窄混搭 patch，这不是偶然。

### 4.3 skip 与 decoder 的边界是 4 月上旬的主战场

- `dirty_skip`
- `decoder_usurpation`
- `no_skip`
- `restore_skip_shortcut`
- `weak_decoder`

这些名字已经说明，团队在反复确认：

1. 风格到底该由 decoder 主导，还是由 skip 携带。
2. skip 应该是“干净保结构”，还是“直接带入高频风格”。
3. 如果 skip 太脏，会不会直接引发棋盘格、黑框、语义污染。

### 4.4 attention 的问题不是“要不要”，而是“温度/位置编码/路由方式”

- 45 与 46 都出现 `style_attn_temperature` / `semantic_attn_temperature`
- chess 直接把 `pos_emb` 拉出来单独排病
- Gate 系列又引入 `attn_gate_mode`

这意味着 4 月的 attention 讨论已经不是 3 月那种“加不加 attention”，而是更工程化的：

- 温度多少
- 位置信息是否污染
- gate 固定还是可学习

---

## 6. 当前文档缺口与后续补法

### 5.1 已经可写实锤部分

- `Ablate43`：可直接写结果表
- `46`：可直接写结果表
- `chess`：可直接写设计意图与目录状态
- `Gate`：可直接写计划矩阵

### 5.2 还要继续挖的部分

- `45` 真实目录与结果
- `repuls` / `46_repulse` / `Layer-Norm-repulse`
- `freq` 系列对应的 `ablate.py` 版本链
- `97de1653` 的 `idt_schedule` 版本

### 5.3 特别提醒

- `Ablate43/` 与 `c8be1f3c` 的 12 组 `abl_*` 不是简单的同名映射，后续需要继续翻中间提交、生成脚本或目录重命名痕迹。
- 这类地方必须保留“不确定性”，不能为了好看强行写死。

---

## 7. 当前阶段结论

如果只看 2026-04-03 到 2026-04-09 这段，`ablate.py` 已经不只是“顺手生成 config 的小工具”，而是事实上的研究编排日志：

1. 先用 45 系列粗暴地试结构方向。
2. 再用 46 系列把 holy grail 附近的结构矛盾拆开。
3. 然后引出 repulsive / in-idt / chess / gate 这些更有针对性的支线。

也就是说，4 月上旬的模型史不能只看 `model.py`，必须把 `ablate.py` 当成“研究问题列表”。  
它记录的不是抽象参数，而是当时开发者脑中最真实的问题排序：

- 是不是 skip 脏了
- 是不是 decoder 抢权
- 是不是位置编码导致棋盘格
- 是不是 attention 温度不对
- 是不是 IDT 约束形态错了
- 是不是还需要新的 gate / aux loss / energy idt

这也是这份 `exp_schedule.md` 的核心用途：  
把模型演化史里的“为什么会做这些实验”重新接回第一手证据。

---

## 附记 A. 2026-04-09 对 `45` 段的修正

早先可以把 `45` 暂时写成“根目录聚合结果未直接看到”，但现在必须补充一条更准确的记法：

- 当前工作树里确实存在 `45` 的真实实验目录
- 只是它们不是规整地放在根目录 `45/45_*`
- 而是落在 `src/45/45/45_*`

对应目录如下：

- `src/45/45/45_01_golden_funnel`
- `src/45/45/45_02_naked_fusion`
- `src/45/45/45_03_macro_dictator`
- `src/45/45/45_04_micro_rebel`

而且这些目录同时具备：

- `config.json`
- `full_eval/epoch_0030/summary.json`

需要额外注意的是，`45` 使用的是老一版 summary 结构，关键字段位于：

- `analysis.style_transfer_ability`
- `analysis.identity_reconstruction`
- `analysis.photo_to_art_performance`

真实结果摘录如下：

| 目录 | style_transfer_style | style_transfer_content | photo_to_art_style | photo_to_art_content | 解释 |
|------|----------------------|------------------------|--------------------|----------------------|------|
| `45_01_golden_funnel` | 0.6674 | 0.5905 | 0.6596 | 0.6209 | 45 基线 |
| `45_02_naked_fusion` | 0.6669 | 0.5761 | 0.6532 | 0.5946 | 去 spatial prior 后内容下滑 |
| `45_03_macro_dictator` | 0.6137 | 0.5288 | 0.6166 | 0.5318 | macro-only 明显失利 |
| `45_04_micro_rebel` | 0.6704 | 0.5994 | 0.6592 | 0.6212 | micro 倾向反而最稳 |

因此，现在对 `45` 更准确的写法应该是：

- **它已有真实结果**
- **但格式较老，目录也更绕**
- **不能再把它视作“只在 ablate.py 里存在”的未落盘计划**

---

## 附记 B. 当前 `Gate` 的状态

当前 `src/ablate.py` 已经切换到 `SERIES_NAME="Gate"`，并且：

- `Gate.bat` 已存在
- `config_01_baseline.json` 到 `config_12_gate_energy_bipolar_low_color.json` 已存在

但根目录 `Gate/` 聚合结果尚未在当前工作树中看到。

所以 `Gate` 在时间线上应被记录为：

- **设计矩阵完整**
- **配置已生成**
- **运行脚本已写**
- **结果链暂未闭环**

详细台账见 `exp_Gate.md`。
