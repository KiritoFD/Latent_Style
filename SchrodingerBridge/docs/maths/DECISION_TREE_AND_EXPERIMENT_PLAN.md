# Sequential clip_style 决策树实验方案

更新日期：`2026-05-19`。

这版方案不再做“先跑完整网格，再事后筛选”。实验必须逐个推进：

```text
train one experiment for 8 epochs
-> eval epoch 4/6/8 immediately
-> record all metrics
-> pick that experiment's best epoch
-> update global history
-> choose the next experiment from the decision tree
```

总预算先设为 `16` 组。每组固定 `batch_size=64`、`num_epochs=8`、`save_interval=1`。

## 1. 当前锚点

参考目录：

```text
S-add__K-1_C-0_W-20_Col-0/full_eval/epoch_0008/summary.json
```

参考指标：

```text
clip_style_all    = 0.7167235834
content_lpips_all = 0.4615265376
clip_content_all  = 0.7977139172
```

最近 batch=64 refactor 分支：

```text
exp/refactor_clean_batch64_e10_fix/full_eval/epoch_0008/summary.json
clip_style_all    = 0.7128111729
content_lpips_all = 0.4613536712
```

## 2. 新增理论钩子：Anti-Grayness Protocol

核心病理假设：`clip_style` 上不去，不只是均值分布没贴近目标风格，也可能是投影后的方差偏低，导致输出发灰、对比度不足、笔触不够极端。

因此 semantic SWD 从：

```text
L_sem = E |sort(<theta, z_pred>) - sort(<theta, z_style>)|
```

扩展为：

```text
L_sem_var = L_sem
          + w_variance_penalty
            * E |Var(<theta, z_pred>) - Var(<theta, z_style>)|
```

这个项只在 `w_variance_penalty > 0` 时启用。它不是恢复旧的 color / low-frequency / NCE 分支，而是在当前有效的 semantic SWD 内部补上对比度约束。

## 3. 固定实验常数

根节点和后续分支默认使用：

```text
swd_num_projections = 32
semantic_swd_num_projections = 32
semantic_attn_temperature = 0.04
semantic_attn_routing_mode = softmax
retired heuristic losses = absent / ignored
```

`ot_cost.py` 中的 CDF soft-SWD 慢路径已经移除，即使旧 config 写着 `swd_distance_mode=cdf`，实际也走 `torch.sort` 版一维 Wasserstein。

## 4. 决策规则

每个实验评估 `epoch_0004`、`epoch_0006`、`epoch_0008`。该实验的 best epoch 由以下 score 选出：

```text
score = 100 * (clip_style_all - baseline_style)
      - 25  * max(0, content_lpips_all - baseline_lpips)
      - time_penalty
      - collapse_penalty
```

标签：

```text
win:
  clip_style_all >= 0.72
  content_lpips_all < 0.45

high_style_borderline:
  clip_style_all >= 0.72
  content_lpips_all <= 0.46

target:
  clip_style_all >= baseline_style
  content_lpips_all <= baseline_lpips + 0.01

promising:
  clip_style_all >= 0.718
  content_lpips_all <= 0.48

collapse:
  content_lpips_all >= 0.53
  or clip_content_all <= 0.74
```

## 5. 16 组自适应树

### Stage 0：根节点确认

第一组：

```text
s00_root_sort32_temp004
w_kinetic = 1.0
terminal_swd_weight = 20
w_variance_penalty = 0
residual_gain = 1.0
```

目的：确认 fast sort SWD + projection=32 + temp=0.04 后，谱系没有跑偏。

### Stage 1：方差突破

先跑：

```text
s01_var1_res115_kin125_swd25
w_variance_penalty = 1.0
residual_gain = 1.15
w_kinetic = 1.25
terminal_swd_weight = 25
```

随后根据结果选择：

- style 未到 `0.72`：继续 `w_variance_penalty = 3 / 5 / 7.5`；
- style 到 `0.72` 且 LPIPS `< 0.45`：进入 confirmation；
- style 到 `0.72` 但 LPIPS `> 0.46`：进入 kinetic compensation。

### Stage 2：动能装甲补偿

如果方差惩罚拿到 style 但撕裂结构，固定当前最好的 variance 强度，测试：

```text
w_kinetic = 1.5,  terminal_swd_weight = 35
w_kinetic = 1.75, terminal_swd_weight = 40
w_kinetic = 2.0,  terminal_swd_weight = 40
```

目的：用更大的运动惩罚把 LPIPS 拉回 `0.45` 附近，同时保留 semantic SWD 的高方差风格。

### Stage 3：端点压力 / 残差幅度

如果 style 长期停在 `0.718` 以下，说明方差项不够，需要继续推 endpoint pressure 或 delivered amplitude：

```text
residual_gain = 1.20
terminal_swd_weight = 30 / 35
w_kinetic = 1.0
w_variance_penalty = 5 / 7.5
```

### Stage 4：温度修复

如果前面仍不能平衡，脚本会用固定温度作为退火代理：

```text
semantic_attn_temperature = 0.06
semantic_attn_temperature = 0.03
```

真正的 epoch-wise annealing 可以后续再进 `trainer.py`，但第一轮 16 组先用固定温度判断 routing 温度是否值得继续投资。

### Stage 5：胜利确认

如果出现 `win`：

```text
seed = 43
projection = 64
```

至少做两个确认实验，避免把一次随机波动误判为可发表结果。

## 6. 执行入口

启动：

```bat
G:\GitHub\Latent_Style\SchrodingerBridge\run_clip_style_decision_tree.bat
```

调试：

```bat
G:\GitHub\Latent_Style\SchrodingerBridge\run_clip_style_decision_tree.bat --dry-run --max-experiments 3
```

输出：

```text
exp/decision_tree_clip_style/decision_tree_results.csv
exp/decision_tree_clip_style/decision_tree_best.csv
exp/decision_tree_clip_style/decision_tree_ledger.jsonl
configs/decision_tree_clip_style/*.json
```

`decision_tree_results.csv` 记录每个实验的所有 eval epoch；`decision_tree_best.csv` 记录每个实验的 best epoch；`decision_tree_ledger.jsonl` 按实验顺序记录候选配置、best epoch 和当时的 global best。

## 7. Legacy A-grid 导入

`2026-05-19` 检查了上一版 batch-grid 的 A 分支，用户确认最后启动过的命令是：

```text
python src/run.py --config configs/decision_tree_clip_style/A_kin0.55_swd24.json
```

这对应旧网格中的前 `14` 组：

```text
A_kin1_swd20
A_kin1_swd24
A_kin1_swd28
A_kin1_swd32
A_kin0.85_swd20
A_kin0.85_swd24
A_kin0.85_swd28
A_kin0.85_swd32
A_kin0.7_swd20
A_kin0.7_swd24
A_kin0.7_swd28
A_kin0.7_swd32
A_kin0.55_swd20
A_kin0.55_swd24
```

当前工作区检查结果：这些 run 目录下没有可 eval 的 `epoch_0004.pt`、`epoch_0006.pt` 或 `epoch_0008.pt`，因此没有写入指标结论。runner 已加入导入命令：

```bat
python run_clip_style_decision_tree.py --import-legacy-a-until A_kin0.55_swd24 --max-experiments 0
```

如果这些 checkpoint 从别处恢复到：

```text
exp/decision_tree_clip_style/A_kin*/epoch_0004.pt
exp/decision_tree_clip_style/A_kin*/epoch_0006.pt
exp/decision_tree_clip_style/A_kin*/epoch_0008.pt
```

再次运行上面的导入命令即可自动 eval、汇总并让 sequential 决策树从这些历史结果之后继续。

## 8. DINO 结构复评：2026-05-19

用户观察新图后指出：内容物体几乎没变，LPIPS 很低并不代表风格转移成功。于是停止 sequential runner，改用 DINOv2 penultimate-patch self-similarity distance 复评结构变化。

指标定义：

```text
dino_structure = MSE(SSM_DINO(gen), SSM_DINO(src))
```

方向：越低代表生成图和源内容的 DINO patch 自相似结构越接近，也就是物体/布局越没动。

输出文件：

```text
exp/decision_tree_clip_style/dino_structure_compare_with_samst.csv
```

同口径 SaMST 对照使用：

```text
Related_Works/baseline_pipeline/results/samst/protocol_a_800
```

### 8.1 结果表

| Run | DINO-structure ↓ | CLIP-style ↑ | CLIP-content ↑ | LPIPS ↓ | 结论 |
|---|---:|---:|---:|---:|---|
| `s21_temp_var0p0_temp0p03_e0008` | 0.01729 | 0.70762 | 0.80483 | 0.44756 | 结构最稳，但风格不足 |
| `s00_root_sort32_temp004_e0006` | 0.01833 | 0.71163 | 0.79793 | 0.45951 | 根节点偏保守 |
| `s30_confirm_seed43_s00_root_sort32_temp004_e0008` | 0.01862 | 0.71296 | 0.80314 | 0.45660 | seed 确认仍偏保守 |
| `s20_temp_var0p0_temp0p06_e0008` | 0.01965 | 0.71090 | 0.78749 | 0.46301 | 温度变软没有带来风格 |
| `SaMST_protocol_a_800` | 0.02012 | 0.72531 | 0.77519 | 0.53904 | 风格强，但结构/LPIPS 代价明显 |
| `s21_temp_var1p0_temp0p03_e0008` | 0.03244 | 0.71877 | 0.69930 | 0.57038 | 方差路线开始结构崩 |
| `s12_comp_var1p0_kin2p0_swd40_e0008` | 0.03311 | 0.72431 | 0.70214 | 0.57060 | 风格接近 SaMST，但结构崩 |
| `s20_temp_var1p0_temp0p06_e0008` | 0.03379 | 0.72655 | 0.70136 | 0.57538 | raw style 最高之一，但不可用 |
| `s01_var1p0_res115_kin125_swd25_e0008` | 0.03497 | 0.72465 | 0.69208 | 0.57797 | anti-grayness 直接撕裂结构 |
| `s10_comp_var1p0_kin1p5_swd35_e0008` | 0.03567 | 0.72416 | 0.68129 | 0.58938 | kinetic compensation 没救回结构 |
| `s11_comp_var1p0_kin1p75_swd40_e0008` | 0.03587 | 0.72642 | 0.68838 | 0.58545 | 更强 kinetic 仍不可用 |

### 8.2 结论

DINO 复评支持用户的视觉判断：

1. 低 LPIPS / 低 DINO 结构距离的模型基本没有充分改变内容物体的视觉结构，因此风格也偏低。
2. `w_variance_penalty=1` 可以把 CLIP-style 推到 `0.724+`，但 DINO-structure 从约 `0.018` 跳到 `0.033-0.036`，LPIPS 也到 `0.57+`，说明它不是“反发灰成功”，而是把结构撕裂了。
3. SaMST 在这组 protocol 下是 `style=0.7253, dino_structure=0.0201, LPIPS=0.5390`。我们的保守分支结构更稳，但 style 明显不足；我们的 variance 分支 style 接近/超过 SaMST，但结构比 SaMST 更坏。

### 8.3 决策树更新

立即停止：

```text
w_variance_penalty >= 1 的主线搜索
继续 kinetic compensation 去救 variance 分支
把低 LPIPS 当成成功信号
```

下一轮目标改为：

```text
在 dino_structure <= 0.0205 附近，把 CLIP-style 从 0.711-0.713 提到至少 0.718
```

新的 guard：

```text
usable:
  dino_structure <= 0.0205
  content_lpips <= 0.48
  clip_content >= 0.775

style frontier:
  clip_style >= 0.718
  且满足 usable

collapse:
  dino_structure >= 0.030
  or content_lpips >= 0.53
  or clip_content <= 0.74
```

下一步不应继续加 terminal/variance 暴力项，而应回到“风格注入机制”本身：例如 style spatial prior / cross-attention 温度调度 / residual amplitude 的更细粒度推理 sweep，并且每一步必须用 DINO guard。
