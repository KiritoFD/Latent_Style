# 02 论文数字的数据源（Results of Record）

状态说明：**verified** = 仓库内有原始 JSON/CSV 且数值一致；**paper-only** = 只在论文/笔记中，原始文件在远程机器或已丢失；
**corrected** = ICME 版已按原始文件更正。

## 数据源优先级

1. 仓库内原始评测输出（`*/results.json`、`per_epoch_metrics.csv`、`WEAVE/state/dino/*.json`、`exp_baselines/*/eval/*.json`）
2. `WEAVE/icme2027/data/main_table.csv`（主表，含出处列）
3. `WEAVE/icme2027/weave.tex` / `weave_supp.tex`
4. 其它 md 笔记（handoff、summary 等）——**只作线索**

## WEAVE 主模型

| 项 | 值 | 来源 | 状态 |
|---|---|---|---|
| 架构 | oriented target-HF route，1,037,087 可训练参数（+84M 冻结 VAE） | `WEAVE/experiments/architecture/hf_oriented_internal_early_stop.json` → `hf_oriented_internal_probe.json` → `hf_oriented_nohh.json` → `WEAVE/config.json` | verified（参数量见 supplement；`config.json` 单独构建的是 873,680 参数的 base 模型） |
| checkpoint | `runs/submission/hf_oriented_internal_early_stop/epoch_0004.pt` | 远程机器 | 不在仓库 |
| D5 主结果 | DINO-S 0.4918 / CLIP-S 0.7128 / LPIPS 0.2595 / DINO-C 0.8102（精确值 .491763/.712837/.259461/.810150） | `SchrodingerBridge/experiments/rebuttal_20260716/expD/results.json` → `D0_full` | verified |
| 同 checkpoint 逐 epoch 评测 | ep4 0.4917/0.7127/0.2595/0.8104 | `SchrodingerBridge/experiments/rebuttal_20260716/expA_seed42/per_epoch_metrics.csv` | verified（与上行差 <2e-4，属重评噪声） |
| 训练时间 | 4 epoch × 52 step，25.72+19.23+18.94+18.90 = 82.80 s（RTX 3060，batch 96） | supplement 训练日志表；原始 `training_*.csv` 在远程 | paper-only |
| 推理时间 | 126.0 s / 750 张（168 ms/张，8 步 + VAE 解码） | 仓库内 `WEAVE/docs/model_probe/generation_only_timing_summary.json` 只有其它 checkpoint（94.63 s、106.25 s） | paper-only |
| P2A-256 行 | 0.4801/0.6681/0.3116/0.8612 | 引用的 `exp/main_table/p2a_256/full_eval/epoch_0005/summary.json` 未入库；`WEAVE/state/dino/P2A-256__weave.json`（0.5100/0.8780）是旧 checkpoint | paper-only |
| R5 行 | 0.5226/0.7747/0.2895/0.7717 | 同上；`R5-WikiArt__weave.json`（0.4800/0.8157）是旧 checkpoint | paper-only |

### 容易混淆的其它 WEAVE 数字

| 数字 | 属于 | 出现位置 | 结论 |
|---|---|---|---|
| 0.4915/0.7126/0.2596/0.8103 | 同配置另一次 15-epoch 训练（nohh）的 epoch 4 | `WEAVE/docs/reproduction/*`、`SUBMISSION_HANDOFF` | 兄弟 run，不是论文行 |
| 0.4859/0.7075/0.2583/0.8287 | **旧 base 模型** `brk_a_ll03_10ep` ep10（873,680 参数，"WEAVE-q"） | `WEAVE/config.json::_main_table_metrics`、`aaai2027_v4/SUPPLEMENTARY_MATERIAL.md`、`docs/delivery/DELIVERY_SUMMARY.md`、`SchrodingerBridge/aaai2027_v4/paper.tex` | 过期，勿引用 |
| 0.4867/0.7074/0.2508/0.8280 | base 模型 15-epoch 复现，epoch 6 | `WEAVE/README.md`、`WEAVE/docs/reproduction/baseline_epoch_metrics.csv` | base 模型参考点，不是论文模型 |
| DINO-S 0.49173 / DINO-C 0.7782 | DINO sidecar 对某 WEAVE checkpoint 的评测 | `WEAVE/state/dino/D5-512__weave.json` | DINO-C 与主结果不一致，checkpoint 未标注 |
| "903K params / 3.08 min" | 更早的 base 架构 | `WEAVE/docs/79/main_table_v4_remote.csv`、旧架构图 PNG | 过期 |

## 主表（Table I）其它方法

逐格出处见 `WEAVE/icme2027/data/main_table.csv` 的 `source_dino` / `source_clip_lpips` / `note` 列。汇总：

| 列 | verified 来源 | paper-only |
|---|---|---|
| DINO-S / DINO-C | `WEAVE/state/dino/*.json`（除 Latent-WCT、WEAVE P2A/R5 外全部一致）；StyTR-2/AesPA：`WEAVE/state/dino/{stytr2,aespa}_*.json` | Latent-WCT（仓库内 `*__wct.json` 是另一个 VGG-WCT，数值 0.136/0.025） |
| CLIP-S / LPIPS | StyTR-2、AesPA：`WEAVE/exp_baselines/*/*/eval/clip_lpips_summary.json`；D5 的 Identity/SD-Turbo/StyleID/CUT/SaMST/SaMam/Seedream：`WEAVE/docs/baseline/README.md`（指向远程 `unified_results.json`） | 其余所有 P2A/R5 格、StyleAligned/Z-STAR/StyleShot/Latent-WCT 全部格（唯一机读副本是 `make_radar_metric_blocks.py` 中从论文抄的 dict） |
| Params / Train / Infer | — | 大多为 paper-only；Z-STAR 3 h 为步数外推估计；CUT/SaMST/SaMam 时间标注为"prior value, not re-measured"（见 [04](04_baselines.md)） |

### ICME 版已做的更正（corrected）

| 格 | AAAI 版 | 原始文件 | ICME 版 |
|---|---|---|---|
| Z-STAR P2A DINO-S / DINO-C | 0.514 / 0.526（与其 R5 值重复，抄错） | `WEAVE/state/dino/P2A-256__zstar.json`：0.4975 / 0.5522 | 0.498 / 0.552 |
| StyleID R5 DINO-S / DINO-C | 0.571 / 0.374 | `R5-WikiArt__styleid.json`：0.5512 / 0.3854 | 0.551 / 0.385 |
| † / ‡ 标记 | 手工，两处与定义不符 | 严格按 IDT/TGT 不等式 | 自动计算 |
| Z-STAR 推理时间 | "3 h"（未说明） | `make_radar*.py` 注释：步数外推估计 | "∼3 h" + 表注说明 |
| R5 描述 | "20-family R5 benchmark" | `WEAVE/docs/79/README.md` | 5 个随机 WikiArt 家族 |
| 消融表 caption | "retrained variants use the same fixed-latent selection rule" | `SchrodingerBridge/experiments/rebuttal_20260716/expA_D*/oracle_regret.json`：表中数字是 oracle（最佳 DINO-S）epoch，内部规则选的是 ep5/ep5/ep4 | 改为"reported at their best-DINO-S epoch"，regret 放入 supplement |

## 消融表（Table II）

| 行 | 数值 | 来源 | 状态 |
|---|---|---|---|
| WEAVE | 0.4918/0.7128/0.2595/0.8102 | `SchrodingerBridge/experiments/rebuttal_20260716/expD/results.json::D0_full` | verified |
| w/o stepwise AdaIN（推理 β=0，修正后） | 0.4821/0.7005/0.2258/0.8455 | `SchrodingerBridge/experiments/rebuttal_20260716/results/d1_adain_corrected.json`（20/20 张 PNG 哈希不同） | verified；`expD/results.json::D1_adain0`（0.4917，"无影响"）是**配置未生效的错误 run**，`batch1_summary.md` 仍引用它 |
| w/o HF conditioning | 0.4825/0.7157/0.2837/0.8125 | `SchrodingerBridge/experiments/rebuttal_20260716/expD/results.json::D2_no_hf_route` | verified |
| direct endpoint | 0.4894/0.7172/0.3138/0.7966（ep13） | `SchrodingerBridge/experiments/rebuttal_20260716/expA_D4_seed42/per_epoch_metrics.csv` | verified |
| λ_LL=1.0 | 0.4910/0.7180/0.2626/0.7874（ep3） | `SchrodingerBridge/experiments/rebuttal_20260716/expA_D3_seed42/per_epoch_metrics.csv` | verified |
| learned HH head | 0.4930/0.7164/0.2670/0.8061（ep4） | `SchrodingerBridge/experiments/rebuttal_20260716/expA_D5_seed42/per_epoch_metrics.csv` | verified |

## 其它正文数字

| 数字 | 来源 | 状态 |
|---|---|---|
| seeds 42/7/123 → epoch 4/4/3，DINO-S 0.4918/0.4910/0.4862 | `WEAVE/docs/reproduction/internal_dynamics_robustness.csv`；`SchrodingerBridge/experiments/rebuttal_20260716/expA_seed*/` | verified |
| 参考池 margin m=8: 0.0298 [0.0204,0.0394]；m=16: 0.0319 [0.0225,0.0380]；逐请求正比例 ≈69% | `SchrodingerBridge/experiments/rebuttal_20260716/results/b1_reference_pool_corrected.json` | verified（`expB1/` 是修正前版本） |
| Other5：WEAVE 0.5313/0.7742/0.2802/0.7587，SaMam、SaMST | 仅论文 | paper-only |
| SD1.5 插件：600 对 0.3974→0.4080 DINO-S 等；750 对；4.9 ms/张 | 仅论文 / supplement | paper-only |
| 频率探针（Fig. 2） | `WEAVE/aaai2027_v4/fig_data/method_probe_*.csv`、`method_probes.json`、`swd_loss_separability.json` | verified |
| ArtFID（supplement） | `canonical_artfid.json` | verified |
| 深度/边缘（supplement） | `SchrodingerBridge/rebuttal_exps/experiments/rebuttal_20260716/task3_topological/task3_summary.json` | verified |

## ICME 版正文中的派生数字（2026-10-01）

| 数字 | 计算方式 | 来源 |
|---|---|---|
| LL 占梯度能量 69.5%（LH 10.3 / HL 11.3 / HH 8.9） | 5,000 对跨风格 D5 latent 的直接位移 Haar 频带能量均值之比；正交 Haar 下等于初始化时的梯度能量占比 | `WEAVE/icme2027/data/probe_frequency.csv`（= `aaai2027_v4/fig_data/method_probe_frequency.csv`） |
| 风格可分性 LL 0.12 / LH 0.19 / HL 0.17 / HH 0.56 | between/within 方差比 | `WEAVE/icme2027/data/probe_separability.csv` |
| 源锚定端点下 LL 占比 ≤ 17.0% | 命题 1：每对样本 ‖u_ℓ‖² ≤ α²‖ℓ_s−ℓ_c‖²（α=0.3），高频位移不变；0.09·4.241/(0.09·4.241+0.628+0.687+0.546) | `WEAVE/icme2027/tools/make_figures.py` 打印 |
| 三个种子 DINO-S 0.4897 ± 0.0030，CLIP-S 0.7137 ± 0.0008，LPIPS 0.2605 ± 0.0059，DINO-C 0.8073 ± 0.0031 | seed 42 用 expD 主结果，seed 7/123 用 `internal_dynamics_robustness.csv` 中的选中 epoch | 见上文"其它正文数字" |
| 全参考池 margin 0.033 | `b1_reference_pool_corrected.json` 中 m30 的 margin.mean = 0.0334 | 同左 |
| "复制输入的 CLIP-S 0.693 高于 SaMam 0.582、SaMST 0.618" | D5 Table I | `main_table.csv` |
| D5、R5 上 WEAVE 在四组风格×内容指标上均帕累托最优；P2A 上 SaMam 四项全优、Seedream 风格与 LPIPS 优 | 对 12 个对比方法逐组检查支配关系 | `main_table.csv` |
| 256 px 时每个 Haar 子带 16×16 系数/通道 | 256/8（VAE 下采样）/2（一级 Haar） | 架构事实 |
| SaMam 参数 8.5×、训练 >300× | 8.8/1.04；436 min / 1.38 min | Table I |

**评测协议补充**：`WEAVE/utils/run_evaluation.py`（约 3440–3480 行）推理时对每个目标风格取该风格测试目录的第一张图编码为参考 latent，
因此 WEAVE 的风格参考就是 TGT 那张图，且该图在 DINO-S 参考池内。参考池重采样（m=8 时包含该图的概率 8/30）显示 margin 不依赖它。
