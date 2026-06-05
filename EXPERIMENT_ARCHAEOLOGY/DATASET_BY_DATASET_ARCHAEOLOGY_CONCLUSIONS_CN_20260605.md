# Dataset-by-Dataset Archaeology Conclusions - 2026-06-05

本文件是对 `EXPERIMENT_ARCHAEOLOGY/final_by_dataset/*.csv` 的逐文件人工复核结论。

这不是递归脚本扫描结论。脚本只用于前期聚合和计数，本轮已经把 25 个 split CSV 逐个打开，检查了字段、代表行、`source_kind`、`validity_class`、metric/timing 分布和代表 `source_path`。结论写在本文和同目录 CSV：

- `EXPERIMENT_ARCHAEOLOGY/dataset_by_dataset_archaeology_conclusions_20260605.csv`

## 口径

- 缺失 timing 不补造，空值保持空。
- 训练时间保留原始单位，不强转秒。
- `infer_time_value` 里有些是 full-eval wall time，不等同于纯模型生成/单图推理。
- `metric_evidence`、`summary_evidence`、`log_evidence`、`timing_evidence`、`indexed_curated_evidence` 不混为一种证据。
- 本轮没有新增删除 whitelist；所有清理结论都是“保留/待裁决/需另开 exact-path proof”，不是删除执行。

## 25 个已逐项打开的 split

| dataset_key | rows | metric | train timing | infer timing | 当前等级 |
| --- | ---: | ---: | ---: | ---: | --- |
| `cycle_nce` | 11794 | 10590 | 0 | 0 | historical metric surface |
| `schrodingerbridge_exp_general` | 4051 | 3239 | 12 | 36 | mixed current/historical LANCET surface |
| `schrodingerbridge_weight_sweep` | 1285 | 800 | 11 | 0 | parameter sweep surface |
| `legacy_style_transfer_experiments` | 1120 | 0 | 0 | 0 | remote legacy context only |
| `schrodingerbridge_grid_search` | 1013 | 544 | 0 | 0 | historical grid metric surface |
| `schrodingerbridge_vae_backend` | 699 | 81 | 0 | 0 | backend probe surface |
| `schrodingerbridge_frontier` | 692 | 600 | 0 | 0 | frontier probe metric surface |
| `schrodingerbridge_representation_probe` | 567 | 471 | 0 | 0 | representation probe surface |
| `distinct5_512` | 417 | 278 | 55 | 113 | formal candidate surface |
| `wikiart512_5style` | 200 | 144 | 14 | 10 | formal candidate surface |
| `schrodingerbridge_root_legacy` | 197 | 64 | 0 | 14 | root legacy anchor surface |
| `legacy256_overfit50` | 131 | 123 | 15 | 5 | legacy baseline with timing |
| `run511_5domain` | 120 | 3 | 40 | 25 | Related_Works timing surface |
| `schrodingerbridge_aaai2027` | 87 | 0 | 0 | 49 | remote full-eval timing surface |
| `strict_protocol_750` | 79 | 53 | 15 | 24 | formal protocol surface |
| `path_family_final_works` | 75 | 75 | 0 | 0 | historical path-family metrics |
| `photo_monet_5x5` | 42 | 0 | 0 | 2 | qualitative and summary surface |
| `schrodingerbridge_docs_experiments` | 13 | 0 | 0 | 0 | docs index only |
| `schrodingerbridge_destructive_ablation` | 12 | 12 | 12 | 12 | clean ablation metric/timing surface |
| `unclassified_curated_experiments` | 11 | 0 | 0 | 0 | unclassified remote context |
| `schrodingerbridge_review_additional` | 10 | 0 | 0 | 5 | review additional timing surface |
| `related_works_baselines` | 5 | 0 | 0 | 0 | remote baseline summary context |
| `path_family_run_summary.csv` | 4 | 0 | 2 | 2 | path status timing only |
| `s2wat` | 3 | 0 | 0 | 0 | S2WAT index only |
| `path_family_step_count_sweep` | 2 | 0 | 0 | 2 | path status timing only |

## 可优先提升的结果层

这些 split 可以继续进入后续 timing promotion 或主结果候选整理，但仍要去重和保留口径：

- `distinct5_512`：当前最重要的 512 Distinct5 证据面。方法包括 LANCET/LBM、SaMAM、LANCET、LBM、No-op、SaMST、idt。已经有 278 条 metric、55 条 train timing、113 条 infer/full-eval timing。下一步是去重、区分 full-eval wall 与纯推理、挑出唯一标定 rows。
- `wikiart512_5style`：WikiArt512-5style formal/context bridge。`local_wsl_wikiart512_hist_b32_e8` 是重要 anchor。已有 144 条 metric、14 条 train timing、10 条 infer timing。
- `strict_protocol_750`：跨方法 strict-750 协议层。包含 LANCET/LBM、StyleID、SaMST、AdaIN 系列等，已有 53 条 metric、15 条 train timing、24 条 infer timing。适合继续做统一标定，但必须拆出 baseline 方法、LANCET ablation 和重复 summary。
- `schrodingerbridge_destructive_ablation`：12 行干净 ablation 表，每行都有 clip/content/train/infer。适合进入 timing candidates 或 appendix，但要先确认这些 ablation 是否仍属于论文主张。
- `schrodingerbridge_exp_general`：大规模 LANCET/LBM 证据面，含 current summary、ablation registry、remote logs。可用于实验脉络，但正式主张必须回到具体 run。

## 逐项结论

### `cycle_nce`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/cycle_nce.csv`

结论：这是历史 Cycle-NCE/AdaCUT/IDT/AdaIN 证据面，不是当前 LANCET formal claim。10590 条 metric 主要来自 `Cycle-NCE/Ablate43/.../grid_metrics.csv`，远程侧补充了 summary/log。方法字段大量为空，说明后续还要按目录名重新归类。

Timing：无训练/推理时间可复用。

清理边界：不能按历史性、文件量、扩展名或目录名批量删除。Cycle-NCE 清理必须继续使用 exact path -> policy -> ledger -> post-delete verify。

### `schrodingerbridge_exp_general`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/schrodingerbridge_exp_general.csv`

结论：这是 LANCET/LBM 最大的混合证据面，覆盖 local analysis CSV、destructive ablation registry、remote summary/log。它能说明 2026-05 后主线搜索脉络，但不能直接作为最终性能表。

Timing：有 12 条 train timing、36 条 infer timing。样本来自 `SchrodingerBridge/exp/ablation_destructive_7epoch/destructive_ablation_7epoch_registry.csv`，例如 D0/D4/D5/D9 等以秒为单位的 train/infer 记录。

清理边界：`SchrodingerBridge/exp` 是 mixed evidence surface，任何 ckpt thinning 都要保留 formal/current anchors。

### `schrodingerbridge_weight_sweep`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/schrodingerbridge_weight_sweep.csv`

结论：这是 kinetic/SWD/lambda sweep 面，可支持参数选择脉络。metric 主要来自 `review_additional_experiments/.../lambda_grid/eval/.../batch_summary.csv`。

Timing：11 条 train timing 来自 `lambda_grid/status.csv`，其中有 `0.000/0.001 s` 级记录。这是状态/调度记录，不能误当训练成本。

清理边界：`review_additional_experiments` 与同名 RAR 仍需 archive proof，不能因路径重复直接删除。

### `legacy_style_transfer_experiments`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/legacy_style_transfer_experiments.csv`

结论：全部来自远程 curated summary/log，是 2026-02 到 2026-04 的早期 legacy style-transfer 上下文。它没有 metric/timing 字段，不能用于当前性能主张。

Timing：无。

清理边界：这些远程 experiment 目录只能在 summary/log 已完整入库且 owner 同意后逐路径清理。

### `schrodingerbridge_grid_search`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/schrodingerbridge_grid_search.csv`

结论：这是 S/K/C/W/Col 结构搜索的历史 metric surface。544 条 metric 来自 old experiment dirs 的 `batch_summary.csv` 和 viewer CSV，远程补充 summary/log。

Timing：无。

清理边界：`archives/old_experiment_dirs/grid_search_3epoch` 是 archive evidence，不能在迁移指标和 anchor 前删除。

### `schrodingerbridge_vae_backend`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/schrodingerbridge_vae_backend.csv`

结论：这是 VAE backend、scale decode、256 MSE controls 的 probe surface。81 条本地 metric 可以用于选择脉络，其余主要是 remote summary/log。

Timing：无。

清理边界：VAE backend runs 是否删除，要先确认当前 config、论文叙述或后续 eval 是否引用。

### `schrodingerbridge_frontier`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/schrodingerbridge_frontier.csv`

结论：这是 frontier decision tree 和 dynamic metric probe 的探索层。600 条 metric 来自 `frontier_decision_tree_8h/.../metrics_reuse_generated.csv`。

Timing：无。

清理边界：frontier 目录不能按“探索实验”直接删。先挑出支撑后续设计的 rows，其余只保留索引后再议。

### `schrodingerbridge_representation_probe`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/schrodingerbridge_representation_probe.csv`

结论：这是 representation/actuator/edge/early-cotrain probe surface。文件首部就包含 `_codex_tmp/probe_style_representation_smoke/eval_metrics.csv`，说明里面混有 smoke/audit-only 数据，不能直接作为正式性能表。

Timing：无。

清理边界：`_codex_tmp` 和 probe 输出不能按名字删除；先确认是否被 docs 或后续设计引用。

### `distinct5_512`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/distinct5_512.csv`

结论：这是当前最重要的 Distinct5-512 formal candidate。来源覆盖 docs comparison CSV、ArtFID keypoints、remote summary/log、AAAI2027 experiment log 和 text timing regex。它支持后续标定，但同一结果可能被多个 CSV 重复引用，必须去重。

Timing：55 条 train timing、113 条 infer timing。训练单位有分钟样本，例如 `1.2 m`，保持原单位。推理列里有 full-eval wall 或 summary wall，不能直接写成纯推理。

清理边界：Distinct5 的 SaMAM diag、No-op、SaMST、LBM/LANCET anchor 都不能整体删除；要先定义代表样本和 summary 留存策略。

### `wikiart512_5style`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/wikiart512_5style.csv`

结论：这是 WikiArt512-5style formal/context bridge。`SchrodingerBridge/exp/local_wsl_wikiart512_hist_b32_e8/full_eval_epoch_0008_b2_opt_nocls/summary.json` 是关键 source。

Timing：14 条 train timing、10 条 infer timing。训练有 `0.7 m` 等原始单位；full-eval wall time 要同纯推理拆分。

清理边界：`local_wsl_wikiart512_hist_b32_e8` 与相关 full_eval/cache/output 是 anchor，不删。

### `schrodingerbridge_root_legacy`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/schrodingerbridge_root_legacy.csv`

结论：这是 root legacy anchor，包括 `S-add__K-1_C-0_W-20_Col-0` 和 theory switch validation。可作为 baseline gate 和历史解释，不是新实验主结果。

Timing：14 条 infer timing，无 train timing。

清理边界：root legacy anchor 保留。旧 theory_switch_validation 需要先迁移指标和引用关系。

### `legacy256_overfit50`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/legacy256_overfit50.csv`

结论：这是 Legacy256/overfit50 历史 baseline。包含 LANCET、SaMAM、No-op、SaMST、idt 等方法。它有指标和少量 timing，但分辨率为 256，不能和 512 formal 表直接混用。

Timing：15 条 train timing、5 条 infer timing，作为历史成本上下文。

清理边界：old experiment dirs 保留到指标、summary、timing 已完整迁移。

### `run511_5domain`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/run511_5domain.csv`

结论：这是 Related_Works 的 5-domain timing/protocol 面，包含 StyTr2、CAST、AdaIN、AesPA-Net、AesFA、SaMST、S2WAT 等。质量 metric 只有 3 条，不是完整性能表。

Timing：40 条 train timing、25 条 infer timing。代表来源是 `Related_Works/results/metrics_summary/timing_summary.csv`。其中有“visual output invalid”的 note，说明 timing 成功不等于输出有效。

清理边界：`Related_Works/run_511/outputs` 是协议证据，不能按无 ckpt 或图片目录直接删。

### `schrodingerbridge_aaai2027`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/schrodingerbridge_aaai2027.csv`

结论：这是远程 AAAI2027/TokenizerClean closing surface。目前 87 行里没有 metric rows，主要是 remote summary/log。

Timing：49 条 infer/full-eval wall time，例如 `aaai2027_longer_train_f_seed42_b44_e8/full_eval/epoch_0001..0008/summary.json`。无 train timing。

清理边界：远程 AAAI2027 exp 不能删 weight/media，除非 owner 给 archive/migration policy。

### `strict_protocol_750`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/strict_protocol_750.csv`

结论：这是最接近统一 benchmark 的 strict-750 protocol surface。包含 LANCET/LBM、StyleID、SaMST、AdaIN 系列、ablation 方法等。它是后续 timing promotion 的核心候选。

Timing：15 条 train timing、24 条 infer timing。来源包括 `training_times_documentation.md`、`Related_Works/run_511/complete_750/*` 和 ablation summary。

清理边界：`complete_750` 和 ablation anchors 不能删。先完成去重和代表输出保留。

### `path_family_final_works`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/path_family_final_works.csv`

结论：这是 `final_works/trial_0016/0019/0044` 的历史路径族 metric。method 为空，不能作为未命名主方法宣传。

Timing：无。

清理边界：final_works 需要补 method/run lineage 后再决定是否迁移或清理。

### `photo_monet_5x5`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/photo_monet_5x5.csv`

结论：这是 CUT、SDEdit、CycleGAN、S2WAT、SD-Turbo、LANCET/LBM 在 Photo/Monet 5x5 相关的 qualitative/repro inventory 面。很多 source 是 `web/images` 或 `infer_5x5/images`，不是 metrics。

Timing：只有 2 条 infer timing，无 train timing。

清理边界：CUT `infer_5x5` 和 checkpoint web images 需要 owner 决定代表样本保留或迁移，不能直接删。

### `schrodingerbridge_docs_experiments`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/schrodingerbridge_docs_experiments.csv`

结论：这是 docs selected metrics 的索引面，列出 AdaIN、Ours、SaMST、StyleID 等被文档选入的方法。字段本身没有 metric/timing，必须回连 raw/eval source。

Timing：无。

清理边界：docs/experiments 是索引证据，不删。

### `schrodingerbridge_destructive_ablation`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/schrodingerbridge_destructive_ablation.csv`

结论：这是 12 行干净 destructive ablation 表。每行都有方法、variant、clip/content、train_time、infer_time，来源是 `ablation_destructive_7epoch/D*/full_eval/epoch_0007`。

Timing：12 条 train timing、12 条 infer timing，单位为秒。

清理边界：`ablation_destructive_7epoch` 是 anchor。不要删 summary/epoch anchors。

### `unclassified_curated_experiments`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/unclassified_curated_experiments.csv`

结论：这是尚未归类的 remote curated evidence，包含 StarGAN 和 TokenizerClean final_works trial 的 summary/log。不能用于主结果。

Timing：无。

清理边界：不删。下一步要归回 `related_works_baselines` 或 `path_family_final_works`。

### `schrodingerbridge_review_additional`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/schrodingerbridge_review_additional.csv`

结论：这是 review_additional step-count sweep timing surface。只有 10 行，5 行 status，5 行 remote summary。

Timing：5 条 infer timing。

清理边界：`review_additional_experiments.rar` 不能删，因为远程当前缺少可靠 RAR listing proof。

### `related_works_baselines`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/related_works_baselines.csv`

结论：这是远程 StarGAN/S2WAT baseline summary context。只有 5 行 summary evidence，没有 metric/timing。

Timing：无。

清理边界：remote Related_Works baseline 先保留，直到 source repo/output 已迁移。

### `path_family_run_summary.csv`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/path_family_run_summary.csv.csv`

结论：这是 root `run_summary.csv` 的 timing/status 面，不是方法性能结果。

Timing：2 条 train timing、2 条 infer timing。

清理边界：保留 `run_summary.csv`，先查明对应真实 run。

### `s2wat`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/s2wat.csv`

结论：这是 S2WAT 本地索引/selected metrics 引用。没有 metric/timing，不能支撑完整 S2WAT 性能或成本表。

Timing：无。

清理边界：S2WAT images 等代表样本/summary 确认后再讨论清理。

### `path_family_step_count_sweep`

打开文件：`EXPERIMENT_ARCHAEOLOGY/final_by_dataset/path_family_step_count_sweep.csv`

结论：这是 step-count status timing 记录，不能作为质量表。

Timing：2 条 infer timing。

清理边界：保留 `step_count_sweep/status.csv`，后续回连到 review_additional step_count_sweep。

## 当前缺口

- `distinct5_512`、`wikiart512_5style`、`strict_protocol_750`、`schrodingerbridge_destructive_ablation` 需要做 timing promotion 和 duplicate collapse。
- `cycle_nce`、`legacy_style_transfer_experiments`、`path_family_final_works`、`unclassified_curated_experiments` 有大量 method 为空或未归类，需要按目录名和真实方法回填。
- `run511_5domain`、`photo_monet_5x5`、`s2wat` 的 Related_Works baseline 多为 timing/index/qualitative，需要补真实 metric 或明确为 qualitative-only。
- `schrodingerbridge_aaai2027` 目前主要是 full-eval wall time，缺质量指标。
- `review_additional_experiments.rar` 仍缺 archive listing proof，不能删。
- TokenizerClean 26 个 cited/current media dirs 仍需 owner archive/migration 决策。

## 清理判断

本轮 dataset-by-dataset 复核没有产生新的删除 whitelist。原因是：这些 split 多数仍在支撑指标、timing、历史解释、baseline/context 或 owner 待裁决 media。后续删除必须继续沿用：

```text
exact path -> policy CSV/MD -> deletion ledger -> post-delete verification
```

不能用“历史实验”“无指标”“图片很多”“路径重复”“hash 相同”作为独立删除理由。
