# Latent_Style 实验考古直接结论 - 2026-06-05

这份文件是当前人工考古工作的总入口，直接回答四个问题：本地是什么，远程是什么，实验脉络是什么，还差什么。它不把脚本扫描当最终结论；扫描只用于导航，结论必须能回到打开过的目录、summary、metrics、training log、policy CSV、cleanup ledger 或 post-delete verification。

## 当前状态

当前不能宣布全仓最终完成。已经完成的是主要实验面、主要权重/媒体清理面、远程主树、远程 TokenizerClean、timing overlay 和多轮 cleanup ledger 的人工分层；仍缺若干 owner-level 和 source-open 决策。

本轮以后，`EXPERIMENT_ARCHAEOLOGY` 是干净提交状态；论文 tex/pdf、paper figure、源码、Related_Works 脚本等用户已有脏文件没有被 stage 或回滚。

## 本地结论

本地根：`G:\GitHub\Latent_Style`

本地不是一个单纯源码仓库，而是混合了论文工作区、当前 SchrodingerBridge/LANCET 实验、Related_Works baseline、Cycle-NCE 历史包、数据集、latent/feature/eval cache、root exp/archive/tmp、以及考古输出目录。

### 本地必须保留的主证据

| 区域 | 当前结论 | 证据入口 | 保留原因 |
| --- | --- | --- | --- |
| `SchrodingerBridge/exp/local_wsl_wikiart512_hist_b32_e8` | 当前 WikiArt512/Distinct5 训练和 full-eval timing anchor | `manual_check_session_20260605_current.csv` | 有 `config.json`、8 个 epoch 权重、training CSV、`TRAIN_WALL_SECONDS=53.19`、750 样本 full eval summary。 |
| `SchrodingerBridge/S-add__K-1_C-0_W-20_Col-0/full_eval/epoch_0008` | 历史 S-add baseline anchor | `manual_check_session_20260605_current.csv` | 有历史 `summary.json` 和 `metrics.csv`，不能当缓存删除。 |
| `Cycle-NCE` | 历史大指标面和早期模型演化证据 | `MANUAL_CYCLE_NCE_ARCHAEOLOGY_20260605.md` | 有家族实验、summary、aggregate CSV、报告、eval cache、源码和媒体。 |
| `Related_Works` | baseline/reproduction/results 面 | `manual_related_works_directory_ledger_20260605.csv` | 包含 CUT/SaMAM/SaMST/S2WAT/Seedream 等 baseline 结果、日志、summary 和 metrics。 |
| `Dataset`, `style_data`, `latent-256`, `clip-feats-vitb32`, `eval_cache`, `SchrodingerBridge/scale` | 数据和 eval 依赖面 | `MANUAL_LOCAL_DATASET_CACHE_POLICY_20260605.md`, `MANUAL_LOCAL_EVAL_CACHE_POLICY_20260605.md` | 是数据、latent tensor、feature tensor、CLIP/ArtFID/VAE/cache 依赖，不是 checkpoint 垃圾。 |

### 本地已经清理的内容

| 清理块 | 数量 | 释放空间 | ledger |
| --- | ---: | ---: | --- |
| 非主线 checkpoint-like 文件 | 875 files | 46032.053 MB | `cleanup/manual_deleted_checkpoints_20260605.csv` |
| 本地 remaining surface exact targets | 5 targets | 237.860 MB | `cleanup/manual_local_remaining_surface_cleanup_20260605.csv` |
| CUT video 中间帧 work dirs | 5 dirs | 3068.463 MB | `cleanup/manual_local_generated_media_intermediate_frame_cleanup_20260605.csv` |

本地清理原则：只删白名单，且必须有 policy、per-target ledger、post-delete verify。不能按 `.pt/.ckpt/.tar/.zip/.rar` 或图片扩展名批量删。

### 本地还没完成的点

- 不是所有 nested generated-image directory 都完成了 owner-level review。
- `seedream_gap`、`inference_param_sweep_*`、部分 baseline raw outputs 已打开但仍需要 owner 决策。
- archive-like 文件只能按 provenance 处理，不能按扩展名或大小处理。

## 远程主树结论

远程入口：`ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62`

远程主树：`I:\Github\Latent_Style`

`I:` 不为空。实际存在 `I:\Github\Latent_Style`、`I:\Github\Latent_Style_TokenizerClean`、数据/latent/cache 根和 remote archaeology/curated 输出。不能用浅层 `I:` 列表替代手检。

### 远程主树必须保留的主证据

| 区域 | 当前结论 | 证据入口 | 保留原因 |
| --- | --- | --- | --- |
| `I:\Github\Latent_Style\SchrodingerBridge\exp` | 当前 Distinct5/SADD/phase-space 证据面 | `MANUAL_REMOTE_SCHRODINGERBRIDGE_EPOCH_THINNING_20260605.md` | thinning 后剩 17 个 anchor/probe 权重；不是整体可删 exp。 |
| SaMAM central `step_checkpoints` | SaMAM 曲线证据 | `MANUAL_REMOTE_SAMAM_CHECKPOINT_THINNING_20260605.md` | 删除了 redundant `last*.ckpt` aliases，保留 12 个 step curve checkpoints。 |
| `data/style_data/latent*/eval_cache` | 数据、latent、eval dependency 面 | `MANUAL_REMOTE_MAIN_DATA_CACHE_ARCHIVE_POLICY_20260605.md` | failed/stale residue 已删，完整 cache/data roots 保留。 |
| `Cycle-NCE\45.rar` | 唯一历史证据包 | `MANUAL_REMOTE_CYCLE_NCE_45_RAR_REVIEW_20260605.md` | archive 内有 configs、summaries、metrics CSV、6008 张 eval images、ma-probe 和权重，不能整包删。 |
| `experiments` expanded dir | resolved duplicate RAR 的保留体 | `MANUAL_REMOTE_EXPERIMENTS_RAR_RESOLVED_POLICY_20260605.md` | `experiments.rar` 已删，expanded `experiments` 和 HF symlink target blobs 保留。 |

### 远程主树已经清理的内容

| 清理块 | 数量 | 释放空间 | ledger |
| --- | ---: | ---: | --- |
| SchrodingerBridge 非保留 epoch checkpoint | 84 files | 4961.604 MB | `cleanup/manual_remote_schrodingerbridge_epoch_cleanup_20260605.csv` |
| SaMAM redundant `last*.ckpt` alias | 7 files | 1931.291 MB | `cleanup/manual_remote_samam_alias_cleanup_20260605.csv` |
| data/cache/archive residue | 11 targets | 381.807 MB | `cleanup/manual_remote_main_data_cache_archive_residue_cleanup_20260605.csv` |
| duplicate/stale archives | 3 files | 3290.714 MB | `cleanup/manual_remote_duplicate_archive_cleanup_20260605.csv` |
| weight-only RAR archives | 6 files | 6553.384 MB | `cleanup/manual_remote_rar_weight_only_archive_cleanup_20260605.csv` |
| resolved duplicate `experiments.rar` | 1 file | 8091.026 MB | `cleanup/manual_remote_experiments_rar_resolved_duplicate_cleanup_20260605.csv` |

远程主树当前硬缺口：cross-cache dedup hash audit 还没做；`45.rar` 如果要删，必须先提取/确认非权重证据包。

## 远程 TokenizerClean 结论

远程 TokenizerClean 主路径：`I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp`

已经对 145 个 exp 顶层目录建立 internal evidence 和 citation graph，并做了多轮 no-summary、uncited、media owner review。

### TokenizerClean 当前结论

| 区域 | 当前结论 | 证据入口 | 决策 |
| --- | --- | --- | --- |
| 145 个 exp 顶层目录 | 已经索引，分 cited/current/no-summary/probe/media 等类 | `MANUAL_REMOTE_TOKENIZERCLEAN_CITATION_GRAPH_20260605.md` | cited/current/formal packet 保留。 |
| uncited summary-backed checkpoints | 非主线 exploratory weights | `cleanup/manual_remote_tokenizerclean_uncited_checkpoint_cleanup_20260605.csv` | 已删 141 个，5198.991 MB。 |
| no-summary probe/calibration weights | probe 或 calibration 负担 | `MANUAL_REMOTE_TOKENIZERCLEAN_NO_SUMMARY_REVIEW_20260605.md` | 已删 18 个，362.391 MB。 |
| retained no-summary orphan probes | 纯 orphan probe weights | `MANUAL_REMOTE_TOKENIZERCLEAN_RETAINED_NO_SUMMARY_OWNER_REVIEW_20260605.md` | 已删 14 个 target，170.017 MB。 |
| uncited generated media | zero-hit summary-backed media | `MANUAL_REMOTE_TOKENIZERCLEAN_GENERATED_MEDIA_PRUNE_20260605.md` | 已删 43008 个，11883.246 MB。 |
| 7 个 trained no-summary payload | 真训练 payload，不是垃圾 | `MANUAL_REMOTE_TOKENIZERCLEAN_TRAINED_NO_SUMMARY_THIRD_PASS_20260605.md`, `MANUAL_REMOTE_TOKENIZERCLEAN_NO_SUMMARY_RECOVERY_PASS_20260605.md` | 保留，等 in-dir summary recovery 或 owner decision。 |

7 个 trained no-summary payload 已逐个远程打开。每个都有 `config.json`、`logs/training_*.csv`、`src`、`numeric_debug.jsonl` 和 checkpoint，但 `summary_like_count=0`。后续远程受限搜索发现 2 个外部间接证据：`wikiart_distinct5_ema_lancet_spectralstat_e2_b80` 是 downstream resume source，`wikiart512_ema_trueint_stylepush_tsw40_kin025_e1_b48` 有 diagnostics summary；其余 5 个仍是 training-log-only。正确结论仍是保留待决策，不能删。

## 实验脉络

| 阶段 | 时间 | 主线 | 当前解释 |
| --- | --- | --- | --- |
| Phase A | 2026-02 到 2026-03 | legacy/no-edge/style-transfer early experiments | 早期 sanity 和失败探索，不支撑当前 formal claim。 |
| Phase B | 2026-03 到 2026-04 | legacy256, StyleID, IDT, tokenized/no-tokenized | baseline 和 sanity 面，timing 混杂，需要质量标签。 |
| Phase C | 2026-04 到 2026-05 | Cycle-NCE / Latent AdaCUT | 历史大指标面，保留 metrics/summary/ref cache。 |
| Phase D | 2026-05 | SchrodingerBridge/LANCET phase-space | grid/search/frontier/vae_backend/representation，主要是探索面。 |
| Phase E | 2026-05-30 到 2026-06-02 | WikiArt512 和 Distinct5 formal evidence | 当前 timing/efficiency claim 的核心证据面。 |
| Phase F | 2026-06-03 起 | AAAI2027 / TokenizerClean claim closing | flow-loss, SA-SWD, tokenizer execution, time-to-parity，仍需 review-grade 整理。 |

论文 claim 只能引用 Phase E/F 中有 full_eval、summary、training/eval timing、source-open 的行。dry-run、quick_eval、failed probe、runtime anomaly、placeholder checkpoint 只能做 audit 或 negative evidence。

## Timing 结论

| 文件 | 行数 | 结论 |
| --- | ---: | --- |
| `SchrodingerBridge/docs/timing/training_inference_timing_master.csv` | 419 | docs timing master，未在本轮修改。 |
| `EXPERIMENT_ARCHAEOLOGY/timing_quality_master_20260605.csv` | 1093 | archaeology timing overlay，有质量标签。 |
| `timing_candidate_claim_reconciliation_20260605.csv` | 53 | overlay 中可候选支持 claim 的 rows。 |

规范化 source path 后，53 个 candidate claim rows 中 27 个已经在 docs timing master，26 个不在。docs master 419 行中只有 49 行被 overlay 覆盖，370 行没有 overlay quality label。结论是：两个表不能互相替代。后续 paper-facing timing 表必须逐条 source-open，不要直接引用全部 1093 行。

原始单位保留，训练时间没有强行换秒；缺失 train/infer 值保持空。

## 清理总账

只按手动白名单 ledger 计入，不含候选表、旧脚本表或按类汇总的重复计数。当前可核释放空间约 `92162.847 MB`，其中：

- 本地清理约 `49338.376 MB`。
- 远程主树清理约 `24670.021 MB`。
- 远程 TokenizerClean 清理约 `17614.039 MB`。

对应总账入口：`cleanup/CLEANUP_AUDIT_SUMMARY.md`。

## 8 小时级继续计划

| block | 预算 | 目标 | 产物 |
| --- | ---: | --- | --- |
| 1 | 1.0h | 本地 remaining generated-media 继续 owner review | media owner policy CSV/MD，可能的 whitelist cleanup。 |
| 2 | 1.0h | TokenizerClean cited/current media archive/migration policy | cited/current media policy 和可选迁移/清理清单。 |
| 3 | 1.0h | 7 个 trained no-summary payload summary recovery 或 owner decision | recovery/decision CSV。 |
| 4 | 1.0h | local/remote cross-cache dedup hash audit | cache dedup CSV/MD，不先删除。 |
| 5 | 1.0h | 26 个 timing candidate missing docs 的 source-open pass | source-open timing promotion table。 |
| 6 | 1.0h | dataset split、timeline、README counts consistency pass | consistency audit 和 README 修正。 |
| 7 | 1.0h | `45.rar` curated nonweight extraction policy | extraction/retention/delete decision policy。 |
| 8 | 1.0h | 最终 requirement-by-requirement completion audit | completion audit CSV/MD。 |

## 当前不能宣布完成的原因

- “每一个目录”仍未完成严格 owner-level 逐目录结论，尤其是 nested generated media。
- 7 个 TokenizerClean trained no-summary payload 仍没有 in-dir summary 或 owner 决策；其中 2 个已有外部间接证据，5 个仍是 training-log-only。
- cross-cache dedup hash audit 未完成。
- docs timing master 和 overlay 只做了 sidecar reconciliation，未形成最终 paper-facing timing 表。
- `Cycle-NCE\45.rar` 保留为唯一历史证据包，如要删除必须先做 curated extraction。
