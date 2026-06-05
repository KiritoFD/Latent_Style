# Latent_Style 实验考古权威总报告 - 2026-06-05

本报告是当前 `EXPERIMENT_ARCHAEOLOGY` 的可读入口。它不把脚本扫描当作最终结论；脚本只用于导航，结论必须能追溯到手动打开过的目录、日志、summary、config、CSV、policy 或 cleanup ledger。

当前结论：任务还没有达到“全仓每个嵌套目录都 owner-level 完成”的最终状态；已经完成的是本地和远程主实验面的分层索引、训练/推理 timing 证据质量分层、多个高价值 checkpoint/media/archive 清理块，以及每次删除对应的 CSV ledger 和 post-delete 验证。

## 1. 当前证据入口

核心索引：

- `manual_top_level_directory_index_20260605.csv`: 本地和远程顶层目录索引，67 行，其中本地 `G:` 32 行、远程 `I:` 35 行。
- `manual_coverage_matrix_20260605.csv`: 覆盖矩阵，41 行，其中本地 26 行、远程主树 12 行、远程 TokenizerClean 3 行。
- `manual_conclusion_index_20260605.csv`: 全局、本地、远程主树、远程 TokenizerClean、timing、lineage 六类结论入口。
- `manual_goal_completion_audit_20260605.csv`: 需求级完成度审计，明确当前仍为 `not complete`。
- `manual_cleanup_retention_and_next_candidates_20260605.csv`: 剩余权重/cache/依赖/归档类别、保留理由和下一步候选。

主要深挖文档：

- 本地：`MANUAL_LOCAL_DATASET_CACHE_POLICY_20260605.md`, `MANUAL_LOCAL_EVAL_CACHE_POLICY_20260605.md`, `MANUAL_LOCAL_ROOT_MISC_POLICY_20260605.md`, `MANUAL_LOCAL_REMAINING_SURFACE_POLICY_20260605.md`, `MANUAL_CYCLE_NCE_ARCHAEOLOGY_20260605.md`
- 远程主树：`MANUAL_REMOTE_SCHRODINGERBRIDGE_EXP_20260605.md`, `MANUAL_REMOTE_SCHRODINGERBRIDGE_EPOCH_THINNING_20260605.md`, `MANUAL_REMOTE_SAMAM_CHECKPOINT_THINNING_20260605.md`, `MANUAL_REMOTE_MAIN_DATA_CACHE_ARCHIVE_POLICY_20260605.md`, `MANUAL_REMOTE_ARCHIVE_PROVENANCE_20260605.md`
- 远程 TokenizerClean：`MANUAL_REMOTE_TOKENIZERCLEAN_CITATION_GRAPH_20260605.md`, `MANUAL_REMOTE_TOKENIZERCLEAN_NO_SUMMARY_REVIEW_20260605.md`, `MANUAL_REMOTE_TOKENIZERCLEAN_GENERATED_MEDIA_PRUNE_20260605.md`
- Timing：`TIMING_EVIDENCE_QUALITY_PASS_20260605.md`, `timing_quality_master_20260605.csv`, `timing_quality_summary_20260605.csv`

## 2. 本地 G:\GitHub\Latent_Style 结论

本地不是“没看”，而是已经拆成多个手动 ledger。当前可证明覆盖包括顶层目录、`SchrodingerBridge/exp` 家族、`Related_Works` 家族、`Cycle-NCE` 家族、数据/latent/feature cache、root `eval_cache`、root archive/tmp/exp，以及最新的 remaining surface 手检。

### 2.1 本地目录分层判断

| 区域 | 当前判断 | 证据 |
| --- | --- | --- |
| `SchrodingerBridge/exp` | 当前 formal/timing evidence 和少量历史 anchor；非主线 probe/calibration weights 已经按 policy 清理或保留 | `manual_schrodingerbridge_exp_topdir_ledger_20260605.csv` |
| `SchrodingerBridge/exp/local_wsl_wikiart512_hist_b32_e8` | WikiArt512 formal full-eval/timing anchor，保留 8 个 epoch 权重 | `manual_cleanup_retention_and_next_candidates_20260605.csv` |
| `SchrodingerBridge/docs/experiments` | 当前实验文档和 evidence pack 入口；不碰 tex/pdf | `manual_coverage_matrix_20260605.csv` |
| `Related_Works` | baseline/repro/metrics 面；主要保留依赖权重、结果、summary、tiny fake eval placeholders | `manual_related_works_directory_ledger_20260605.csv` |
| `Cycle-NCE` | 历史大指标面；保留 metrics/summary/ref cache；不按 checkpoint 垃圾目录处理 | `MANUAL_CYCLE_NCE_ARCHAEOLOGY_20260605.md` |
| `Dataset`, `style_data`, `latent-256`, `clip-feats-vitb32`, `SchrodingerBridge/scale` | 数据、latent tensor、feature tensor、VAE/data backend；不作为 checkpoint cleanup 目标 | `MANUAL_LOCAL_DATASET_CACHE_POLICY_20260605.md` |
| root `eval_cache` | ArtFID/CLIP/VAE/DINO/reference feature cache；只删除坏下载和空 temp | `MANUAL_LOCAL_EVAL_CACHE_POLICY_20260605.md` |
| root `archive`, `tmp`, root `exp` | duplicate archive、stale launcher residue、空 probe residue 已处理；paper/PDF/TEX/PNG scratch 不动 | `MANUAL_LOCAL_ROOT_MISC_POLICY_20260605.md` |
| local remaining surface | 逐项检查 archive/cache/lock/empty-dir；删除仅限 5 个白名单目标 | `MANUAL_LOCAL_REMAINING_SURFACE_POLICY_20260605.md` |

### 2.2 本地已清理

| 清理块 | 删除内容 | 释放空间 | Ledger |
| --- | --- | ---: | --- |
| 本地人工 checkpoint cleanup | 875 个非主线 checkpoint-like 文件 | 46032.053 MB | `cleanup/manual_deleted_checkpoints_20260605.csv` |
| root misc cleanup | duplicate `Cycle-NCE.tar`、stale launcher residue、空 probe | 1503.203 MB | `cleanup/manual_root_misc_cleanup_20260605.csv` |
| root eval_cache cleanup | invalid `.incomplete` HF blob、空 ModelScope temp | 55.994 MB | `cleanup/manual_cache_cleanup_20260605.csv` |
| dataset/cache cleanup | failed `wikiart_81k` HF cache residue | 63.948 MB | `cleanup/manual_dataset_cache_cleanup_20260605.csv` |
| local remaining surface cleanup | 2 个空目录、2 个 fully duplicated zip、1 个 fully duplicated output tar | 237.860 MB | `cleanup/manual_local_remaining_surface_cleanup_20260605.csv` |

### 2.3 本地明确保留

不能按名字或大小继续删除的本地对象：

- `latent-256`, `clip-feats-vitb32`, `Dataset`, `style_data`, `SchrodingerBridge/scale`: 数据/latent/feature backend。
- root `eval_cache`: CLIP、ArtFID、VAE、DINO、reference feature 等评测依赖。
- `Related_Works` 的 VGG/Inception/LPIPS 等 baseline/eval dependency。
- `SchrodingerBridge/exp/local_wsl_wikiart512_hist_b32_e8`: 当前 WikiArt512 timing anchor。
- `Cycle-NCE`: 历史指标、summary、reference cache 和 archive provenance 待定面。
- `Related_Works\runs\lbm_train_wds_smoke_photo_to_monet\train-000000.tar` 和 `val-000000.tar`: WebDataset shard，未找到同路径展开副本。
- `exp\highres_eval_local\samst_ckpts_epoch50.tar`: checkpoint archive，5 个模型条目未找到展开同路径副本。
- `Related_Works\repos\ArtBank\clip\bpe_simple_vocab_16e6.txt.gz`: CLIP vocabulary dependency。
- `Related_Works\repos\AdaIN-style-official\.git\shallow.lock`: 外部 repo 的 `.git` 内部 lock，且存在 git/GitHubDesktop 进程，不作为实验 payload 删除。
- `Cycle-NCE\uv.lock` 和 `Cycle-NCE\src\uv.lock`: uv dependency lock，不是 stale temp lock。

## 3. 远程主树 I:\Github\Latent_Style 结论

远程入口：`ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62`

远程主树已完成三类重点手检：`SchrodingerBridge/exp` epoch checkpoint thinning、SaMAM central curve alias cleanup、data/cache/archive residue 和 proven duplicate archive cleanup。

### 3.1 Remote `SchrodingerBridge/exp`

| 状态 | 文件数 | 大小 | 证据 |
| --- | ---: | ---: | --- |
| 清理前 | 101 checkpoint | 5945.063 MB | `manual_remote_schrodingerbridge_epoch_evidence_20260605.csv` |
| 已删除 | 84 `.pt` | 4961.604 MB | `cleanup/manual_remote_schrodingerbridge_epoch_cleanup_20260605.csv` |
| 剩余 | 17 checkpoint | 983.457 MB | `manual_remote_schrodingerbridge_remaining_weights_after_thinning_20260605.csv` |

保留原则：

- path-stability probe 的 base/k000/k025 `epoch_0001`。
- Distinct5 formal ablation 的 cited/best/anchor epoch。
- K/L/M 的单点 anchor。
- SADD exact/repro 的 e7/e8，因为 full_eval summary 锁定这些后段点。

删除原则：

- F-longer/K-longer 的非保留中间 epoch。
- rejected A/J ablation checkpoint。
- SADD e1-e6 中间 checkpoint。
- negative evidence 的 summary/metrics/log/grid 保留，只删 checkpoint。

### 3.2 Remote SaMAM

路径：
`I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag\step_checkpoints`

| 状态 | 文件数 | 大小 | 证据 |
| --- | ---: | ---: | --- |
| 清理前 | 19 ckpt | 约 5242 MB | `manual_remote_samam_checkpoint_thinning_policy_20260605.csv` |
| 已删除 | 7 `last*.ckpt` aliases | 1931.291 MB | `cleanup/manual_remote_samam_alias_cleanup_20260605.csv` |
| 剩余 | 12 `step-step=*.ckpt` | 3310.776 MB | `manual_remote_samam_remaining_step_checkpoints_after_alias_cleanup_20260605.csv` |

关键判断：

- whole-file SHA 不同，所以不能靠普通文件 hash 直接删。
- PyTorch metadata/state-dict hash 证明 `last*.ckpt` 是 paired step checkpoint 的模型重复。
- paired step 文件保留 optimizer/scheduler，是更完整的 curve/restart evidence。
- 12 个 step checkpoint 仍保留，不再继续删，除非明确放弃完整 curve/restart 能力。

### 3.3 Remote main data/cache/archive

已完成 residue-only 和 proven duplicate/stale archive cleanup：

| 清理块 | 删除内容 | 释放空间 | Ledger |
| --- | --- | ---: | --- |
| remote data/cache/archive residue | failed `.incomplete`、stale locks/tmp、空 temp dirs | 381.807 MB | `cleanup/manual_remote_main_data_cache_archive_residue_cleanup_20260605.csv` |
| remote duplicate/stale archives | stale `eval_cache.zip`、legacy checkpoint archive zip、exact duplicate archive | 3290.714 MB | `cleanup/manual_remote_duplicate_archive_cleanup_20260605.csv` |
| remote RAR weight-only archives | `Gate.rar`, `Attn_48.part*.rar`, `chess.part*.rar`，非权重条目全部已在展开目录同名同大小存在，archive 唯一 payload 是旧 checkpoint/tokenizer 权重 | 6553.384 MB | `cleanup/manual_remote_rar_weight_only_archive_cleanup_20260605.csv` |

保留原则：

- `data`, `style_data`, `latents*`, `latent-256*`: 数据/latent backend，未发现 bad markers。
- `eval_cache`: 完整 manual CLIP、offline pairing、ArtFID、VAE cache 保留。
- `SchrodingerBridge\scale\datasets`: dataset bodies 和 latent split 保留。
- `Related_Works\repos`: baseline repos 和 dependency weights 保留。
- `Cycle-NCE` 和 `experiments`: RAR/legacy archive 只在 provenance 证明后删除，不能按扩展名或大小删除。

仍未完成：

- RAR provenance 已完成一轮：`Gate.rar`, `Attn_48.part*.rar`, `chess.part*.rar` 已删除；仍保留 `experiments.rar` 和 `Cycle-NCE\45.rar`。
- complete cache 是否跨目录重复仍需 hash audit。
- legacy `experiments` 的全部 nested family 还没 owner-level 完成。

## 4. 远程 TokenizerClean 结论

路径：
`I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp`

当前判断：这不是垃圾目录，而是 AAAI2027/tokenizer claim closing 工作树。只能 citation-aware 清理，不能 mass delete。

### 4.1 已完成覆盖

- 覆盖 `exp` 全部 145 个 top-level 目录。
- 建立 docs/reviews/master/paper citation graph。
- 打开 no-summary checkpoint dirs 的 config/log/training CSV tails。
- 建立 generated media inventory 和 cleanup policy。

主要证据：

- `manual_remote_tokenizerclean_exp_internal_evidence_after_no_summary_cleanup_20260605.csv`
- `manual_remote_tokenizerclean_exp_citation_graph_all_20260605.csv`
- `manual_remote_tokenizerclean_no_summary_review_20260605.csv`
- `manual_remote_tokenizerclean_generated_media_inventory_after_cleanup_20260605.csv`

### 4.2 已清理

| 清理块 | 删除内容 | 释放空间 | Ledger |
| --- | --- | ---: | --- |
| uncited summary-backed checkpoint cleanup | 141 个 checkpoint files | 5198.991 MB | `cleanup/manual_remote_tokenizerclean_uncited_checkpoint_cleanup_20260605.csv` |
| no-summary probe checkpoint cleanup | 18 个 probe/calibration checkpoint files | 362.391 MB | `cleanup/manual_remote_tokenizerclean_no_summary_probe_checkpoint_cleanup_20260605.csv` |
| retained no-summary orphan probe cleanup | 11 个纯 orphan probe weight files 和 3 个空目录 | 170.017 MB | `cleanup/manual_remote_tokenizerclean_orphan_probe_weight_cleanup_20260605.csv` |
| uncited generated media cleanup | 43008 个 zero-hit summary-backed media files | 11883.246 MB | `cleanup/manual_remote_tokenizerclean_uncited_generated_media_cleanup_20260605.csv` |

### 4.3 仍保留

- checkpoint：29 个目录仍有 156 个 weight-like 文件，合计 5643.952 MB。
- generated media：26 个目录仍有 46483 个 media 文件，合计 7501.518 MB。
- 保留原因：docs/paper/master 命中、current `aaai2027_*` packet、trained no-summary payload 需 owner review/summary recovery、或 media 需要 archive/migration policy。

未完成：

- 7 个 trained no-summary payload dirs 仍需 owner review 或补 summary。
- cited/current media 需要 archive/migration policy。
- `aaai2027_*` formal packet 只能 packet-specific thinning，不能按空间压力直接删。

## 5. 实验脉络

当前仓库按六阶段理解：

| 阶段 | 时间 | 主线 | 当前读法 |
| --- | --- | --- | --- |
| Phase A | 2026-02 到 2026-03 | legacy/no-edge/style-transfer 早期实验 | 历史脉络和 sanity，不作为当前 claim |
| Phase B | 2026-03 到 2026-04 | legacy256、StyleID、IDT、no-tokenized/tokenized | baseline 和 sanity check，timing 混杂 |
| Phase C | 2026-04 到 2026-05 | Cycle-NCE / Latent AdaCUT | 历史大指标面，保留 metrics/summary/ref cache |
| Phase D | 2026-05 | SchrodingerBridge/LANCET phase-space | grid/search/frontier/vae_backend/representation，多数是探索面 |
| Phase E | 2026-05-30 到 2026-06-02 | WikiArt512 和 Distinct5 formal evidence | 当前 timing/efficiency claim 的核心证据面 |
| Phase F | 2026-06-03 起 | AAAI2027 / TokenizerClean claim closing | flow-loss、SA-SWD、tokenizer execution、time-to-parity，仍需 review-grade 整理 |

当前不能把 legacy dry-run、failed probe、runtime-anomalous row、quick_eval row 直接提升为 formal timing claim。

## 6. Timing 结论

当前 timing 证据质量分层：

| quality_class | claim_use | rows |
| --- | --- | ---: |
| `full_eval_summary_wall_time_tokenizerclean` | audit_full_eval_wall_time_only | 744 |
| `quick_eval_or_probe_wall_time` | audit_full_eval_wall_time_only | 234 |
| `full_eval_wall_time` | candidate_claim_support_with_caveat | 51 |
| `historical_timing_context` | historical_context | 28 |
| `partial_training_or_missing_eval` | audit_only | 20 |
| `smoke_or_failed_probe` | exclude_formal_claim | 7 |
| `invalidated_or_negative_audit_only` | audit_only | 4 |
| `training_log_only` | audit_training_cost_only | 2 |
| `train_and_eval_wall_time` | candidate_claim_support_with_caveat | 2 |
| `runtime_anomalous_exclude_speed_claim` | quality_only_or_anomaly | 1 |

可复用的当前 timing 入口：

- `manual_timing_evidence_20260605.csv`
- `manual_remote_tokenizerclean_timing_evidence_20260605.csv`
- `timing_quality_master_20260605.csv`
- `timing_quality_summary_20260605.csv`
- `TIMING_EVIDENCE_QUALITY_PASS_20260605.md`

关键读法：

- WikiArt512 LANCET/LBM full eval 有本地 wall-time anchor，约 210 秒级。
- Distinct5 LBM formal retained points 是分钟级训练证据。
- SaMAM Distinct5-512 step 3000 有小时级训练成本和约 289 秒 eval anchor。
- SaMST strict 750 historical inference 有 39.826s / 750 images 记录。
- TokenizerClean 1024 行 summary-level wall time 主要是 full_eval/quick_eval audit，不是训练成本。
- `lambda_grid` / `step_count_sweep` 的 `0.000/0.001s` 属于 dry-run，不可作训练/推理速度。
- SA-SWD random arm runtime-anomalous，只能作 quality/anomaly 记录。

未完成：

- `SchrodingerBridge/docs/timing/training_inference_timing_master.csv` 尚未和 `timing_quality_master_20260605.csv` reconciliation。
- claim-facing prose 使用前仍要逐条 source-open。
- 缺失训练时间继续留空，不补猜。

## 7. 清理总量和边界

已记录释放空间的主要块：

| 区域 | 删除内容 | 释放空间 |
| --- | --- | ---: |
| local manual checkpoint cleanup | 875 个非主线 checkpoint-like 文件 | 46032.053 MB |
| remote SchrodingerBridge/exp | 84 个非保留 epoch checkpoint | 4961.604 MB |
| remote SaMAM | 7 个 redundant `last*.ckpt` alias | 1931.291 MB |
| remote TokenizerClean checkpoint cleanup | 141 个 uncited checkpoint | 5198.991 MB |
| remote TokenizerClean no-summary probe cleanup | 18 个 probe/calibration checkpoint | 362.391 MB |
| remote TokenizerClean retained orphan probe cleanup | 11 个 orphan probe weights | 170.017 MB |
| remote TokenizerClean generated media cleanup | 43008 个 uncited generated media | 11883.246 MB |
| remote main data/cache/archive residue | 11 个 failed/stale/empty residue targets | 381.807 MB |
| remote duplicate/stale archive cleanup | 3 个 archive files | 3290.714 MB |
| remote RAR weight-only archive cleanup | 6 个 RAR/part files | 6553.384 MB |
| local remaining surface cleanup | 5 个 exact whitelist targets | 237.860 MB |
| local cache/root/dataset cleanup | invalid cache、duplicate archive、failed dataset cache 等 | 1623.145 MB |

边界：

- 不碰论文 tex/pdf/png。
- 不碰用户已有脏文件。
- 不按扩展名批量删 `.pt/.ckpt/.tar/.zip/.rar`。
- 删除必须有 policy CSV/MD、per-file ledger、post-delete verification 或等价证据。
- 当前 staged 的论文相关文件不是考古范围，不能和本任务一起提交。

## 8. 8 小时级剩余执行计划

| block | 预算 | 目标 | 产物 | 状态 |
| --- | ---: | --- | --- | --- |
| 0 | 0.25h | 固化当前 git 和证据状态 | status snapshot | 已做 |
| 1 | 0.75h | 产出可读总报告，替代乱码 CN 入口 | 本报告 + evidence map CSV | 本轮推进 |
| 2 | 1.00h | TokenizerClean 10 个 retained no-summary dirs owner review | owner-review CSV/MD + orphan cleanup ledger | 第二轮完成；7 个 trained payload 仍需 owner/summary |
| 3 | 1.00h | TokenizerClean cited/current media archive/migration policy | media migration policy | 未完成 |
| 4 | 1.00h | Remote RAR archive provenance | RAR policy + cleanup ledger | 一轮完成；`experiments.rar` cache mismatch 和 `45.rar` unique archive 仍保留 |
| 5 | 1.00h | Cross-cache dedup hash audit | cache dedup CSV/MD | 未完成 |
| 6 | 1.00h | Docs timing master reconciliation | reconciled timing master / sidecar | 未完成 |
| 7 | 1.00h | Dataset split、timeline、README counts consistency pass | consistency audit | 未完成 |
| 8 | 0.75h | Completion audit、CSV import、diff check、只提交 archaeology | final audit + commit | 持续执行 |

## 9. 当前硬缺口

不能宣布完成的原因：

- 7 个 retained TokenizerClean trained no-summary payload 权重目录还没有 owner-level 最终决定。
- cited/current TokenizerClean media 还没有迁移/归档策略。
- RAR provenance 已推进：`Gate.rar`, `Attn_48.part*.rar`, `chess.part*.rar` 已按 weight-only archive policy 删除；仍缺 `experiments.rar` cache mismatch audit 和 `Cycle-NCE\45.rar` unique archive owner decision。
- cross-cache dedup 还没有 hash audit。
- docs timing master 还没和 timing quality overlay 对齐。
- 全仓 legacy generated media 的 nested owner-level review 还没全部完成。

下一步应优先做 TokenizerClean retained no-summary dirs owner review，因为它仍包含真实 weight-like 文件，又最容易因缺 summary 而误删当前证据。
