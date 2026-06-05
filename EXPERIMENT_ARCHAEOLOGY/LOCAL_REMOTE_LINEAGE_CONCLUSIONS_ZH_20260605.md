# Latent_Style 本地/远程实验考古结论总览 - 2026-06-05

本文件是给后续继续考古用的结论入口。它不把脚本扫描当作最终结论；脚本只用于定位，结论必须能回到具体目录、日志、summary、config、CSV policy、cleanup ledger 或 post-delete verification。

当前状态：没有完成全仓最终闭环。已经完成的是本地主要实验面、远程主树、远程 TokenizerClean 的分层归纳和多轮清理；仍缺 cross-cache dedup、docs timing master reconciliation、TokenizerClean cited/current media 迁移、7 个 no-summary trained payload 的 summary/owner 决策、以及 nested generated-image owner-level review。

## 一句话结论

- 本地 `G:\GitHub\Latent_Style`：主实验面已经按目录家族归纳，非主线 checkpoint 大规模清理完成；保留对象主要是 formal timing anchor、数据/latent/feature/eval cache、baseline 依赖、历史指标证据和未证明可删的 archive。
- 远程 `I:\Github\Latent_Style`：主实验、SaMAM、data/cache/archive/RAR 面已经逐项手检并清理；`experiments.rar` 已证明为 resolved duplicate 并删除；`Cycle-NCE\45.rar` 已打开，因含唯一非权重证据而保留。
- 远程 `I:\Github\Latent_Style_TokenizerClean`：145 个 exp 顶层目录已建 citation/no-summary/media 证据层；uncited/probe/orphan checkpoint 和 uncited generated media 已清理；剩余 no-summary trained payload 已第三遍确认不是孤儿垃圾。
- 实验脉络：从 legacy style-transfer/Cycle-NCE 到 SchrodingerBridge/LANCET，再到 WikiArt512/Distinct5 formal evidence，最后进入 AAAI2027/TokenizerClean claim closing；不能把 dry-run、quick_eval、failed probe、runtime anomaly 提升为 formal claim。

## 本地结论

| 本地区域 | 结论 | 已清理 | 保留理由 | 证据入口 |
| --- | --- | --- | --- | --- |
| `SchrodingerBridge/exp` | formal/timing anchor 和少量历史 anchor；不是可一键清空的 exp 垃圾桶 | 本地非主线 checkpoint cleanup 覆盖到这类目录 | `local_wsl_wikiart512_hist_b32_e8` 等保留为 WikiArt512 full-eval/timing anchor | `manual_schrodingerbridge_exp_topdir_ledger_20260605.csv` |
| `Related_Works` | baseline/repro/metrics 面；有依赖权重、结果、summary、tiny placeholders | 非主线 checkpoint 已清理 | VGG/Inception/LPIPS、baseline repo 依赖和结果证据保留 | `manual_related_works_directory_ledger_20260605.csv` |
| `Cycle-NCE` | 历史大指标面；不能按 checkpoint/cache 名称清空 | 空 eval cache/HF residue 已处理 | metrics、summary、ref cache、历史 archive provenance 保留 | `MANUAL_CYCLE_NCE_ARCHAEOLOGY_20260605.md` |
| `Dataset`, `style_data`, `latent-256`, `clip-feats-vitb32`, `SchrodingerBridge/scale` | 数据、latent tensor、feature tensor、VAE/data backend | failed `wikiart_81k` cache residue 已删 | source data / latent / CLIP feature backend，不是训练 checkpoint | `MANUAL_LOCAL_DATASET_CACHE_POLICY_20260605.md` |
| root `eval_cache` | ArtFID/CLIP/VAE/DINO/reference feature cache | invalid `.incomplete` 和空 ModelScope temp 已删 | eval reproducibility dependency | `MANUAL_LOCAL_EVAL_CACHE_POLICY_20260605.md` |
| root `archive`, `tmp`, root `exp` | archive/tmp/launcher residue 分开处理 | duplicate `Cycle-NCE.tar`、stale launcher residue、空 probe 已删 | paper/PDF/TEX/PNG scratch 不碰；小 archive 证据保留 | `MANUAL_LOCAL_ROOT_MISC_POLICY_20260605.md` |
| local remaining surface | 逐项打开 zip/tar/lock/empty-dir | 2 空目录、2 duplicated zip、1 duplicated output tar 已删 | WDS shards、checkpoint tar、dependency gzip、repo lock、uv lock 保留 | `MANUAL_LOCAL_REMAINING_SURFACE_POLICY_20260605.md` |

本地主要清理量：

- 手动 checkpoint cleanup：875 个非主线 checkpoint-like 文件，`46032.053 MB`。
- root misc/cache/dataset cleanup：duplicate archive、invalid cache、failed dataset cache 等，约 `1623.145 MB`。
- local remaining surface：5 个 exact whitelist target，`237.860 MB`。

本地还没有宣布完成的原因：不是每一个 nested generated image directory 都已经 owner-level 完成；部分 local archive-like 文件仍需要独立 archive policy，而不能按扩展名删除。

## 远程主树结论

远程入口：`ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62`

| 远程区域 | 结论 | 已清理 | 保留理由 | 证据入口 |
| --- | --- | --- | --- | --- |
| `I:\Github\Latent_Style\SchrodingerBridge\exp` | current Distinct5/SADD lineage evidence；已做 epoch thinning | 删除 84 个非保留 `.pt`，`4961.604 MB` | 17 个 cited/probe/anchor epoch 保留 | `MANUAL_REMOTE_SCHRODINGERBRIDGE_EPOCH_THINNING_20260605.md` |
| SaMAM central `step_checkpoints` | 当前 SaMAM baseline curve；不能直接删 step 曲线 | 删除 7 个 redundant `last*.ckpt` aliases，`1931.291 MB` | 12 个 step checkpoint 保留完整 curve/restart/cited evidence | `MANUAL_REMOTE_SAMAM_CHECKPOINT_THINNING_20260605.md` |
| remote data/cache/archive residue | failed/stale/empty residue-only 面 | 删除 11 个 residue target，`381.807 MB` | data/latent/eval cache bodies 保留 | `MANUAL_REMOTE_MAIN_DATA_CACHE_ARCHIVE_POLICY_20260605.md` |
| duplicate/stale archive | proven duplicate/stale archive 面 | 删除 3 个 archive，`3290.714 MB` | 未证明可删的 archive 不动 | `MANUAL_REMOTE_ARCHIVE_PROVENANCE_20260605.md` |
| RAR archives | 已用临时 UnRAR 逐项打开 | 删除 Gate/Attn_48/chess RAR，共 `6553.384 MB`；删除 `experiments.rar`，`8091.026 MB` | `Cycle-NCE\45.rar` 含唯一非权重证据，保留 | `MANUAL_REMOTE_RAR_DEEP_PROVENANCE_20260605.md` |
| `Cycle-NCE\45.rar` | 唯一历史证据包，不是纯权重垃圾 | 无删除 | 4 configs、8 summaries、8 metrics CSV、6008 eval images、ma-probe、12 weights | `MANUAL_REMOTE_CYCLE_NCE_45_RAR_REVIEW_20260605.md` |

远程主树当前硬缺口：cross-cache dedup 还没做 hash audit；`45.rar` 如果要删除，必须先提取/确认非权重证据包，不能直接删 archive。

## 远程 TokenizerClean 结论

路径：`I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp`

| 层面 | 结论 | 已清理 | 保留理由 | 证据入口 |
| --- | --- | --- | --- | --- |
| 145 个 exp 顶层目录 | 已建立 internal evidence 和 citation graph | 只对 uncited/probe/orphan 类做清理 | cited/current/formal packet 不动 | `manual_remote_tokenizerclean_exp_internal_evidence_after_no_summary_cleanup_20260605.csv` |
| uncited summary-backed checkpoints | 可删非主线 exploratory checkpoint | 删除 141 个，`5198.991 MB` | summary/config/log 证据保留 | `cleanup/manual_remote_tokenizerclean_uncited_checkpoint_cleanup_20260605.csv` |
| no-summary probe/calibration | 孤儿 probe 或 calibration 权重 | 删除 18 个 probe/calibration checkpoint，`362.391 MB` | 真实 trained payload 不删 | `MANUAL_REMOTE_TOKENIZERCLEAN_NO_SUMMARY_REVIEW_20260605.md` |
| retained no-summary orphan probe | 纯 orphan probe weights | 删除 11 个权重和 3 个空目录，`170.017 MB` | diagnostics output 保留 | `MANUAL_REMOTE_TOKENIZERCLEAN_RETAINED_NO_SUMMARY_OWNER_REVIEW_20260605.md` |
| 7 个 trained no-summary payload | 第三遍确认是真训练 payload | 无删除 | 有 config/training CSV/log/src/weights；无 summary，需 recovery 或 owner decision | `MANUAL_REMOTE_TOKENIZERCLEAN_TRAINED_NO_SUMMARY_THIRD_PASS_20260605.md` |
| generated media | zero-hit summary-backed media 可删 | 删除 43008 个 media，`11883.246 MB` | cited/current/paper-facing media 保留 | `MANUAL_REMOTE_TOKENIZERCLEAN_GENERATED_MEDIA_PRUNE_20260605.md` |

TokenizerClean 当前硬缺口：7 个 trained no-summary payload 还需要 summary recovery 或 owner-delete decision；cited/current media 还需要 archive/migration policy。

## 实验脉络

| 阶段 | 时间 | 主线 | 现在应该怎么读 |
| --- | --- | --- | --- |
| Phase A | 2026-02 到 2026-03 | legacy/no-edge/style-transfer early experiments | 历史 sanity 和早期失败/探索，不作为当前 formal claim |
| Phase B | 2026-03 到 2026-04 | legacy256、StyleID、IDT、no-tokenized/tokenized | baseline 和 sanity；timing 混杂，需要 evidence quality 标签 |
| Phase C | 2026-04 到 2026-05 | Cycle-NCE / Latent AdaCUT | 历史大指标面，保留 metrics/summary/ref cache |
| Phase D | 2026-05 | SchrodingerBridge/LANCET phase-space | grid/search/frontier/vae_backend/representation，多数是探索面 |
| Phase E | 2026-05-30 到 2026-06-02 | WikiArt512 和 Distinct5 formal evidence | 当前 timing/efficiency claim 的核心证据面 |
| Phase F | 2026-06-03 起 | AAAI2027 / TokenizerClean claim closing | flow-loss、SA-SWD、tokenizer execution、time-to-parity；需要 review-grade 整理 |

脉络判断：正式论文/claim 只能引用 Phase E/F 中有 full_eval、summary、training/eval timing、source-open 的行。dry-run、failed probe、quick_eval、runtime anomaly、placeholder checkpoint 只能做 audit 或 negative evidence。

## Timing 结论

当前 timing 质量层已经有 `1093` 行 overlay：

- claim-support candidate：`full_eval_wall_time` 51 行，`train_and_eval_wall_time` 2 行。
- audit-only 或 historical：绝大多数 TokenizerClean full_eval/quick_eval wall-time、historical timing、partial training。
- exclude/negative：dry-run、failed probe、runtime anomaly。

当前还不能把 timing 当最终表使用，因为 `SchrodingerBridge/docs/timing/training_inference_timing_master.csv` 尚未与 `timing_quality_master_20260605.csv` reconciliation。

## 清理总账

本轮之前和本轮已经形成的主要清理证据块：

| 区域 | 内容 | 释放空间 |
| --- | --- | ---: |
| local manual checkpoint cleanup | 875 个非主线 checkpoint-like 文件 | 46032.053 MB |
| remote SchrodingerBridge/exp | 84 个非保留 epoch checkpoint | 4961.604 MB |
| remote SaMAM | 7 个 redundant `last*.ckpt` alias | 1931.291 MB |
| remote TokenizerClean checkpoint/probe/orphan cleanup | 170 个 checkpoint/probe/orphan weight files | 5731.399 MB |
| remote TokenizerClean generated media cleanup | 43008 个 uncited generated media | 11883.246 MB |
| remote main data/cache/archive residue | 11 个 failed/stale/empty residue targets | 381.807 MB |
| remote duplicate/stale archive cleanup | 3 个 archive files | 3290.714 MB |
| remote RAR weight-only archive cleanup | 6 个 RAR/part files | 6553.384 MB |
| remote `experiments.rar` resolved duplicate cleanup | 1 个 RAR file | 8091.026 MB |
| local remaining surface cleanup | 5 个 exact whitelist targets | 237.860 MB |

所有未来删除必须继续满足：policy CSV/MD、per-file ledger、post-delete verification。不能按 `.pt/.ckpt/.tar/.zip/.rar` 扩展名或文件大小批量删。

## 8 小时级后续计划

| block | 预算 | 目标 | 产物 | 当前状态 |
| --- | ---: | --- | --- | --- |
| 1 | 1.0h | 本地 nested generated-image owner-level review | media/owner policy CSV/MD | 未完成 |
| 2 | 1.0h | TokenizerClean cited/current media archive/migration policy | media migration policy + optional cleanup whitelist | 未完成 |
| 3 | 1.0h | 7 个 trained no-summary payload summary recovery / owner decision | recovery/decision CSV | 未完成 |
| 4 | 1.0h | cross-cache dedup hash audit | cache dedup CSV/MD | 未完成 |
| 5 | 1.0h | docs timing master reconciliation | reconciled timing master / sidecar | 未完成 |
| 6 | 1.0h | dataset split、timeline、README counts consistency pass | consistency audit | 未完成 |
| 7 | 1.0h | optional `45.rar` curated nonweight extraction policy | extraction/retention policy | 可选，未完成 |
| 8 | 1.0h | final completion audit | requirement-by-requirement audit | 未完成 |

## 当前不能宣布完成的原因

- 本地不是所有 nested generated-image directory 都完成 owner-level review。
- TokenizerClean 7 个 trained no-summary payload 已确认不是垃圾，但仍缺 summary recovery / owner decision。
- TokenizerClean cited/current media 还没有 archive/migration policy。
- cross-cache dedup 还没有 hash audit。
- docs timing master 还没和 timing quality overlay 对齐。
- `Cycle-NCE\45.rar` 若要删除，还需要先提取或确认非权重证据包。
