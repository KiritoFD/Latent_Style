# Latent_Style 实验考古总归纳 - 2026-06-05

本文件是新的干净入口报告，用来回答三个问题：本地现在是什么状态、远程现在是什么状态、实验脉络是什么。旧的 broad CSV 仍然是索引，本报告只引用已经有人工 ledger、cleanup ledger 或 post-delete verification 支撑的结论。

## 当前结论

整体状态：还不能宣称“全仓每一个嵌套目录都完成 owner-level 复核”。已经完成的是主线实验面、checkpoint 清理面、remote TokenizerClean `exp` 面、remote main data/cache/archive residue 面，以及 timing 证据的一轮归档。仍缺的是部分低价值 generated media 的迁移策略、10 个 TokenizerClean no-summary 权重目录的 owner review、Cycle-NCE/root archive provenance、`experiments.rar` provenance、跨 cache dedup、timing quality 二次分层。

当前已归档索引规模：

- `final_master_experiments.csv`: 22629 行实验/证据记录，其中本地 `G:` 17134 行、远程 `I:` 5495 行。
- `final_timeline.csv`: 7829 条时间线事件。
- `final_by_dataset/`: 25 个 dataset/setting 分表。
- `manual_coverage_matrix_20260605.csv`: 40 个覆盖面状态条目。
- `manual_goal_completion_audit_20260605.csv`: 10 条需求级完成度审计，当前仍为 not complete。

## 本地 G:\GitHub\Latent_Style

本地已完成的人工覆盖面不是“一个脚本扫完”，而是按目录面拆成多个 ledger：

- `manual_top_level_directory_index_20260605.csv`: 67 行本地/远程顶层目录索引。
- `manual_schrodingerbridge_exp_topdir_ledger_20260605.csv`: 118 行本地 `SchrodingerBridge/exp` 顶层 ledger。
- `manual_related_works_directory_ledger_20260605.csv`: 128 行本地 `Related_Works` ledger。
- `manual_cycle_nce_directory_ledger_20260605.csv`: 36 行本地 `Cycle-NCE` family ledger。
- `manual_local_eval_cache_policy_20260605.csv`, `manual_local_dataset_cache_policy_20260605.csv`, `manual_local_root_misc_policy_20260605.csv`: 本地 cache/data/root misc 的逐项保留/清理策略。

本地结论：

- `SchrodingerBridge/exp`: 保留 formal WikiArt512 timing anchor、Distinct5/AAAI 当前证据、必要 full_eval/log/config；非主线 probe/calibration weights 已经按 ledger 清理。
- `Related_Works`: 主要是 baseline/repro/metrics 证据。VGG/Inception/LPIPS 依赖权重保留，tiny placeholder 只作为后续语义清理候选，不作为释放空间目标。
- `Cycle-NCE`: 是历史大指标面，保留 metrics/summary/ref cache。它不是简单 checkpoint 垃圾目录，后续只能走 archive/provenance policy。
- `Dataset`, `style_data`, `latent-256`, `clip-feats-vitb32`, `SchrodingerBridge/scale`: 是数据、latent、feature cache 或高分辨率数据集，不按 checkpoint cleanup 删除。
- root `eval_cache`: ArtFID/CLIP/VAE/DINO/reference feature cache 是评估依赖。只删除了失败 `.incomplete` 和空 temp。
- root `archive`/`tmp`/`exp`: 删除过重复 archive tar、stale launcher residue 和空 probe residue；paper `.tex/.pdf/.png` 边界保持不动。

本地已执行清理：

- `cleanup/local_deleted_checkpoints.csv`: broad pass 中实际 deleted 329 项，释放 `11575.670 MB`；另有 38026 项 skipped/review，不应被当成已删除。
- `cleanup/manual_deleted_checkpoints_20260605.csv`: 后续人工 checkpoint cleanup deleted 875 项，释放 `46032.053 MB`。
- `cleanup/manual_cache_cleanup_20260605.csv`: 本地 eval cache residue cleanup，删除无效 cache residue `55.994 MB`。
- `cleanup/manual_root_misc_cleanup_20260605.csv`: 删除重复 `Cycle-NCE.tar` 和 stale launcher residue，释放 `1503.203 MB`。
- `cleanup/manual_dataset_cache_cleanup_20260605.csv`: 删除失败 `wikiart_81k` HF cache residue，释放 `63.948 MB`。

本地未完成：

- 每个 nested generated image 目录还没有全部 owner-level 复核。
- `Cycle-NCE` 和 root/archive 历史包需要 archive provenance。
- timing 还需要 full_eval/smoke/dry-run/anomalous 质量分层。

## 远程主仓 I:\Github\Latent_Style

远程主仓已完成四类深查：

- `SchrodingerBridge/exp` 顶层 inventory: `manual_remote_schrodingerbridge_exp_topdir_inventory_20260605.csv`，124 行。
- `SchrodingerBridge/exp` checkpoint epoch policy: `manual_remote_schrodingerbridge_epoch_evidence_20260605.csv` 和 cleanup ledger。
- SaMAM central curve checkpoint hash/metadata audit: `manual_remote_samam_*_20260605.csv`。
- data/cache/archive residue pass: `MANUAL_REMOTE_MAIN_DATA_CACHE_ARCHIVE_POLICY_20260605.md` 和同名 policy/cleanup/post-delete CSV。

远程主仓结论：

- `SchrodingerBridge/exp`: 保留 cited/probe/anchor epochs 和 full_eval/log/config；删除失败或非保留中间 epoch checkpoint。
- SaMAM central `step_checkpoints`: 删除 `last*.ckpt` aliases，但保留 12 个 `step-step=*.ckpt` 作为 curve/restart/evidence anchor。
- `data`, `style_data`, `latents*`, `latent-256*`: 已逐项打开，都是数据/latent backend，无失败残留，保留。
- `eval_cache`: 删除失败 CLIP `.incomplete` 和 stale ref_feats tmp；完整 manual CLIP/offline pairing/ArtFID/VAE cache 保留。
- `SchrodingerBridge/scale/datasets`: 删除 `wikiart_81k` 失败下载 residue 和 stale lock；数据集实体保留。
- `Cycle-NCE`: 删除失败 ArtFID/CLIP `.incomplete` 与 stale locks；`Gate.rar`, `1-decoder...zip`, Attn/chess 分卷、`45.rar` 未证明可弃，保留待 provenance。
- `experiments`: 删除递归空的 ModelScope `._____temp`；legacy experiments 和完整 eval caches 保留。
- `Related_Works/repos`: 删除空 `S2WAT-main/pre_trained_models/tmp_timing`；baseline repos 和依赖权重保留。
- `StarGAN`, `seedream45_api`: 打开过，未发现 bad markers，释放空间收益低，保留。

远程主仓已执行清理：

- `cleanup/manual_remote_schrodingerbridge_epoch_cleanup_20260605.csv`: 删除 84 个 checkpoint，释放 `4961.604 MB`。
- `cleanup/manual_remote_samam_alias_cleanup_20260605.csv`: 删除 7 个 alias checkpoint，释放 `1931.291 MB`。
- `cleanup/manual_remote_main_data_cache_archive_residue_cleanup_20260605.csv`: 删除 11 个 residue/empty-temp 目标，释放 `381.807 MB`，post-delete 全部 `post_exists=False`。

远程主仓未完成：

- `Cycle-NCE` 大包和 root `experiments.rar` 需要 duplicate/provenance audit，不能按扩展名删。
- complete model/eval caches 是否跨目录重复，需要 hash audit。
- legacy `experiments` 的所有 nested family 还未全部 owner-level 归档。

## 远程 TokenizerClean

远程路径：

`I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp`

已完成覆盖：

- `manual_remote_tokenizerclean_exp_internal_evidence_after_no_summary_cleanup_20260605.csv`: 145 个 `exp` 顶层目录全部纳入 evidence table。
- `manual_remote_tokenizerclean_exp_citation_graph_all_20260605.csv`: docs/reviews/master/paper citation graph。
- `manual_remote_tokenizerclean_no_summary_review_20260605.csv`: 28 个 no-summary checkpoint 目录逐项 review。
- `manual_remote_tokenizerclean_generated_media_inventory_after_cleanup_20260605.csv`: generated media cleanup 后复验 inventory。

已执行清理：

- `cleanup/manual_remote_tokenizerclean_uncited_checkpoint_cleanup_20260605.csv`: 删除 141 个 uncited summary-backed exploratory checkpoint，释放 `5198.991 MB`。
- `cleanup/manual_remote_tokenizerclean_no_summary_probe_checkpoint_cleanup_20260605.csv`: 删除 18 个 no-summary probe/calibration checkpoint，释放 `362.391 MB`。
- `cleanup/manual_remote_tokenizerclean_uncited_generated_media_cleanup_20260605.csv`: 删除 43008 个 zero-hit summary-backed generated media 文件，释放 `11883.246 MB`。

清理后仍保留：

- `manual_remote_tokenizerclean_remaining_weight_classes_after_no_summary_cleanup_20260605.csv`: 32 个目录仍含 167 个 weight-like 文件，合计 `5813.970 MB`。
- `manual_remote_tokenizerclean_remaining_media_classes_after_cleanup_20260605.csv`: 26 个目录仍含 46483 个 media 文件，合计 `7501.518 MB`。

保留原因：

- cited/docs/paper/current packet 命中，或缺 summary 但可能是 payload/orphan，需要 owner review。
- `aaai2027_*` 近端 formal packet 不做批量删除。
- cited/current media 需要 archive/migration policy，不按“图片很多”直接删。

## 实验脉络

当前仓库可以按 6 个阶段读：

1. 2026-02 到 2026-03: legacy/no-edge/style-transfer 早期实验，主要是历史脉络和 baseline sanity。
2. 2026-03 到 2026-04: legacy256、StyleID、IDT、no-tokenized/tokenized 早期风格迁移探索，timing 混杂，不能直接当当前 claim。
3. 2026-04 到 2026-05: Cycle-NCE / Latent AdaCUT 历史大指标面，保留 metrics/summary/cache，用于解释早期方向。
4. 2026-05: SchrodingerBridge/LANCET phase-space，包含 grid/search/frontier/vae_backend/representation，大量探索面已经按 checkpoint policy 清理。
5. 2026-05-30 到 2026-06-02: WikiArt512 与 Distinct5 formal evidence，是当前 timing/efficiency claim 的核心证据面。
6. 2026-06-03 以后: AAAI2027 / TokenizerClean claim closing，包含 flow-loss、SA-SWD、tokenizer execution、time-to-parity；这部分仍在整理和 owner-review。

## Timing 证据

当前 timing 入口：

- `manual_timing_evidence_20260605.csv`: 69 行人工 timing evidence。
- `manual_remote_tokenizerclean_timing_evidence_20260605.csv`: 1024 行 TokenizerClean full_eval wall-time evidence。
- `SchrodingerBridge/docs/timing/training_inference_timing_master.csv`: 文档目录下 timing master。
- `final_master_experiments.csv`: 约 400 行含 timing 字段或 timing validity 的记录。

已知可用结论：

- Distinct5 LANCET/LBM formal retained points 是分钟级训练证据。
- WikiArt512 LANCET/LBM full_eval 有约 210 秒级 wall-time anchor。
- SaMAM Distinct5-512 step 3000 有小时级训练成本和约 289 秒 eval anchor。
- SaMST strict 750 historical inference 有 39.826s / 750 images 记录。
- `lambda_grid` / `step_count_sweep` 的 `0.000/0.001s` 属 dry-run，不可作为训练/推理速度。
- SA-SWD random arm runtime-anomalous，只能作为 quality-only 或异常记录。

本轮追加的 timing quality overlay：

- `timing_quality_master_20260605.csv`: 1093 行，合并 69 行人工 timing 和 1024 行 TokenizerClean summary timing。
- `timing_quality_summary_20260605.csv`: quality class 汇总。
- `TIMING_EVIDENCE_QUALITY_PASS_20260605.md`: 解释哪些 rows 可作为 claim candidate，哪些只能 archive/audit。

质量分层结论：

- `candidate_claim_support_with_caveat`: 53 行，其中 51 行 full-eval wall-time，2 行 train+eval wall-time。
- `audit_full_eval_wall_time_only`: 978 行，主要是 TokenizerClean full_eval/quick_eval summary wall time，不能当训练成本。
- `historical_context`: 28 行。
- `audit_only`: 24 行。
- `exclude_formal_claim`: 7 行。
- `quality_only_or_anomaly`: 1 行。

剩余缺口：

- 还没有把 quality overlay 回写到 `SchrodingerBridge/docs/timing/training_inference_timing_master.csv`。
- claim-facing prose 使用前仍需逐条 source-open，训练时间字段缺失处继续留空，不补猜。

## 清理原则

已经执行的删除都必须满足三件事：

- 有 policy CSV 或 narrative 说明为什么可删。
- 有 per-file ledger 记录路径、大小、原因。
- 有 post-delete 或同等验证，至少证明目标不存在且父目录未被误删。

不能直接删除的对象：

- source datasets、latent roots、feature caches。
- ArtFID/CLIP/VAE/DINO/LPIPS/VGG 等 metric/model dependencies。
- cited/current/paper-facing evidence。
- 只有“看起来大”的 archives，除非 provenance/hash/owner policy 已证明可删。

## 下一轮 8 小时级计划

当前 `manual_8h_execution_plan_20260605.csv` 中 0-4 已完成或一轮完成，5-8 仍待做：

1. Local remaining data/cache/dependency surfaces: 复核本地剩余 data/cache/dependency 和 placeholder policy。
2. Timing master quality pass: 已完成 archaeology overlay；后续只剩 docs timing master 回写与 claim-facing source-open。
3. Experiment lineage and dataset split finalization: 修正 dataset split、timeline、README counts 与 master report 一致性。
4. Completion audit and commit: 每个 block 做 CSV import、`git diff --check`、只 stage `EXPERIMENT_ARCHAEOLOGY`。

当前不能宣称完成的硬缺口：

- 10 个 TokenizerClean no-summary 权重目录需要 owner review 或补 summary。
- cited/current TokenizerClean media 需要 archive/migration policy。
- Cycle-NCE/root archives、`experiments.rar`、cross-cache dedup 需要 provenance/hash audit。
- docs timing master 仍需与 `timing_quality_master_20260605.csv` 对齐。
- 本地和远程 legacy generated media 的 nested owner-level review 仍未全覆盖。
