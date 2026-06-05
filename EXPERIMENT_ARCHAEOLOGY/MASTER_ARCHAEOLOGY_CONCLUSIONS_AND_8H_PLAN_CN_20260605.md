# Latent_Style 实验考古总报告与 8 小时执行计划 - 2026-06-05

## 先给结论

这项任务现在不能宣称“全部完成”。已经完成的是高价值 checkpoint 层的多轮清理和大部分主实验面的索引；还没完成的是所有低价值生成图、无 summary 权重目录、远程 archive/cache 的逐项 owner 级复核。

当前可信状态：

- 本地 `G:\GitHub\Latent_Style`：已经按 top-level 和主要实验家族打开索引，非主线 checkpoint 已有删除 ledger；剩余大头主要是数据、latent、feature、eval cache、依赖权重、当前 timing anchor、历史结果面。
- 远程主工作树 `I:\Github\Latent_Style`：`SchrodingerBridge/exp` epoch checkpoint thinning 和 SaMAM alias cleanup 已完成；剩余主要是被引用 anchor、SaMAM curve step checkpoints、data/cache/archive。
- 远程 TokenizerClean `I:\Github\Latent_Style_TokenizerClean`：已覆盖 `SchrodingerBridge/exp` 全部 145 个一级目录，完成 citation graph，并删除 141 个无引用且已有 summary/metrics 的探索 checkpoint；剩余 28 个无 summary 权重目录和大量 generated image evidence 需要下一轮 policy。

本报告的入口文件：

- `manual_goal_completion_audit_20260605.csv`
- `manual_8h_execution_plan_20260605.csv`
- `manual_coverage_matrix_20260605.csv`
- `manual_cleanup_retention_and_next_candidates_20260605.csv`
- `manual_remaining_weight_classes_20260605.csv`

## 本地 G: 结论

本地已经看过的主面不是单个脚本扫出来的，而是分成目录级 ledger：

| 本地区域 | 当前判断 | 证据 |
|---|---|---|
| `SchrodingerBridge/exp` | 当前 formal/timing evidence 和少量历史 anchor；非主线 checkpoint 已清理或分类 | `manual_schrodingerbridge_exp_topdir_ledger_20260605.csv` |
| `SchrodingerBridge/exp/local_wsl_wikiart512_hist_b32_e8` | WikiArt512 full-eval timing anchor，8 个 epoch 权重保留 | `manual_remaining_weight_classes_20260605.csv` |
| `SchrodingerBridge/docs/experiments` | 当前论文/实验证据入口，不动 tex/pdf | `manual_coverage_matrix_20260605.csv` |
| `Related_Works` | baseline/repro/metrics 面，主要保留依赖和结果；不是大 checkpoint 垃圾面 | `manual_related_works_directory_ledger_20260605.csv` |
| `Cycle-NCE` | 历史大指标面，保留 metrics/summary/ref cache；不按 checkpoint 垃圾处理 | `MANUAL_CYCLE_NCE_ARCHAEOLOGY_20260605.md` |
| `Dataset`, `style_data`, `latent-256`, `clip-feats-vitb32` | 数据/latent/feature cache，不是 checkpoint cleanup 目标 | `MANUAL_LOCAL_DATASET_CACHE_POLICY_20260605.md` |
| root `eval_cache` | ArtFID/CLIP/VAE/DINO/reference feature cache；只删坏下载和空 temp | `MANUAL_LOCAL_EVAL_CACHE_POLICY_20260605.md` |
| `archive`, root `exp`, `tmp` | duplicate tar、stale pid/log、空 probe 残留已处理；paper tmp/tex/pdf/png 不动 | `MANUAL_LOCAL_ROOT_MISC_POLICY_20260605.md` |

本地已执行的清理：

- `cleanup/local_deleted_checkpoints.csv` 的 broad cleanup 层显示：329 个 deleted、11575.670 MB；38026 个 skipped、16243.914 MB。这层包含很多 review candidates，没有盲删。
- `cleanup/manual_deleted_checkpoints_20260605.csv` 的后续 manual checkpoint 清理层显示：875 个 deleted、46032.053 MB。
- root misc 删除：重复 `Cycle-NCE.tar` 释放 1503.203 MB；stale PID/失败 launcher residue 删除。
- eval cache 删除：坏 `.incomplete` HF blob 释放 55.994 MB，空 ModelScope temp 删除。
- dataset cache 删除：失败 `wikiart_81k` cache residue 释放 63.948 MB。

本地不能继续删的主要类：

- `latent-256`, `clip-feats-vitb32`, `Dataset`, `style_data`, `SchrodingerBridge/scale`：数据和特征缓存。
- `eval_cache`：评测依赖和离线模型缓存。
- `Related_Works` 的 VGG/Inception/LPIPS 等依赖权重。
- `SchrodingerBridge/exp/local_wsl_wikiart512_hist_b32_e8`：当前 WikiArt512 timing anchor。
- `Cycle-NCE`：历史指标和 reference cache，除非制定 archive policy。

## 远程主工作树 I: 结论

远程主工作树：

`I:\Github\Latent_Style`

### `SchrodingerBridge/exp`

已完成 epoch-level thinning：

| 状态 | 文件数 | 大小 | 证据 |
|---|---:|---:|---|
| 清理前 | 101 checkpoint | 约 5945 MB | `manual_remote_schrodingerbridge_epoch_evidence_20260605.csv` |
| 已删除 | 84 `.pt` | 4961.604 MB | `cleanup/manual_remote_schrodingerbridge_epoch_cleanup_20260605.csv` |
| 剩余 | 17 checkpoint | 983.457 MB | `manual_remote_schrodingerbridge_remaining_weights_after_thinning_20260605.csv` |

保留原则：

- path-stability probe 的 base/k000/k025 `epoch_0001`。
- Distinct5 formal ablation 的 cited/best/anchor epoch。
- K/L/M 单点 anchor。
- SADD exact/repro 的 e7/e8，因为 full_eval summary 锁定这些后段点。

删除原则：

- F-longer/K-longer 的非保留中间 epoch。
- rejected A/J ablation ckpt。
- SADD e1-e6 中间 ckpt。
- negative evidence 的 summary/metrics/log 保留，只删 checkpoint。

### Remote SaMAM

路径：

`I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag\step_checkpoints`

已完成 alias cleanup：

| 状态 | 文件数 | 大小 | 证据 |
|---|---:|---:|---|
| 清理前 | 19 ckpt | 约 5242 MB | `manual_remote_samam_checkpoint_thinning_policy_20260605.csv` |
| 已删除 | 7 `last*.ckpt` aliases | 1931.291 MB | `cleanup/manual_remote_samam_alias_cleanup_20260605.csv` |
| 剩余 | 12 `step-step=*.ckpt` | 3310.776 MB | `manual_remote_samam_remaining_step_checkpoints_after_alias_cleanup_20260605.csv` |

关键判断：

- whole-file SHA 不同，所以不能靠普通 hash 直接删。
- PyTorch metadata/state-dict hash 证明 `last*.ckpt` 是 paired step checkpoint 的模型重复。
- paired step 文件保留 optimizer/scheduler，是更完整的 curve/restart evidence。

### Remote main 剩余缺口

- `eval_cache`：评测/模型缓存，未做 file-level archive policy。
- `latent-*`, `latents*`, `SchrodingerBridge/scale/datasets`：输入数据/latent backend，不按 checkpoint 处理。
- `Cycle-NCE`, `experiments`：历史 archive/cache/dependency surface，需要 archive policy。

## 远程 TokenizerClean 结论

远程 TokenizerClean：

`I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge`

已完成：

- 覆盖 `exp` 全部 145 个一级目录。
- 打开并核对 TokenizerClean policy、tokenizer restart、main-table gap、flow-loss、SA-SWD、execution-alignment、L-family successor、master log。
- 生成 all-directory citation graph。
- 删除 44 个“无引用、非 aaai2027、有 summary/metrics”的探索目录中的 checkpoint。

删除结果：

| 项 | 数值 |
|---|---:|
| 删除 checkpoint | 141 |
| 释放空间 | 5198.991 MB |
| 删除候选目录 post-check | 0 个剩余 `.pt/.ckpt/.pth` |
| 保留 summary/metrics/config/log/png/csv | 是 |

删除后剩余权重：

| 类别 | 目录数 | 文件数 | 大小 | 判断 |
|---|---:|---:|---:|---|
| docs/reviews/master/paper 命中 | 34 | 122 | 3813.414 MB | 保留，直到 citation 迁移 |
| 当前 `aaai2027_*` packet | 9 | 24 | 1451.217 MB | 保留，只能 packet-specific thinning |
| 无引用但无 summary | 28 | 39 | 911.730 MB | 第一轮后待 review；后续已清理其中 18 个短 probe/calibration checkpoint |
| 已清理候选 | 44 | 0 | 0 MB | checkpoint 已删，数据证据保留 |
| 无 checkpoint | 30 | 0 | 0 MB | 本轮不动 |

后续 no-summary pass 已追加完成：

- 审查 28 个 no-summary checkpoint 目录。
- 删除其中 18 个短 probe/calibration checkpoint，释放 362.391 MB。
- 保留 7 个可能需要补 summary 的 payload 目录和 3 个无 config/summary orphan 目录。
- 最新 post-check 剩余 TokenizerClean `exp` checkpoint：167 个文件，5813.970 MB。

generated-media pass 也已追加完成：

- 按照远程已有 `2026-06-03-exploratory-image-prune.md` 规则，只删除未引用且有 summary/CSV 结构化证据的 generated media。
- 删除 50 个零引用 summary-backed 目录中的 `.png/.jpg/.jpeg/.webp/.gif`：43008 个文件，释放 11883.246 MB。
- 保留 `summary.json`、`metrics.csv`、logs、configs、checkpoints。
- post-check 证明 50 个删除候选目录媒体文件全部归零。
- 剩余媒体：cited media 5704.518 MB；current `aaai2027_*` media 1797.000 MB。

TokenzierClean 不能继续直接删的点：

- `diagnostics`：约 2868.443 MB media，因 docs/paper 命中保留。
- `tokenizer_control_probes`：已删除未引用 generated media，summary/metrics 保留。
- `configs`：约 368.559 MB，被 docs/master 命中，含 phase-space sweep 配置和 full_eval。
- `aaai2027_tokenizer_localization_*`：目前 docs 引用没跟上，但目录结构是新近 formal packet，先保留。

## 实验脉络

当前仓库可以按 6 个阶段理解：

| 阶段 | 时间 | 主线 | 当前读法 |
|---|---|---|---|
| Phase A | 2026-02 到 2026-03 | legacy/no-edge/style-transfer 早期实验 | 历史脉络，不作为当前 claim |
| Phase B | 2026-03 到 2026-04 | legacy256/StyleID/IDT/no-tokenized/tokenized | baseline 和 sanity check，timing 混杂 |
| Phase C | 2026-04 到 2026-05 | Cycle-NCE / Latent AdaCUT | 本地大指标面，保留 metrics/summary |
| Phase D | 2026-05 | SchrodingerBridge/LANCET phase-space | grid/search/frontier/vae_backend/representation，很多是探索面 |
| Phase E | 2026-05-30 到 2026-06-02 | WikiArt512 与 Distinct5 formal evidence | 当前 timing/efficiency claim 核心证据面 |
| Phase F | 2026-06-03 起 | AAAI2027 / TokenizerClean claim closing | flow-loss、SA-SWD、tokenizer execution、time-to-parity，仍需 review-grade 整理 |

## Timing 结论

当前可复用 timing 入口：

- `manual_timing_evidence_20260605.csv`
- `manual_remote_tokenizerclean_timing_evidence_20260605.csv`
- `SchrodingerBridge/docs/timing/training_inference_timing_master.csv`
- `Related_Works/results/metrics_summary/timing_summary.csv`

已经能读出的关键点：

- WikiArt512 LANCET/LBM full eval 有本地 wall time anchor：约 210.67s external / 206.79s internal。
- Distinct5 LBM formal retained points是分钟级训练证据。
- SaMAM Distinct5-512 的 indexed partial curve 是小时级训练成本，step 3000 有 `TRAIN_STEP_3000_WALL_SECONDS=3156.25` 和 `EVAL_STEP_3000_WALL_SECONDS=289.31`。
- SaMST historical strict 750 inference 有 39.826s / 750 images，单图约 0.0531s/img。
- TokenizerClean summary-level full_eval wall time 已新增 1024 行；训练时间字段空白，除非日志明确记录，不编造。
- `lambda_grid` / `step_count_sweep` 的 `0.000/0.001s` dry-run 不能当训练/推理 timing。
- SA-SWD random arm 是 runtime-anomalous，只能 quality-only，不能当正常 speed evidence。

## 当前 8 小时计划

详细 CSV：`manual_8h_execution_plan_20260605.csv`。

执行顺序：

1. 稳定当前状态和总报告入口。
2. 逐个打开 TokenizerClean 28 个 no-summary checkpoint dirs。
3. 给 remote generated-image evidence 制定 archive/delete policy。
4. 给 remote data/cache/archive 制定保留/删除 policy。
5. 复核 local data/cache/dependency surfaces。
6. 把 timing 做二次质量分层：dry-run/anomalous/smoke/full-eval。
7. 对 dataset split、timeline、README counts 做一致性修复。
8. 跑 completion audit、CSV import、`git diff --check`，只提交 archaeology。

## 不能宣称完成的部分

- TokenizerClean 28 个 no-summary checkpoint dirs 已做第一轮 review 并清理 18 个；仍剩 10 个需要 owner review 或补 summary。
- TokenizerClean generated image 已完成零引用 summary-backed media prune；仍需处理 cited/current/paper-facing media 的 archive/migration policy。
- 还没有 file-level 复核 remote latent backend / eval_cache / Cycle-NCE archive / experiments archive。
- timing master 还没有合并 TokenizerClean 1024 行和质量标签。
- 旧 broad summary 与 manual cleanup ledger 的聚合数字还需要 reconciliation，不能继续用单一“清理总量”一句话盖过去。

## 追加：远程主仓 data/cache/archive 人工复核 - 2026-06-05

本轮已完成 `I:\Github\Latent_Style` 的远程主仓数据、latent、eval cache、dataset cache、Cycle-NCE、experiments、StarGAN、seedream、Related_Works results/repos 的逐项打开复核。入口文件：

- `MANUAL_REMOTE_MAIN_DATA_CACHE_ARCHIVE_POLICY_20260605.md`
- `manual_remote_main_data_cache_archive_policy_20260605.csv`
- `manual_remote_main_data_cache_archive_delete_candidates_20260605.csv`
- `cleanup/manual_remote_main_data_cache_archive_residue_cleanup_20260605.csv`
- `manual_remote_main_data_cache_archive_post_delete_verify_20260605.csv`

结论：

- `data`、`style_data`、`latents*`、`latent-256*` 均为数据/latent backend，无失败残留，保留。
- `eval_cache` 删除 1 个失败 CLIP `.incomplete` 和 1 个 128-byte stale ref_feats tmp；完整 manual CLIP、offline pairing、ArtFID、VAE cache 保留。
- `SchrodingerBridge\scale\datasets` 删除 `wikiart_81k` 失败下载 `.incomplete` 和 stale `.lock`；数据集实体保留。
- `Cycle-NCE` 删除 3 个失败 `.incomplete` 和 2 个 stale lock；大包 `Gate.rar`、`1-decoder...zip`、分卷 rar、`45.rar` 不删，因还未证明重复/可弃。
- `experiments` 删除递归空的 ModelScope `._____temp`；`uv.lock` 和 torch compile hash 误报保留。
- `Related_Works\repos` 删除空 `S2WAT-main\pre_trained_models\tmp_timing`；baseline repo 权重保留。

删除结果：11 个精确目标，释放 `381.807389 MB`，post-delete 复验全部 `post_exists=False`，父目录仍存在。现在 remote main 的 data/cache/archive 层已完成 residue-only pass；剩余缺口是 Cycle-NCE/root archives 的所有权与重复性证明、`experiments.rar` 的 provenance、跨 cache dedup hash audit。