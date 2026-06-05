# Latent_Style 实验考古权威归纳 - 2026-06-05

本文件是当前 `EXPERIMENT_ARCHAEOLOGY` 的阅读入口。它把已经手工打开过的本地目录、远程目录、日志、summary、config、CSV、cleanup ledger 串成结论。它不是脚本扫描的替代品，也不把 broad scan 当成最终判断；每个 destructive cleanup 均以对应的 policy/ledger 为准。

## 当前结论

仓库还没有达到“所有目录都已逐项完全考古完毕”的终态，但已经完成了三类实质工作：

- 已把本地 G: 和远程 I: 的主要实验面分层：当前证据、历史证据、baseline 证据、数据/cache、依赖、失败残留。
- 已对本地非主线残留和远程 checkpoint 大头做了 policy-driven cleanup，而不是继续按扩展名猜删。
- 已形成可复用的 timing/metric/cleanup 索引，后续应继续从 `manual_coverage_matrix_20260605.csv` 和本文件进入。

当前不能再用一句“清理干净了”概括。正确状态是：

- 本地 main checkout：非主线训练 ckpt 已大幅清理；剩余大头主要是数据、feature/latent cache、eval cache、依赖、当前 formal evidence。
- 远程 main `I:\Github\Latent_Style`：`SchrodingerBridge/exp` 和 SaMAM 中央 ckpt 已做细化 thinning；剩余大头主要是 SaMAM step 曲线、latent backend cache、eval cache、Cycle-NCE/experiments archive、TokenizerClean 另一个工作树。
- 远程 TokenizerClean：仍是当前 AAAI/tokenizer 工作树，不允许 mass delete；下一步必须先做 citation graph。

## 本地 G: 结论

本地根目录 `G:\GitHub\Latent_Style` 的手工覆盖见：

- `manual_top_level_directory_index_20260605.csv`
- `manual_coverage_matrix_20260605.csv`
- `manual_cleanup_retention_and_next_candidates_20260605.csv`
- `manual_remaining_weight_classes_20260605.csv`

### 本地当前证据

| 区域 | 结论 | 证据入口 |
|---|---|---|
| `SchrodingerBridge/exp` | 当前 local formal evidence 主要是 WikiArt512 timing anchor 和少量留存 gate；非主线 ckpt 已基本清空 | `manual_schrodingerbridge_exp_topdir_ledger_20260605.csv` |
| `SchrodingerBridge/exp/local_wsl_wikiart512_hist_b32_e8` | 8 个 epoch 权重保留，是 WikiArt512 formal training/full-eval timing anchor | `manual_remaining_weight_classes_20260605.csv` |
| `SchrodingerBridge/docs/experiments` | 当前论文/实验 evidence pack、timing sidecar、Distinct5/K-longer/path-stability 文档入口 | `manual_coverage_matrix_20260605.csv` |
| `Related_Works` | baseline/repro/metrics 面，主要剩余 weight-like 是依赖和 tiny fake eval placeholders，不是有效 disk cleanup 大头 | `manual_related_works_directory_ledger_20260605.csv` |
| `Cycle-NCE` | 历史大规模指标/summary 面，weight-like 主要是 eval ref cache，不是训练 ckpt | `MANUAL_CYCLE_NCE_ARCHAEOLOGY_20260605.md` |
| `Dataset`, `style_data`, `latent-256`, `clip-feats-vitb32`, `SchrodingerBridge/scale` | 数据、latent、feature tensor，不作为 checkpoint 垃圾处理 | `MANUAL_LOCAL_DATASET_CACHE_POLICY_20260605.md` |
| root `eval_cache` | ArtFID/CLIP/VAE/DINO/reference feature cache，除坏下载残留外保留 | `MANUAL_LOCAL_EVAL_CACHE_POLICY_20260605.md` |
| `archive`, root `exp`, `tmp` | duplicate archive tar/stale launcher residue 已处理；paper tmp/tex/pdf/png 不在 sidecar 写入范围 | `MANUAL_LOCAL_ROOT_MISC_POLICY_20260605.md` |

### 本地已清理

本地手工清理已经记录到：

- `cleanup/manual_deleted_checkpoints_20260605.csv`
- `cleanup/manual_root_misc_cleanup_20260605.csv`
- `cleanup/manual_cache_cleanup_20260605.csv`
- `cleanup/manual_dataset_cache_cleanup_20260605.csv`
- `cleanup/manual_empty_directory_cleanup_20260605.csv`

明确已做的手工清理包括：

- 删除重复归档 `archive/2026-05-19_cleanup/root/Cycle-NCE.tar`，释放 `1503.203 MB`。
- 删除 root `exp` 下 stale PID/失败 launcher log 和空 probe shell。
- 删除 root `eval_cache` 中坏的 `.incomplete` HF blob 和空 ModelScope temp 目录，释放 `55.994 MB`。
- 删除失败 HF dataset cache `SchrodingerBridge/scale/datasets/wikiart_81k`，释放 `63.948 MB`。
- 删除前序已确认的非主线 local checkpoints，详见 `cleanup/manual_deleted_checkpoints_20260605.csv`。

本地未删但应保留的主要类别：

- `latent-256`, `clip-feats-vitb32`, `Dataset`, `style_data`, `SchrodingerBridge/scale`: 数据/latent/feature tensor。
- `eval_cache`: 当前评测依赖和缓存。
- `Related_Works` 依赖权重：VGG/Inception/LPIPS 等 baseline/eval dependency。
- `SchrodingerBridge/exp/local_wsl_wikiart512_hist_b32_e8`: 当前 timing anchor。
- `Cycle-NCE` 历史指标和 ref cache：作为历史实验面保留，除非做 archive policy。

## 远程 I: 结论

远程主机：

`ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62`

远程覆盖入口：

- `manual_remote_schrodingerbridge_exp_topdir_inventory_20260605.csv`
- `MANUAL_REMOTE_SCHRODINGERBRIDGE_EPOCH_THINNING_20260605.md`
- `MANUAL_REMOTE_SAMAM_CHECKPOINT_THINNING_20260605.md`
- `manual_coverage_matrix_20260605.csv`
- `manual_cleanup_retention_and_next_candidates_20260605.csv`

### Remote main `SchrodingerBridge/exp`

当前状态已经从“101 个 checkpoint 待 policy”更新为“已完成 epoch thinning”：

| 状态 | 文件数 | 大小 | 证据 |
|---|---:|---:|---|
| 清理前 | 101 ckpt | about `5945 MB` | `manual_remote_schrodingerbridge_epoch_evidence_20260605.csv` |
| 删除 | 84 `.pt` | `4961.604 MB` | `cleanup/manual_remote_schrodingerbridge_epoch_cleanup_20260605.csv` |
| 剩余 | 17 ckpt | `983.457 MB` | `manual_remote_schrodingerbridge_remaining_weights_after_thinning_20260605.csv` |

保留原则：

- path-stability probe 保留 base/k000/k025 的 `epoch_0001`。
- Distinct5 formal ablation 保留 cited/best/anchor epoch。
- K/L/M 保留单点 anchor。
- SADD exact/repro 保留 e7/e8，因为 full_eval summary 只锚定这些后段点。

删除原则：

- F-longer/K-longer 已作为 negative evidence 闭合，summary/metrics/grid/log 保留，ckpt 删除。
- 非保留中间 epoch 删除。
- rejected A/J ablation ckpt 删除。
- SADD e1-e6 中间 ckpt 删除，e7/e8 保留。

### Remote SaMAM central curve

路径：

`I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag\step_checkpoints`

当前状态：

| 状态 | 文件数 | 大小 | 证据 |
|---|---:|---:|---|
| 清理前 | 19 ckpt | about `5242 MB` | `manual_remote_samam_checkpoint_thinning_policy_20260605.csv` |
| 删除 | 7 `last*.ckpt` alias | `1931.291 MB` | `cleanup/manual_remote_samam_alias_cleanup_20260605.csv` |
| 剩余 | 12 `step-step=*.ckpt` | `3310.776 MB` | `manual_remote_samam_remaining_step_checkpoints_after_alias_cleanup_20260605.csv` |

关键判断：

- whole-file SHA 显示 `last*.ckpt` 与 paired step 文件不同，所以不能只凭文件 hash 删。
- PyTorch metadata/state-dict 复核显示 7 个 `last*.ckpt` 的 `state_dict` SHA 与对应 `step-step=000250..001750.ckpt` 完全一致。
- paired step 文件同样是 Lightning full checkpoint，含 optimizer/scheduler。
- 因此删除 alias 不牺牲模型/curve/restart 证据。
- 12 个 step checkpoint 仍完整保留，用于 SaMAM convergence curve、cited step、repair step 和 last step。

### Remote TokenizerClean

路径：

`I:\Github\Latent_Style_TokenizerClean`

当前结论：

- 这是当前 AAAI/tokenizer 工作树，不是垃圾目录。
- `SchrodingerBridge/exp` 仍被 docs/master log 直接引用。
- 目前只做了 top-level/citation-aware 初步分类，未做 destructive cleanup。
- 下一步必须先做 citation graph：docs/master log -> referenced exp path -> retained/deletable ckpt。

### Remote data/cache/archive

当前保留：

- `I:\Github\Latent_Style\eval_cache`: remote eval/model cache。
- `I:\Github\Latent_Style\latent-*` and `latents*`: latent backend/input cache。
- `I:\Github\Latent_Style\SchrodingerBridge\scale\datasets`: dataset tensor surface。
- `I:\Github\Latent_Style\Cycle-NCE`, `I:\Github\Latent_Style\experiments`: historical archive/cache/dependency surfaces，不能按 checkpoint cleanup 处理。

## 实验脉络

当前仓库可以按 6 个阶段理解：

| 阶段 | 时间 | 主线 | 当前读法 |
|---|---|---|---|
| Phase A | 2026-02 到 2026-03 | legacy/no-edge/style-transfer 早期实验 | 大量 unknown/legacy summary/log；作为历史脉络，不作为当前 claim |
| Phase B | 2026-03 到 2026-04 | legacy256/StyleID/IDT/no-tokenized/tokenized | baseline 和 sanity check；timing 不完整 |
| Phase C | 2026-04 到 2026-05 | Cycle-NCE / Latent AdaCUT | 本地 `Cycle-NCE` 是历史大指标面；保留指标，非当前 checkpoint 大头 |
| Phase D | 2026-05 | SchrodingerBridge/LANCET phase-space | grid/search/frontier/vae_backend/representation，很多是探索面；dry-run timing 不可引用 |
| Phase E | 2026-05-30 到 2026-06-02 | WikiArt512 和 Distinct5 formal evidence | 当前 timing/efficiency claim 的核心证据面；LBM vs SaMAM/SaMST/IDT/no-op |
| Phase F | 2026-06-03 后 | AAAI2027 / TokenizerClean claim-closing | K-longer、path-stability、endpoint/SA-SWD/tokenizer claims；远程 TokenizerClean 仍需 citation graph |

## Timing 结论

可信 timing 来源：

- `manual_timing_evidence_20260605.csv`
- `SchrodingerBridge/docs/experiments/2026-06-04-distinct5_same_cost_inventory.csv`
- `SchrodingerBridge/docs/experiments/2026-06-02-wikiart512-inference-speed.md`
- `SchrodingerBridge/docs/experiments/2026-06-05-timing-sidecar-inventory.md`
- `Related_Works/results/metrics_summary/timing_summary.csv`
- remote TokenizerClean `aaai2027_master_experiment_log.csv`

重要读法：

- Distinct5 LBM retained points是分钟级 compact evidence，和 SaMAM/SaMST 小时级 baseline cost 不是严格 time-to-parity closure。
- K-longer e5-e8 已完成 ArtFID summary，但仍未过 `+0.006` transfer CLIP-S gate，属于 negative longer-train evidence。
- F-longer 已在 writing gate 中判定未过 retention gate。
- SaMAM 3000 有 `TRAIN_STEP_3000_WALL_SECONDS=3156.25` 和 `EVAL_STEP_3000_WALL_SECONDS=289.31`。
- `lambda_grid` / `step_count_sweep` dry-run `0.000/0.001s` 不能当训练/推理 timing。
- runtime-anomalous SA-SWD random arm只能 quality-only，不当 speed evidence。

缺口：

- DisDict 512 timing 没找到。
- legacy baselines timing 混杂，很多只有 metrics。
- SaMST 缺 matched time-to-parity curve。
- TokenizerClean H-family 缺 H e1 payload，L e1 是 successor family，不是同族 fallback。

## Cleanup 总结

已经完成的高价值清理：

| 区域 | 删除内容 | 释放 |
|---|---|---:|
| remote `SchrodingerBridge/exp` | 84 个非保留 epoch `.pt` | `4961.604 MB` |
| remote SaMAM `step_checkpoints` | 7 个 redundant `last*.ckpt` aliases | `1931.291 MB` |
| local archive/root misc | duplicate `Cycle-NCE.tar` + stale launcher residue | `1503.203 MB` |
| local root `eval_cache` | invalid `.incomplete` blob and empty temp dirs | `55.994 MB` |
| local scale dataset cache | failed `wikiart_81k` HF residue | `63.948 MB` |

仍不能删的主要大户：

- remote SaMAM 12 个 step checkpoints：完整 curve/cited/repair/last evidence。
- remote TokenizerClean `exp`: 需要 citation graph。
- remote latent backends / eval_cache / scale datasets：输入数据或评测依赖。
- remote `Cycle-NCE` / `experiments`: archive policy 未定。
- local data/latent/feature/eval cache：不是 checkpoint 垃圾。

## 8 小时级别后续计划

这个任务还应继续按 block 推进，不应再做 broad delete：

| block | 预计 | 目标 | 产物 |
|---|---:|---|---|
| 1 | 0.5h | 固化当前权威归纳入口、校验索引互相不矛盾 | 本文件、README、coverage matrix commit |
| 2 | 1.0h | TokenizerClean citation graph：docs/master log -> exp paths -> ckpt roles | `manual_tokenizerclean_citation_graph_*.csv/md` |
| 3 | 1.0h | TokenizerClean exp thinning policy，只处理 citation graph 证明可删的 checkpoint | policy CSV + cleanup ledger |
| 4 | 1.0h | Remote latent backend/cache 分类：哪些是 Distinct5/WikiArt512 当前输入，哪些是 obsolete backend | cache retention policy |
| 5 | 1.0h | Remote `Cycle-NCE` / `experiments` archive/rar policy，判定 archive duplicate vs unique history | archive policy + possible deletion ledger |
| 6 | 1.0h | Local `Related_Works` placeholders/dependency policy，明确 tiny fake eval 和 pretrained deps | baseline dependency ledger |
| 7 | 1.0h | Timing master 二次过滤：dry-run/anomalous/smoke/full-eval 分层 | timing quality CSV |
| 8 | 0.5h | 全局复核：remote剩余权重、本地剩余权重、CSV 可读性、git 范围、提交 | verification report + commit |

## 不能宣称完成的项目

- 没有逐项完成 TokenizerClean destructive thinning。
- 没有决定 remote latent backend cache 是否可删。
- 没有处理 remote `Cycle-NCE` / `experiments` archive/rar。
- 没有把所有 legacy baseline timing 都提升为 evidence-grade timing。
- 没有把“所有目录每个子目录”都开到同一深度；当前是按 disk/evidence 风险分层推进。

当前下一步应从 TokenizerClean citation graph 开始，因为它是剩余 checkpoint 大头里最可能继续释放空间、同时最容易误删当前引用证据的区域。

## 2026-06-05 TokenizerClean citation graph 与清理更新

已完成 remote TokenizerClean 的 citation graph 和一轮 checkpoint-only 清理：

- 远程根目录：`I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge`。
- 已覆盖 `exp` 下全部 145 个一级目录，不再只是 94 个带权重目录。
- 已打开并核对 TokenizerClean policy 文档、tokenizer restart 设计、main-table gap analysis、flow-loss ablation、SA-SWD ablation、tokenizer execution alignment、L-family successor packet、`aaai2027_master_experiment_log.csv`。
- 已生成 `manual_remote_tokenizerclean_exp_internal_evidence_20260605.csv`、`manual_remote_tokenizerclean_exp_citation_graph_all_20260605.csv`、`manual_remote_tokenizerclean_cleanup_policy_20260605.csv`。
- 已按 policy 删除 44 个“无引用、非 aaai2027、有 summary/metrics”的探索目录中的 checkpoint 文件：141 个 `.pt/.ckpt/.pth`，释放 `5198.991 MB`。
- 删除 ledger：`cleanup/manual_remote_tokenizerclean_uncited_checkpoint_cleanup_20260605.csv`。
- post-delete 复核：`manual_remote_tokenizerclean_exp_internal_evidence_after_cleanup_20260605.csv` 和 `manual_remote_tokenizerclean_remaining_weight_classes_after_cleanup_20260605.csv`。

删除后 TokenizerClean 剩余权重分布：

| 类别 | 目录数 | 剩余权重文件 | 剩余大小 | 处理 |
|---|---:|---:|---:|---|
| cited/docs/master/paper 命中 | 34 | 122 | `3813.414 MB` | 保留直到 citation graph 迁移 |
| 当前 `aaai2027_*` packet | 9 | 24 | `1451.217 MB` | 保留，后续只可做 packet-specific thinning |
| 无 summary 的 review candidate | 28 | 39 | `911.730 MB` | 未删，需要逐目录 owner review |
| 已删除候选 | 44 | 0 | `0 MB` | checkpoint 已清理，数据证据保留 |
| 无 checkpoint | 30 | 0 | `0 MB` | 本轮不动 |

仍未完成：

- 28 个无 summary 的剩余权重目录需要下一轮逐个打开 config/log 后决定是否删除。
- `diagnostics`、`tokenizer_control_probes` 等约 4.8GB 主要是 generated image/summary/metrics，不是 checkpoint，必须另立图像证据归档策略，不能混入 checkpoint cleanup。
- `aaai2027_tokenizer_localization_*` 当前 docs 引用未跟上，但内部是新近 formal packet 形态，先保留。
