# Remote TokenizerClean 人工引用图与 checkpoint 清理记录 - 2026-06-05

## 范围

远程根目录：

`I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge`

本轮只处理 `exp` 下的实验面，目标是把上一次“只看带权重目录”的草稿扩展成 145 个一级目录的人工证据图，并只删除已经满足 policy 的非主线 checkpoint。未碰论文 `tex/pdf`，未改远程源码，未删除 summary、metrics、CSV、日志或生成图。

## 手工打开过的核心来源

- `docs/experiments/2026-06-03-exp-surface-classification.md`：确认 TokenizerClean 的根规则是“被 docs/reviews/master log 引用的 `exp/...` 路径不能 mass delete，必须先做 citation graph”。
- `docs/experiments/2026-05-30-tokenizer-restart-design.md`：确认 tokenizer 工作线来自 `t01`/EC endpoint 选择，`tokenizer_t01_*` 属于冻结探索或表示探针，不是可直接当论文主证据的目录。
- `docs/experiments/2026-06-01-main-table-gap-analysis.md`：确认当前主表强点、SaMAM/SaMST timing 缺口，以及 tokenizer/representation 后续方向。
- `docs/experiments/2026-06-03-flow-loss-metric-ablation/README.md`：确认原始 flow-loss trio 因 `w_flow=0.0` 失效，repaired endpoint packet 是当前正式负结论包。
- `docs/experiments/2026-06-03-saswd-axis-ablation/README.md`：确认 semantic/random SA-SWD 包已 landed，但 random arm 是 runtime anomaly，只能作 quality-only evidence。
- `docs/experiments/2026-06-03-tokenizer-execution-alignment/README.md`：确认 H-family e1 缺 checkpoint，不能静默替换。
- `docs/experiments/2026-06-03-tokenizer-execution-alignment-l-family/README.md`：确认 L-family 是 successor packet，不是 H-family fallback。
- `docs/experiments/aaai2027_master_experiment_log.csv`：确认 invalid/repaired/SA-SWD/tokenizer/time-to-parity/path-stability 的 claim safety band。

## 产出的索引

- `manual_remote_tokenizerclean_exp_internal_evidence_20260605.csv`：删除前 145 个 `exp` 一级目录的内部证据，包含权重数、summary 数、config 关键项、summary 核心指标、训练日志尾部。
- `manual_remote_tokenizerclean_exp_citation_graph_all_20260605.csv`：145 个一级目录对 `docs/` 和 `aaai_submission/` 的引用图。
- `manual_remote_tokenizerclean_cleanup_policy_20260605.csv`：145 个目录的保留/删除 policy。
- `cleanup/manual_remote_tokenizerclean_uncited_checkpoint_cleanup_20260605.csv`：逐文件删除 ledger。
- `cleanup/manual_remote_tokenizerclean_uncited_checkpoint_cleanup_by_dir_20260605.csv`：按目录汇总的删除 ledger。
- `manual_remote_tokenizerclean_exp_internal_evidence_after_cleanup_20260605.csv`：删除后复核快照。
- `manual_remote_tokenizerclean_remaining_weight_classes_after_cleanup_20260605.csv`：删除后剩余权重分类。

## Policy

本轮把 145 个目录分为 5 类：

| 类别 | 目录数 | 删除前权重 | 处理 |
|---|---:|---:|---|
| `keep_cited_until_citation_migrated` | 34 | 3813.414 MB | 被 docs/reviews/master/paper 命中，保留 |
| `keep_current_aaai2027_packet` | 9 | 1451.217 MB | 当前或新近 AAAI packet，保留 |
| `delete_ckpt_candidate_uncited_with_summary` | 44 | 5198.996 MB | 无引用、非 aaai2027、有 summary/metrics，删除 checkpoint，只留数据 |
| `review_ckpt_candidate_no_summary` | 28 | 911.730 MB | 无引用但没有 summary，先不删 |
| `no_checkpoint_delete` | 30 | 0 MB | 没有 checkpoint，本轮不动 |

删除规则只覆盖 `.pt/.ckpt/.pth`，并且删除脚本逐个校验目标文件必须位于 `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\<selected_dir>\` 下。没有做目录级递归删除。

## 已删除

本轮远程删除：

- 141 个 checkpoint 文件。
- 合计释放 `5198.991 MB`。
- 删除范围：44 个“无引用、非 aaai2027、有 summary/metrics”的探索目录。

删除最多的目录：

| 目录 | 删除文件数 | 释放 |
|---|---:|---:|
| `wikiart_distinct5_ema_lancet_spectralstat_from_e16_e24_b48` | 24 | 1047.096 MB |
| `wikiart_distinct5_ema_lancet_spectralstat_from_e8_e16_b48` | 16 | 698.064 MB |
| `wikiart512_ema_direct_atom_residual_continue_e12_from_e8_b48` | 12 | 523.080 MB |
| `metric_tokenizer_init` | 11 | 365.582 MB |
| `spatial_prototype_init` | 8 | 349.134 MB |
| `wikiart_distinct5_ema_lancet_spectralstat_from_e2_e8_b64` | 8 | 349.032 MB |
| `wikiart512_ema_direct_atom_residual_e8_b48` | 8 | 348.720 MB |

这些目录的 summary/metrics/png/csv/log/config 均保留；后续仍能读到实验结果，但不能从这些非主线 checkpoint 继续训练。

## 保留原因

### 当前或被引用的权重

删除后仍保留：

- `keep_cited_until_citation_migrated`：122 个权重，3813.414 MB。
- `keep_current_aaai2027_packet`：24 个权重，1451.217 MB。

保留理由：

- `tokenizer_t01_*` 中部分目录被 tokenizer restart 设计文档引用，是冻结探索证据。
- `wikiart512_ema_spectral_stat_full_adapt_e2_b48` 是 main-table gap analysis 中的当前 representation/tokenizer 基础点。
- `aaai2027_endpoint_metric_*` 是 repaired endpoint metric packet，虽然是负结论，仍是正式 current claim packet。
- `aaai2027_flow_loss_h_base_*` 是 invalidated operational control；因文档明确讨论，不能删。
- `saswd_axis_*`、`aaai2027_tokenizer_execution_alignment_l_e1` 等被 master log/docs 命中，保留。

### 无权重但占空间的目录

这些目录没有 checkpoint，所以本轮未删：

- `diagnostics`：约 2872.709 MB，主要是 quick/full eval 生成图、summary、metrics，0 个权重。
- `tokenizer_control_probes`：约 1977.872 MB，主要是 generated PNG、summary、metrics，0 个权重。
- `configs`：约 368.559 MB，含 phase-space sweep 配置和 full_eval 结果，被 docs/master 命中。
- `moment_sweep_spectral_full`、`pareto_probe_4`、`field_scale_probe`：主要是生成图和 summary/metrics，需要单独的 generated-image archive policy，不能混入 ckpt 清理。

### 无 summary 的剩余权重

删除后仍有 28 个无引用但无 summary 的权重目录，合计 39 个文件、911.730 MB。它们没有 summary 可作为结果替代，所以本轮只列入 review，不直接删。典型项：

- `axis_scale_probe`：6 个权重，90.266 MB。
- `tokenizer_t01_carrier_base_b160`：2 个权重，87.408 MB。
- `wikiart_distinct5_ema_lancet_spectralstat_e2_b80`：2 个权重，87.258 MB。
- `wikiart512_ema_spectral_stat_full_e2_from_tok_b48`：2 个权重，87.248 MB。
- `pair_relative_geometry_release_probe`：3 个权重，49.662 MB。

下一轮如果继续清理，应逐个打开这些目录的 config/log，确认是否只有失败/校准记录；如果没有任何 metrics 或 doc lineage，再按 no-summary policy 删除。

## 结论

这次不是“脚本扫完就算看过”。实际做了三层核验：

1. 先打开 TokenizerClean 的 policy 文档和 master log，确认哪些目录不能动。
2. 再对 145 个 `exp` 一级目录生成内部证据表和引用图。
3. 最后只对“无引用、非当前包、有 summary/metrics”的非主线目录删除 checkpoint，并做 post-delete 复核。

当前 TokenizerClean 仍未完成的缺口是：

- 28 个无 summary 的剩余权重目录还需要逐个 owner review。
- 约 4.8GB 的 generated image/diagnostic 目录不是 checkpoint，需要另设图像证据归档/删除 policy。
- `aaai2027_tokenizer_localization_*` 目前 docs 引用没有跟上，但命名和内部结构显示它是新近 formal packet，先保留。
