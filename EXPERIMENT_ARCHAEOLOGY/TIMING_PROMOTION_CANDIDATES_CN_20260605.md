# Timing Promotion Candidates - 2026-06-05

本文件整理训练/推理时间的“可提升候选”层，供后续标定复用。它不修改 `SchrodingerBridge/docs/timing/training_inference_timing_master.csv`，也不修改论文、TeX、PDF、源码或 Related_Works 文件。

输出 CSV：

- `EXPERIMENT_ARCHAEOLOGY/timing_promotion_candidates_20260605.csv`

## 输入

本轮只合并两个已有证据表：

- `EXPERIMENT_ARCHAEOLOGY/timing_candidate_claim_reconciliation_20260605.csv`
- `EXPERIMENT_ARCHAEOLOGY/timing_candidate_missing_docs_source_open_20260605.csv`

其中缺 docs timing master 的 26 行已经在前序 pass 中按 exact source path 打开；已在 docs timing master 的 27 行本轮不重复打开，只标记为 `already_in_docs_verify_before_use`。

## 当前 timing 总体状态

`timing_quality_master_20260605.csv` 有 1093 行，质量类别分布如下：

| quality_class | rows | 解释 |
| --- | ---: | --- |
| `full_eval_summary_wall_time_tokenizerclean` | 744 | TokenizerClean full-eval summary wall time，主要 audit-only |
| `quick_eval_or_probe_wall_time` | 234 | quick/probe wall time，不用于 formal claim |
| `full_eval_wall_time` | 51 | 可作为候选，但必须说明是 full-eval wall |
| `historical_timing_context` | 28 | 历史上下文 |
| `partial_training_or_missing_eval` | 20 | 部分训练或缺 eval |
| `smoke_or_failed_probe` | 7 | 排除 formal claim |
| `invalidated_or_negative_audit_only` | 4 | 负例或无效审计 |
| `training_log_only` | 2 | 只有训练成本 |
| `train_and_eval_wall_time` | 2 | 训练和 eval/generation wall 同时存在 |
| `runtime_anomalous_exclude_speed_claim` | 1 | 运行时异常，排除 speed claim |

按 `claim_use` 分布：

| claim_use | rows | 解释 |
| --- | ---: | --- |
| `audit_full_eval_wall_time_only` | 978 | 只做审计，不进主 claim |
| `candidate_claim_support_with_caveat` | 53 | 本文件处理的候选层 |
| `historical_context` | 28 | 历史上下文 |
| `audit_only` | 24 | 审计-only |
| `exclude_formal_claim` | 7 | 排除 |
| `audit_training_cost_only` | 2 | 只做训练成本 |
| `quality_only_or_anomaly` | 1 | 质量或异常说明 |

## 53 行候选的 promotion 桶

| docs_promotion_decision | rows | 使用边界 |
| --- | ---: | --- |
| `already_in_docs_verify_before_use` | 27 | 已在 docs timing master；写进论文前仍要源文件复核 |
| `retain_archaeology_only_trajectory_ablation` | 12 | remote phase/compact/path trajectory ablation，只做考古和轨迹证据 |
| `owner_review_before_docs_promotion` | 5 | TokenizerClean current audit packet，需要 owner 决定 |
| `promote_candidate_with_caveat` | 3 | 可提升候选，但 caveat 必须保留 |
| `promote_timing_note_with_caveat` | 3 | timing note 支撑，可提升为 timing note，不等同纯推理 |
| `promote_only_if_owner_accepts_missing_artfid_packet` | 2 | 指标闭合但缺 retained targetwise ArtFID packet |
| `retain_archaeology_only_nonmainline_calibration` | 1 | local calibration，非主线，只保留考古 |

## 可提升候选

这些行可以作为后续标定复用候选，但不代表已经写入 docs master：

| source_row | method | dataset | train | infer | 决策 |
| ---: | --- | --- | --- | --- | --- |
| 1 | LANCET/LBM F e1 | Distinct5-512 | 1.2161 min | 90.82485276100124 s | compact LPIPS anchor；infer 是 full-eval wall |
| 4 | LANCET/LBM K e1 | Distinct5-512 | 1.2077 min | 101.0314540090003 s | compact style anchor；infer 是 full-eval wall |
| 8 | SaMST e5 | Distinct5-512 | 115.9750 min | 323.071 s | baseline packet；infer 是 750 image generation wall |
| 24 | LANCET/LBM WikiArt512 epoch8 eval | WikiArt512-5style |  | 210.67 s | full-eval wall，不是纯 generation |
| 44 | LANCET/LBM WikiArt512 epoch8 generation-only | WikiArt512-5style |  | 54.80 s | 750 PNG generation-only external wall |
| 45 | LANCET/LBM WikiArt512 from-scratch e8 | WikiArt512-5style | 66.56 s | 55.16 s | note-backed external timing；direct full eval 另有 106.62 s |

## 可提升但缺 owner/packet 确认

| source_row | method | dataset | train | infer | 缺口 |
| ---: | --- | --- | --- | --- | --- |
| 2 | LANCET/LBM H e1 | Distinct5-512 | 1.2207 min | 97.59912403999988 s | closed metrics，但缺 retained targetwise ArtFID packet |
| 3 | LANCET/LBM H e2 | Distinct5-512 | 2.2656 min | 95.98756044499896 s | closed metrics，但缺 retained targetwise ArtFID packet |

这两行不是“无效”，但如果要进入 docs timing/paper-facing 表，需要 owner 接受 ArtFID packet 缺口，或者补回 retained indexed packet。

## 当前审计证据，先不提升

TokenizerClean rows 50-54 是 current audit evidence，但本轮不自动提升：

| source_row | method | train | infer | 结论 |
| ---: | --- | --- | --- | --- |
| 50 | endpoint-metric Huber e1 | 291.682 s | 97.276 s | repaired endpoint-metric packet；owner review |
| 51 | endpoint-metric L1 e3 | 290.102 s | 100.001 s | e3 是 best LPIPS point，但仍 owner review |
| 52 | tokenizer localization stylebranch e2 | 160.100 s | 151.641 s | current evidence；checkpoints retained |
| 53 | tokenizer localization executoronly e3 | 179.850 s | 87.606 s | current evidence；checkpoints retained |
| 54 | SA-SWD semantic e1 | 198.769 s | 96.894 s | path-quality caveat，special-char root path |

这些行可以继续保留为 current audit/timing evidence，但不能在没有 owner 决策时提升为最终 docs timing rows。

## 只保留考古的 timing

`retain_archaeology_only_nonmainline_calibration`：

- source_row 43：`LANCET/LBM local ckptsync K step350`
- train `100.500922 s`
- infer `78.317428 s`
- 结论：local WSL calibration，三枚非主线权重已删除；保留 logs/images/summary，不能当主线 speed claim。

`retain_archaeology_only_trajectory_ablation`：

- source_rows 56-65、67-68
- 覆盖 remote phase1 ablation、compact ablation、path kinetic packets。
- 训练时间约 `198.939151 s` 到 `525.351942 s`，full-eval wall 约 `88.408188 s` 到 `95.980974 s`。
- 结论：这些是 trajectory/ablation evidence，不是当前 compact manuscript anchor。

## 已在 docs timing master 的 27 行

这 27 行已被 `SchrodingerBridge/docs/timing/training_inference_timing_master.csv` 表示，本轮只标记为：

```text
already_in_docs_verify_before_use
```

它们覆盖：

- Distinct5 longer F/K、SaMAM step 3000。
- AAAI2027 endpoint metric Huber/L1/MSE。
- AAAI2027 flow loss Huber/L1/MSE。
- AAAI2027 tokenizer localization executoronly/stylebranch。

注意：`already_in_docs` 不等于可以不复核。写进论文前仍要打开对应 summary/log，确认时间列是 full-eval wall、generation wall、训练 wall，还是其他 wall time。

## 当前结论

- 可直接进入下一步 promotion review 的核心是 6 行：source_rows 1、4、8、24、44、45。
- 需要 owner/packet 决策的是 7 行：source_rows 2、3、50-54。
- 只保留考古的是 13 行：source_row 43 和 56-65、67-68。
- 已在 docs master 但仍需用前复核的是 27 行。

## 仍缺

- 没有修改 docs timing master；后续如果要写入，需要单独开 docs timing update。
- WikiArt from-scratch 的 `66.56 s` 和 `55.16 s` 是 timing note backed external wall；打开的训练 CSV/summary 没有直接同字段 wall 值。
- Full-eval wall time 不可写成纯推理耗时。
- SaMST e5 的训练 `115.9750 min` 与 packet seconds 一致，但 params_m 仍空。
- 370 条 docs master rows 还未完成同等 source-open 审查。
