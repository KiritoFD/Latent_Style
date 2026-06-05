# Timing Evidence Quality Pass - 2026-06-05

This pass separates usable timing evidence from archive-only timing. It does not convert training time units and does not fill missing values.

## Inputs

- `manual_timing_evidence_20260605.csv`: 69 manually checked rows.
- `manual_remote_tokenizerclean_timing_evidence_20260605.csv`: 1024 TokenizerClean `summary.json` wall-time rows.

Output:

- `timing_quality_master_20260605.csv`: 1093 rows.
- `timing_quality_summary_20260605.csv`: quality-class row counts.

## Quality Classes

| quality_class | claim_use | rows | meaning |
| --- | --- | ---: | --- |
| `full_eval_summary_wall_time_tokenizerclean` | `audit_full_eval_wall_time_only` | 744 | TokenizerClean full_eval `summary.json` wall-time rows with training time absent in this source table. |
| `quick_eval_or_probe_wall_time` | `audit_full_eval_wall_time_only` | 234 | Quick-eval/probe rows; useful for archaeology, not headline speed. |
| `full_eval_wall_time` | `candidate_claim_support_with_caveat` | 51 | Full-eval wall time rows that can support timing claims with the caveat that the infer field is eval wall_total, not pure generation. |
| `historical_timing_context` | `historical_context` | 28 | Legacy/Cycle-NCE/strict-protocol context; do not mix into current formal claims. |
| `partial_training_or_missing_eval` | `audit_only` | 20 | Partial, interrupted, or missing paired eval evidence. |
| `smoke_or_failed_probe` | `exclude_formal_claim` | 7 | Smoke/probe/failed inference rows. |
| `invalidated_or_negative_audit_only` | `audit_only` | 4 | Arms marked invalidated, non-probing, or negative closure. |
| `training_log_only` | `audit_training_cost_only` | 2 | Training timing exists, no paired eval/inference wall time. |
| `train_and_eval_wall_time` | `candidate_claim_support_with_caveat` | 2 | Both train and eval/inference fields present, original units preserved. |
| `runtime_anomalous_exclude_speed_claim` | `quality_only_or_anomaly` | 1 | Runtime explicitly anomalous; keep only as archaeology/anomaly evidence. |

## Practical Conclusion

Do not quote the full 1093-row timing set as formal speed evidence.

For paper/claim-facing timing, use only rows where `claim_use` is `candidate_claim_support_with_caveat`, then read `quality_reason` and `note` before quoting. There are 53 such rows in this pass: 51 full-eval wall-time rows plus 2 train-and-eval rows.

Rows tagged `audit_full_eval_wall_time_only` are still useful, but they are not clean training-cost evidence. In particular, TokenizerClean summary rows mostly record full-eval wall time from `summary.json`; training time remains blank unless a log explicitly records it.

Rows tagged `exclude_formal_claim`, `quality_only_or_anomaly`, `audit_only`, or `historical_context` should stay in archaeology tables but should not be used as current headline speed.

## Known Remaining Work

- This pass did not rewrite `SchrodingerBridge/docs/timing/training_inference_timing_master.csv`; it creates an archaeology quality overlay instead.
- A later pass should reconcile this quality overlay with the docs timing master and final paper-facing timing table.
- The classifier is conservative and string-based; any row promoted to claim-facing prose should still be source-opened before final use.
