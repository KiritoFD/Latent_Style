# README / Count Consistency Audit - 2026-06-05

This pass checks current high-level counts and README/index coverage after the
latest source-open and policy blocks.

No cleanup was performed.

## Current Count Checks

| item | current count | source |
| --- | ---: | --- |
| final master experiment rows | 22629 | `final_master_experiments.csv` |
| final timeline events | 7829 | `final_timeline.csv` |
| dataset conclusion rows | 25 | `conclusions_by_dataset.csv` |
| timing quality overlay rows | 1093 | `timing_quality_master_20260605.csv` |
| docs timing master rows | 419 | `SchrodingerBridge/docs/timing/training_inference_timing_master.csv` |
| missing-docs timing source-open rows | 26 | `timing_candidate_missing_docs_source_open_20260605.csv` |
| TokenizerClean cited/current media policy rows | 26 | `manual_remote_tokenizerclean_cited_current_media_archive_policy_20260605.csv` |
| TokenizerClean trained no-summary owner rows | 7 | `manual_remote_tokenizerclean_trained_no_summary_owner_decision_20260605.csv` |
| Cycle-NCE 45.rar extraction class rows | 8 | `manual_remote_cycle_nce_45_rar_curated_extraction_policy_20260605.csv` |
| current status requirement rows | 8 | `archaeology_current_status_requirements_20260605.csv` |

## README Coverage

README now includes entries for the current high-signal reports:

- `ARCHAEOLOGY_CURRENT_STATUS_AND_CONCLUSIONS_ZH_20260605.md`
- `TIMING_CANDIDATE_MISSING_DOCS_SOURCE_OPEN_20260605.md`
- `MANUAL_REMOTE_TOKENIZERCLEAN_CITED_CURRENT_MEDIA_POLICY_20260605.md`
- `MANUAL_REMOTE_TOKENIZERCLEAN_TRAINED_NO_SUMMARY_OWNER_DECISION_20260605.md`
- `MANUAL_REMOTE_CYCLE_NCE_45_RAR_CURATED_EXTRACTION_POLICY_20260605.md`

## Direct Index Coverage

`archaeology_direct_conclusions_index_20260605.csv` now includes these current
areas:

- `local`
- `remote-main`
- `remote-main-45rar`
- `cache-dedup`
- `remote-tokenizerclean`
- `remote-tokenizerclean-no-summary-owner`
- `remote-tokenizerclean-media`
- `lineage`
- `timing`
- `timing-source-open`
- `cleanup-ledger`
- `current-status`
- `readme-count-consistency`

## Remaining Inconsistency

The README count section is a mixed historical summary. It still has useful
top-level counts, but a single line such as `Timing rows: 416` is no longer
sufficient after the later timing reconciliation/source-open passes.

Current timing should be read as three separate counts:

- docs timing master: 419 rows
- archaeology timing overlay: 1093 rows
- source-opened missing-docs claim candidates: 26 rows

The newest human-readable entry point is now:

```text
ARCHAEOLOGY_CURRENT_STATUS_AND_CONCLUSIONS_ZH_20260605.md
```

The old reports remain useful evidence, but this current-status report is the
first place to read after this pass.
