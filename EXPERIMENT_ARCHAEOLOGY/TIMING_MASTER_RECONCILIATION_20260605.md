# Timing Master Reconciliation - 2026-06-05

This pass is read-only against `SchrodingerBridge\docs\timing\training_inference_timing_master.csv`. It creates sidecar reconciliation files under `EXPERIMENT_ARCHAEOLOGY` only.

## Inputs Opened

- `SchrodingerBridge\docs\timing\training_inference_timing_master.csv`: 419 rows.
- `EXPERIMENT_ARCHAEOLOGY\timing_quality_master_20260605.csv`: 1093 rows.
- `EXPERIMENT_ARCHAEOLOGY\TIMING_EVIDENCE_QUALITY_PASS_20260605.md`: quality policy for the overlay.

## Outputs

- `timing_reconciliation_summary_20260605.csv`
- `timing_candidate_claim_reconciliation_20260605.csv`
- `timing_docs_master_overlay_reconciliation_20260605.csv`

## Main Counts

| item | count | meaning |
| --- | ---: | --- |
| docs timing master rows | 419 | Existing docs table rows; not modified here. |
| timing quality overlay rows | 1093 | Archaeology timing evidence rows. |
| overlay candidate claim rows | 53 | Rows with `claim_use=candidate_claim_support_with_caveat`. |
| overlay candidate rows already in docs by normalized source path | 27 | Candidate rows represented in docs master. |
| overlay candidate rows missing from docs by normalized source path | 26 | Candidate rows not represented in docs master. |
| docs rows covered by overlay by normalized source path | 49 | Docs rows with a source-path match in overlay. |
| docs rows not covered by overlay by normalized source path | 370 | Existing docs timing rows that need their own source-open review before claim use. |

## Practical Conclusion

The docs timing master and the archaeology quality overlay are not equivalent. The overlay is broader and quality-classified, while the docs table has many paper-facing or historical rows not represented by the overlay. Do not treat either file alone as final.

The immediate paper/claim-facing gap is the 26 overlay candidate rows not represented in the docs timing master. They include Distinct5 LANCET/LBM short runs, SaMST e5, WikiArt512 epoch8 generation/eval rows, TokenizerClean audit rows, and remote phase-space ablations. These should only be promoted after source-open verification and owner approval.

The 370 docs rows not covered by the overlay should remain in docs, but any row used in prose should be source-opened because it has no archaeology quality label in this overlay.

## Unit Policy

Original units are preserved. Training time was not converted to seconds. Blank train/infer fields remain blank.

## Next Manual Step

Source-open the 26 missing candidate rows one by one, then decide whether to add them to a future paper-facing timing table. This pass intentionally does not edit `SchrodingerBridge\docs\timing\training_inference_timing_master.csv`.
