# Remote Cycle-NCE 45.rar Manual Review - 2026-06-05

Remote root:
`I:\Github\Latent_Style`

Archive:
`Cycle-NCE\45.rar`

## Scope

This is the retained RAR/archive gap after deleting:

- exact duplicate `Cycle-NCE\src\45.rar`
- weight-only `Gate.rar`
- weight-only `Attn_48.part*.rar`
- weight-only `chess.part*.rar`
- resolved duplicate `experiments.rar`

## Manual Findings

`Cycle-NCE\45.rar` is not a duplicate archive under the current remote tree.

- `Cycle-NCE\45.rar` exists.
- Expanded directory `Cycle-NCE\45` does not exist.
- Archive file entries: `6096`.
- Historical run directories: `4`.
- Weight files: `12`, totaling `423.786869 MB`.
- Generated/eval images: `6008`.
- Config files: `4`.
- Summary JSON files: `8`.
- Metrics CSV files: `8`.
- Root `ma_probe_all_pairs.*` files: `3`, totaling about `4.503393 MB`.

## Opened Evidence

The archive was opened with temporary remote `UnRAR.exe`; no full extraction was performed.

Generated files:

- `manual_remote_cycle_nce_45_rar_run_ledger_20260605.csv`
- `manual_remote_cycle_nce_45_rar_entry_classes_20260605.csv`
- `manual_remote_cycle_nce_45_rar_text_evidence_20260605.csv`
- `manual_remote_cycle_nce_45_rar_summary_metrics_20260605.csv`
- `manual_remote_cycle_nce_45_rar_summary_overview_20260605.csv`
- `manual_remote_cycle_nce_45_rar_policy_20260605.csv`

The text evidence CSV directly opens each run's `config.json`, `summary.json`, and `metrics.csv` via `UnRAR p`. The summary metrics CSV parses all 8 `summary.json` files into metric-path rows with no parse errors.

## Decision

Keep `Cycle-NCE\45.rar` for now.

Reason: it contains unique nonweight experiment evidence, not only stale weights. Deleting the archive would remove configs, metrics, summaries, generated/eval images, and root ma-probe artifacts.

## Future Cleanup Option

If space pressure requires deleting it, the safe path is a separate whitelist block:

1. Extract a curated nonweight evidence package for `45`.
2. Verify configs, summaries, metrics, and selected images are present outside the archive.
3. Decide whether full generated-image retention is required.
4. Delete the archive only after the retained evidence package is accepted.

Do not delete `45.rar` by extension, size, or because it contains old `.pt` files.
