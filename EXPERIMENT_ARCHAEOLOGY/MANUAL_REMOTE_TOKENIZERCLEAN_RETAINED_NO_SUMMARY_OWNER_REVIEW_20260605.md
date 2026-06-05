# Remote TokenizerClean Retained No-Summary Owner Review - 2026-06-05

Scope: the 10 TokenizerClean `exp` directories that survived the earlier no-summary cleanup pass.

Remote root:
`I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge`

This is a second-pass manual review. The first pass kept these directories because they lacked `summary.json`, so the checkpoints could be the only payload. This pass re-opened each directory on the remote host and cross-checked the citation graph plus diagnostics evidence.

## Current facts

- All 10 directories have `total_hit_count=0` in `manual_remote_tokenizerclean_exp_citation_graph_all_20260605.csv`.
- 3 directories contain only orphan probe weights: no config, no log, no training CSV, no summary-like file, no citation.
- 7 directories contain `config.json`, `logs/training_*.csv`, `numeric_debug.jsonl`, `src`, and one or two epoch checkpoints. These are real trained payloads without summary, not pure orphan files.
- Diagnostics evidence exists for the 3 orphan probe families under `exp\diagnostics`.

## Delete whitelist

The following three directories are approved for checkpoint deletion in this pass:

- `axis_scale_probe`: 6 weights, `90.265951 MB`; diagnostics output includes axis-scale n6 directories plus `axis_scale_probe_n6_summary.json`.
- `field_budget_release_probe`: 2 weights, `30.089247 MB`; diagnostics output includes `field_budget_release_release_s100_n6` and `field_budget_release_release_s150_n6`.
- `pair_relative_geometry_release_probe`: 3 weights, `49.661502 MB`; diagnostics output includes pairrel n6 directories and logs.

Deletion scope is only the orphan weight files and the now-empty directory. Diagnostics outputs remain.

## Retained payloads

The following seven directories remain retained because they have config and training CSV evidence but no summary:

- `tokenizer_t01_carrier_base_b160`
- `wikiart_distinct5_ema_lancet_spectralstat_e2_b80`
- `wikiart_distinct5_ema_lancet_spectralstat_from_e8_e16_b56`
- `wikiart512_ema_pair_budget_tokonly_e1_b80`
- `wikiart512_ema_spectral_stat_full_e2_from_tok_b48`
- `wikiart512_ema_tokenbudget_tokonly_e1_from_spectral_b48`
- `wikiart512_ema_trueint_stylepush_tsw40_kin025_e1_b48`

These are not deletion targets until either a summary is recovered/generated or owner review confirms the checkpoints are disposable.

## Files produced

- `manual_remote_tokenizerclean_retained_no_summary_owner_review_20260605.csv`: current remote directory evidence for the 10 retained no-summary directories.
- `manual_remote_tokenizerclean_orphan_probe_diagnostics_20260605.csv`: diagnostics cross-check for the three orphan probe families.
- `manual_remote_tokenizerclean_retained_no_summary_owner_policy_20260605.csv`: second-pass policy and delete whitelist.

## Remaining gap after this pass

After deleting the three pure-orphan probe directories, seven trained no-summary payload directories remain. They should not be deleted by size or by missing summary alone; they need summary recovery/evaluation or explicit owner decision.
