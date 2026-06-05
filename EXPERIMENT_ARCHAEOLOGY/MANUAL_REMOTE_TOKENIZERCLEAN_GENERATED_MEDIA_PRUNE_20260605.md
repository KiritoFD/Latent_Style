# Remote TokenizerClean generated media prune - 2026-06-05

## Scope

Remote root:

`I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp`

This pass handled generated visual payloads after checkpoint cleanup. It did not delete checkpoints, summaries, metrics, configs, or logs.

## Prior policy checked

Opened remote docs:

- `docs/experiments/2026-06-03-exploratory-image-prune.md`
- `docs/experiments/2026-06-03-repo-cleanup-and-archive-pass.md`
- `docs/experiments/2026-06-03-timing-artifact-prune.md`

The existing repo rule is:

1. delete generated image payloads only;
2. keep `summary.json`, `metrics.csv`, ledgers, checkpoints, logs, and config snapshots;
3. do not touch directories whose generated images are directly referenced by paper or current docs.

## Inputs

- `manual_remote_tokenizerclean_generated_media_inventory_20260605.csv`
- `manual_remote_tokenizerclean_exp_citation_graph_all_20260605.csv`
- `manual_remote_tokenizerclean_generated_media_cleanup_policy_20260605.csv`

The inventory counted media extensions:

- `.png`
- `.jpg`
- `.jpeg`
- `.webp`
- `.gif`

## Policy

Generated media directories were classified as:

| class | dirs | media files before deletion | media size before deletion | action |
|---|---:|---:|---:|---|
| `delete_uncited_summary_backed_media` | 50 | 43008 | 11883.246 MB | delete media only |
| `keep_cited_media` | 18 | 28459 | 5704.518 MB | keep |
| `keep_current_aaai2027_media` | 8 | 18024 | 1797.000 MB | keep |
| `no_media` | 69 | 0 | 0 MB | no action |

Deletion criteria:

- zero docs/reviews/master/paper citation hits;
- at least one `summary.json` or CSV metrics file;
- no checkpoint files in the selected media deletion candidates;
- delete only image/media extensions.

## Deleted

Deleted:

- 43008 media files.
- 11883.246 MB.

Ledger:

- `cleanup/manual_remote_tokenizerclean_uncited_generated_media_cleanup_20260605.csv`
- `cleanup/manual_remote_tokenizerclean_uncited_generated_media_cleanup_by_dir_20260605.csv`

Largest deleted groups:

| exp dir | files | reclaimed |
|---|---:|---:|
| `tokenizer_control_probes` | 4125 | 1974.441 MB |
| `wikiart512_ema_direct_atom_residual_e8_b48` | 3030 | 1478.104 MB |
| `metric_tokenizer_init` | 2710 | 1320.787 MB |
| `spatial_prototype_init` | 1057 | 525.502 MB |
| `wikiart512_ema_truegrad_tokenbudget_full_e1_b32` | 1053 | 516.105 MB |
| `wikiart512_ema_direct_atom_residual_continue_e12_from_e8_b48` | 959 | 475.643 MB |
| `wikiart512_ema_tokenbudget_gradfix_tokonly_e1_b16` | 902 | 442.295 MB |
| `moment_sweep_spectral_full` | 604 | 297.057 MB |
| `field_scale_probe` | 304 | 150.476 MB |
| `full_eval_history_backfill_20260601` | 7510 | 108.459 MB |

## Post-delete verification

Post-delete inventory:

- `manual_remote_tokenizerclean_generated_media_inventory_after_cleanup_20260605.csv`
- `manual_remote_tokenizerclean_remaining_media_classes_after_cleanup_20260605.csv`

Verification result:

- all 50 `delete_uncited_summary_backed_media` directories now have `0` remaining media files;
- remaining media is concentrated in cited/current/paper-facing classes.

Remaining media:

| class | dirs | remaining media files | remaining media size |
|---|---:|---:|---:|
| `keep_cited_media` | 18 | 28459 | 5704.518 MB |
| `keep_current_aaai2027_media` | 8 | 18024 | 1797.000 MB |
| deleted/none classes | 119 | 0 | 0 MB |

Examples of retained media:

- `diagnostics`: cited by docs and paper sources.
- `configs`: cited by docs/master/reviews.
- `aaai2027_*`: current formal packets.
- `pareto_probe_4`, `orth12`, `runs`, `legacy`: cited by current docs/reviews.

## Remaining gap

No further generated-media deletion should happen in TokenizerClean without one of:

1. citation migration for cited dirs;
2. packet-specific archive policy for `aaai2027_*`;
3. paper figure asset audit for `diagnostics`.
