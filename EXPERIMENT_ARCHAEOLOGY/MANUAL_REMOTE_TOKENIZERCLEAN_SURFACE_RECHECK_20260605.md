# Remote TokenizerClean Surface Recheck - 2026-06-05

Remote:

```text
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62
I:\Github\Latent_Style_TokenizerClean
```

This block rechecks the separate TokenizerClean tree. No remote deletion was
performed.

## Current Shape

| path | current finding | decision |
| --- | --- | --- |
| `I:\Github\Latent_Style_TokenizerClean` | 17 top-level dirs, 37 top-level files | separate evidence tree |
| `SchrodingerBridge\exp` | 142 dirs, 23 files | keep exact-path review boundary |
| cited/current media manifest | 26 dirs, 46483 media files, 11977.341 MB, 118 weights | retain until owner archive/migration |
| training-log-only live recheck | 7 dirs, 3 remaining weights, 130.883 MB | mixed retained/meta-only after cleanup |
| post-delete verification | 20 checks | all pass |
| missing resume anomaly | resume points to absent `epoch_0004.pt` | metadata-only, not clean lineage |

Row-level CSV:

- `manual_remote_tokenizerclean_surface_recheck_20260605.csv`

## What Was Manually Reopened

- Top-level TokenizerClean root and `SchrodingerBridge\exp`.
- Existing cited/current media manifest:
  `manual_remote_tokenizerclean_cited_current_media_manifest_20260605.csv`.
- Existing training-log-only live recheck:
  `manual_remote_tokenizerclean_training_log_only_live_recheck_20260605.csv`.
- Existing post-delete verification:
  `manual_remote_tokenizerclean_training_log_only_weight_post_delete_verify_20260605.csv`.
- Existing missing resume anomaly:
  `manual_remote_tokenizerclean_missing_resume_anomaly_20260605.csv`.

## Largest Retained Media Packets

| exp_dir | MB | media | weights | decision |
| --- | ---: | ---: | ---: | --- |
| `diagnostics` | 2872.709 | 7723 | 0 | retain_large_cited_diagnostics_surface |
| `wikiart512_ema_spectral_stat_full_adapt_e2_b48` | 2111.208 | 3608 | 8 | retain_large_cited_model_packet |
| `tokenizer_t01_carrier_base_b176_e16` | 739.909 | 2253 | 16 | retain_tokenizer_t01_cited_packet |
| `aaai2027_flow_loss_h_base_l1_seed42_b44` | 430.316 | 2253 | 3 | retain_full_packet_until_owner_archive_selection |
| `aaai2027_flow_loss_h_base_mse_seed42_b44` | 430.287 | 2253 | 3 | retain_full_packet_until_owner_archive_selection |

## Cleanup Boundary

All rows remain `delete_whitelist=no`.

The remaining TokenizerClean cleanup is not a blind delete task:

- The 26 cited/current media dirs need an owner archive/migration decision.
- The 3 remaining weights are evidence-bearing or downstream-resume payloads;
  deleting them requires a new checkpoint policy.
- The missing-resume directory is retained as metadata-only archaeology and
  must not be promoted as a clean evaluated result.
