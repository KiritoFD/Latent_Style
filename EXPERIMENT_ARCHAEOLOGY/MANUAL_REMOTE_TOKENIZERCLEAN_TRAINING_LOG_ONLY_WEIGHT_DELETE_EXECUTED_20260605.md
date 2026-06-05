# Remote TokenizerClean Training-Log-Only Weight Delete Executed - 2026-06-05

Scope:

```text
I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp
```

This pass deleted checkpoint weights from five trained no-summary payload dirs
that were training-log-only and had no downstream or diagnostic evidence. It
did not delete experiment directories, `config.json`, or `logs\training_*.csv`.

## Deleted

Exact deleted checkpoint files: `7`.

Released space: `248.429 MB`.

| exp_dir | deleted files | released MB |
| --- | ---: | ---: |
| `tokenizer_t01_carrier_base_b160` | 2 | 87.408 |
| `wikiart_distinct5_ema_lancet_spectralstat_from_e8_e16_b56` | 1 | 43.629 |
| `wikiart512_ema_pair_budget_tokonly_e1_b80` | 1 | 15.044 |
| `wikiart512_ema_spectral_stat_full_e2_from_tok_b48` | 2 | 87.248 |
| `wikiart512_ema_tokenbudget_tokonly_e1_from_spectral_b48` | 1 | 15.110 |

## Retained

The following weights were deliberately retained:

- `wikiart_distinct5_ema_lancet_spectralstat_e2_b80\epoch_0001.pt`
- `wikiart_distinct5_ema_lancet_spectralstat_e2_b80\epoch_0002.pt`
- `wikiart512_ema_trueint_stylepush_tsw40_kin025_e1_b48\epoch_0001.pt`

Reason: the first directory is a downstream resume source for an evaluated
continuation; the second checkpoint is directly referenced by a diagnostic
summary with metrics.

## Verification

Post-delete verification passed all `20` checks:

- the 7 deleted checkpoints are absent
- each of the 5 parent directories still has `config.json`
- each of the 5 parent directories still has at least one `training_*.csv`
- all 3 external-evidence checkpoints remain present

Ledgers:

- `manual_remote_tokenizerclean_training_log_only_weight_delete_whitelist_20260605.csv`
- `manual_remote_tokenizerclean_training_log_only_weight_delete_execution_20260605.csv`
- `manual_remote_tokenizerclean_training_log_only_weight_post_delete_verify_20260605.csv`
- `manual_remote_tokenizerclean_training_log_only_remaining_weights_20260605.csv`

## Remaining Gap

The five training-log-only directories now retain metadata only. They still lack
summary/full-eval outputs, but they no longer hold checkpoint weights. The
remaining TokenizerClean work is to build a cited/current media manifest and to
handle any future summary recovery only for retained evidence-bearing payloads.
