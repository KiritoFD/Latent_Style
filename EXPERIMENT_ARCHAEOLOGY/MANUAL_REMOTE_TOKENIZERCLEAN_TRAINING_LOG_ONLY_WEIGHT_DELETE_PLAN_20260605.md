# Remote TokenizerClean Training-Log-Only Weight Delete Plan - 2026-06-05

Scope:

```text
I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp
```

This plan narrows the previous no-summary owner review into a checkpoint-only
cleanup. It does not delete experiment directories. It deletes only exact `.pt`
checkpoint files from the five trained no-summary payload dirs that have:

- `config.json`
- at least one `logs\training_*.csv`
- no in-dir non-source `summary.json`, `metrics.csv`, or `full_eval`
- no external downstream/diagnostic evidence

The config and training CSV remain as the archaeology record.

## Delete Whitelist

Delete exactly 7 checkpoint files:

| exp_dir | files | MB |
| --- | ---: | ---: |
| `tokenizer_t01_carrier_base_b160` | 2 | 87.408 |
| `wikiart_distinct5_ema_lancet_spectralstat_from_e8_e16_b56` | 1 | 43.629 |
| `wikiart512_ema_pair_budget_tokonly_e1_b80` | 1 | 15.044 |
| `wikiart512_ema_spectral_stat_full_e2_from_tok_b48` | 2 | 87.248 |
| `wikiart512_ema_tokenbudget_tokonly_e1_from_spectral_b48` | 1 | 15.110 |

Expected release: `248.429 MB`.

Whitelist CSV:

- `manual_remote_tokenizerclean_training_log_only_weight_delete_whitelist_20260605.csv`

## Retain Boundary

Do not delete weights from:

- `wikiart_distinct5_ema_lancet_spectralstat_e2_b80`: retained as a downstream
  resume source for an evaluated continuation.
- `wikiart512_ema_trueint_stylepush_tsw40_kin025_e1_b48`: retained because a
  diagnostic summary directly references `epoch_0001.pt` and reports metrics.

Do not delete any `config.json`, `logs\training_*.csv`, source snapshots, or
directories in this pass.

## Required Post-Delete Verification

For each deleted checkpoint:

- checkpoint path absent
- parent `config.json` present
- parent has at least one `logs\training_*.csv`

For retained external-evidence dirs:

- their checkpoint files remain present
