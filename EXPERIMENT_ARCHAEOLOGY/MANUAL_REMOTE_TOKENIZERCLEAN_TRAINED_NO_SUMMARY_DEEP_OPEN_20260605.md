# Remote TokenizerClean Trained No-Summary Deep Open - 2026-06-05

Scope:

```text
I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp
```

This pass manually reopened the 7 retained trained no-summary directories from
`MANUAL_REMOTE_TOKENIZERCLEAN_TRAINED_NO_SUMMARY_OWNER_DECISION_20260605.md`.
No deletion was performed.

Output table:

- `manual_remote_tokenizerclean_trained_no_summary_deep_open_20260605.csv`

## Manual Open Standard

Each directory was checked by opening:

- top-level files and timestamps
- `config.json`
- latest `logs/training_*.csv`
- retained epoch weight files
- non-source summary/full_eval/metrics search
- external downstream or diagnostic evidence where claimed

`src/utils/run_evaluation.py` inside source snapshots is not counted as an eval
result.

## Directory Results

| exp_dir | rows | weights | in-dir eval | external evidence | decision |
| --- | ---: | ---: | --- | --- | --- |
| `tokenizer_t01_carrier_base_b160` | 2 | 2 / 87.408 MB | none | none | retain pending summary or owner |
| `wikiart_distinct5_ema_lancet_spectralstat_e2_b80` | 2 | 2 / 87.258 MB | none | downstream resume and full_eval | retain as downstream resume source |
| `wikiart_distinct5_ema_lancet_spectralstat_from_e8_e16_b56` | 1 | 1 / 43.629 MB | none | none | retain pending summary or owner |
| `wikiart512_ema_pair_budget_tokonly_e1_b80` | 1 | 1 / 15.044 MB | none | none | retain pending summary or owner |
| `wikiart512_ema_spectral_stat_full_e2_from_tok_b48` | 2 | 2 / 87.248 MB | none | none | retain, but flag config anomaly |
| `wikiart512_ema_tokenbudget_tokonly_e1_from_spectral_b48` | 1 | 1 / 15.110 MB | none | none | retain pending summary or owner |
| `wikiart512_ema_trueint_stylepush_tsw40_kin025_e1_b48` | 1 | 1 / 43.625 MB | none | diagnostics summary | retain as diagnostic-evaluated payload |

## Key Findings

Five directories remain true training-log-only payloads:

- `tokenizer_t01_carrier_base_b160`
- `wikiart_distinct5_ema_lancet_spectralstat_from_e8_e16_b56`
- `wikiart512_ema_pair_budget_tokonly_e1_b80`
- `wikiart512_ema_spectral_stat_full_e2_from_tok_b48`
- `wikiart512_ema_tokenbudget_tokonly_e1_from_spectral_b48`

Two directories have external evidence:

- `wikiart_distinct5_ema_lancet_spectralstat_e2_b80` is referenced by
  `wikiart_distinct5_ema_lancet_spectralstat_from_e2_e8_b64/config.json`,
  which resumes from its `epoch_0002.pt`. The downstream directory has
  full_eval `summary.json` and `metrics.csv` for epochs `0004`, `0006`, and
  `0008`.
- `wikiart512_ema_trueint_stylepush_tsw40_kin025_e1_b48` is referenced by
  `exp/diagnostics/true_integrate_stylepush_tsw40_kin025_e1_n6/summary.json`.
  That diagnostic summary reports `count=150`,
  `clip_style=0.8007408837477367`, and
  `content_lpips=0.3313907140493393`.

One directory has a config lineage anomaly:

- `wikiart512_ema_spectral_stat_full_e2_from_tok_b48/config.json` records
  `resume_checkpoint` as
  `./exp/wikiart512_ema_spectral_stat_full_e2_from_tok_b48/epoch_0004.pt`, but
  remote verification shows that file does not exist. The directory does have
  `epoch_0001.pt`, `epoch_0002.pt`, and two training rows.

## Delete Decision

All 7 remain `delete_whitelist=no`.

Reason: every row has training evidence plus retained weights, and none has an
owner-approved invalid label. Deleting these by no-summary status alone would
remove restorable experiment payloads. The safe next actions are:

- recover or run a minimal summary/full_eval for the five training-log-only
  payloads
- get owner approval before deleting any training-log-only payload
- repair or annotate the config lineage anomaly for
  `wikiart512_ema_spectral_stat_full_e2_from_tok_b48`
