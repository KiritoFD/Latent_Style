# Remote TokenizerClean Training-Log-Only Live Recheck - 2026-06-05

Scope:

```text
I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp
```

This is a fixed-path live recheck after the checkpoint-weight cleanup. It is not
a whole-disk scan and not a substitute for owner review. The seven exact
trained no-summary payload directories were reopened over SSH one by one:

1. `tokenizer_t01_carrier_base_b160`
2. `wikiart_distinct5_ema_lancet_spectralstat_from_e8_e16_b56`
3. `wikiart512_ema_pair_budget_tokonly_e1_b80`
4. `wikiart512_ema_spectral_stat_full_e2_from_tok_b48`
5. `wikiart512_ema_tokenbudget_tokonly_e1_from_spectral_b48`
6. `wikiart_distinct5_ema_lancet_spectralstat_e2_b80`
7. `wikiart512_ema_trueint_stylepush_tsw40_kin025_e1_b48`

For each directory, the live check opened the top-level listing, `.pt` weights,
`config.json` resume/checkpoint fields, `logs\training_*.csv` first and last
rows, and the presence of `summary.json`, `metrics.csv`, and `full_eval`.

## Result

- Five training-log-only directories now have `0` remaining `.pt` weights. They
  still retain `config.json`, `logs\training_*.csv`, `numeric_debug.jsonl`, and
  source snapshots where present.
- Two evidence-bearing directories still retain weights:
  `wikiart_distinct5_ema_lancet_spectralstat_e2_b80` retains 2 weights as a
  downstream resume source, and
  `wikiart512_ema_trueint_stylepush_tsw40_kin025_e1_b48` retains 1 diagnostic
  referenced weight.
- All seven directories still lack in-directory `summary.json`, `metrics.csv`,
  and `full_eval`.
- `wikiart512_ema_spectral_stat_full_e2_from_tok_b48` still records the config
  anomaly `resume_checkpoint=./exp/wikiart512_ema_spectral_stat_full_e2_from_tok_b48/epoch_0004.pt`;
  the live directory listing has no `epoch_0004.pt`.

Row-level evidence:

- `manual_remote_tokenizerclean_training_log_only_live_recheck_20260605.csv`

This confirms the cleanup boundary: checkpoint weights were removed only from
training-log-only payloads without external evidence, while archaeology metadata
and evidence-bearing weights remain.
