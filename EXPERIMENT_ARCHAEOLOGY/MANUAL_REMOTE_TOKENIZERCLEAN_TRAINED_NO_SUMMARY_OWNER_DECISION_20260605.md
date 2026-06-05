# Remote TokenizerClean Trained No-Summary Owner Decision - 2026-06-05

This pass manually reopened the 7 retained trained no-summary payload
directories under:

```text
I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp
```

No deletion was performed. The output table is:

- `manual_remote_tokenizerclean_trained_no_summary_owner_decision_20260605.csv`

## Why This Pass Exists

Earlier passes removed pure orphan/probe no-summary weights, but these 7
directories are different. Each has training payload evidence:

- `config.json`
- `logs/training_*.csv`
- retained epoch checkpoint weights
- no in-dir `summary.json` or full_eval summary

Deleting them now would remove the only checkpoint payload before either owner
approval or summary recovery.

## Remote Open Result

| exp_dir | training rows | weights | summary_like | owner decision |
| --- | ---: | ---: | ---: | --- |
| `tokenizer_t01_carrier_base_b160` | 2 | 2 | 0 | retain pending summary or owner |
| `wikiart_distinct5_ema_lancet_spectralstat_e2_b80` | 2 | 2 | 0 | retain as downstream resume source |
| `wikiart_distinct5_ema_lancet_spectralstat_from_e8_e16_b56` | 1 | 1 | 0 | retain pending summary or owner |
| `wikiart512_ema_pair_budget_tokonly_e1_b80` | 1 | 1 | 0 | retain pending summary or owner |
| `wikiart512_ema_spectral_stat_full_e2_from_tok_b48` | 2 | 2 | 0 | retain pending summary or owner |
| `wikiart512_ema_tokenbudget_tokonly_e1_from_spectral_b48` | 1 | 1 | 0 | retain pending summary or owner |
| `wikiart512_ema_trueint_stylepush_tsw40_kin025_e1_b48` | 1 | 1 | 0 | retain as diagnostic-evaluated payload |

## External Evidence

Two dirs have indirect evidence outside their own directory:

- `wikiart_distinct5_ema_lancet_spectralstat_e2_b80`: downstream config
  `exp/wikiart_distinct5_ema_lancet_spectralstat_from_e2_e8_b64/config.json`
  resumes from `epoch_0002.pt`; the downstream directory has later full_eval
  summaries.
- `wikiart512_ema_trueint_stylepush_tsw40_kin025_e1_b48`: diagnostic summary
  `exp/diagnostics/true_integrate_stylepush_tsw40_kin025_e1_n6/summary.json`
  references its checkpoint and reports `count=150`,
  `clip_style=0.8007408837477367`, and
  `content_lpips=0.3313907140493393`.

The other five dirs remain training-log-only payloads.

## Decision

All 7 are `delete_whitelist=no`.

Allowed next actions:

- run or recover a minimal summary/full_eval for each payload
- ask owner for explicit deletion approval on training-log-only payloads
- keep downstream-resume and diagnostic-evaluated payloads unless their
  external evidence is superseded

Blocked action:

- do not delete these weights by no-summary status alone
- do not treat them as pure orphan probes
- do not include them in size-based cleanup without a per-dir owner decision
