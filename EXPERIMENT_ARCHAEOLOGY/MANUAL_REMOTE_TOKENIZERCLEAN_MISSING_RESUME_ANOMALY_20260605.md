# Remote TokenizerClean Missing Resume Anomaly - 2026-06-05

Scope:

```text
I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\wikiart512_ema_spectral_stat_full_e2_from_tok_b48
```

This fixed-path check documents a config lineage anomaly. It does not modify the
remote config and does not restore or delete any files.

## Live Finding

`config.json` exists and records:

```text
resume_checkpoint="./exp/wikiart512_ema_spectral_stat_full_e2_from_tok_b48/epoch_0004.pt"
batch_size=48
num_epochs=2
```

Remote live file checks:

| target | exists |
| --- | --- |
| `epoch_0001.pt` | false |
| `epoch_0002.pt` | false |
| `epoch_0004.pt` | false |

The directory still retains one training CSV:

```text
logs\training_20260531_035513.csv
```

The last row records epoch `2`, `epoch_time_sec=264.62290501594543`, and
`samples_per_sec=68.02132263991044`.

In-directory evaluation evidence is absent:

| artifact | exists |
| --- | --- |
| `summary.json` | false |
| `metrics.csv` | false |
| `full_eval` | false |

## Conclusion

This directory is retained only as metadata/training-log archaeology. It must
not be promoted as a clean resume lineage or evaluated result. The old
`epoch_0004.pt` resume target should be treated as a stale/self-referential
config field unless an external owner later provides a separate checkpoint
artifact.

Row-level record:

- `manual_remote_tokenizerclean_missing_resume_anomaly_20260605.csv`
