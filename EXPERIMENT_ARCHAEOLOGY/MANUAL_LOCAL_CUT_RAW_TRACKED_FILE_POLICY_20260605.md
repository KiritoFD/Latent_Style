# Local CUT Raw Tracked-File Policy - 2026-06-05

Scope:

```text
G:\GitHub\Latent_Style\Related_Works\runs\cut_5x5\raw_results
G:\GitHub\Latent_Style\Related_Works\runs\cut_5x5\raw_results_val
G:\GitHub\Latent_Style\Related_Works\runs\cut_5x5\infer_5x5
G:\GitHub\Latent_Style\Related_Works\runs\cut_5x5\infer_val_clean_5x5
G:\GitHub\Latent_Style\Related_Works\runs\cut_5x5\logs\train_cut_to_*.log
```

This pass manually re-opened the remaining CUT raw web-output boundary. No file
was deleted.

## Fixed-Path Findings

| path | files | media | MB | git state | decision |
| --- | ---: | ---: | ---: | --- | --- |
| `raw_results` | 3755 | 3750 JPG | 109.181 | 5 tracked HTML + 3750 ignored images | retain_tracked_boundary |
| `raw_results_val` | 3755 | 3750 PNG | 453.017 | 3755 tracked files | retain_tracked_boundary |
| `infer_5x5` | 1260 | summary/grid/images | 47.290 | mixed tracked/ignored metric output | keep |
| `infer_val_clean_5x5` | 1260 | summary/grid/images | 46.938 | mixed tracked/ignored metric output | keep |

Both raw trees contain five exact target directories:

```text
cut_to_cezanne
cut_to_Hayao
cut_to_monet
cut_to_photo
cut_to_vangogh
```

For every target, the native CUT web index was opened:

- `raw_results\*\test_latest\index.html`: title confirms `Phase = test`,
  `Epoch = latest`, with `real_A`, `fake_B`, and `real_B` image links.
- `raw_results_val\*\val_latest\index.html`: title confirms `Phase = val`,
  `Epoch = latest`, with the same `real_A`, `fake_B`, and `real_B` image
  triads.

Representative image checks confirmed 256x256 output images. The downstream
summary grids in `infer_5x5` and `infer_val_clean_5x5` are 1528x1444.

## Timing Evidence Opened

The five training logs were opened directly. The value below is the sum of the
original `Time Taken: ... sec` epoch lines, so the unit remains seconds from the
log rather than a normalized wall-clock estimate.

| target | epoch lines | train time value | train time unit | tail evidence |
| --- | ---: | ---: | --- | --- |
| cezanne | 30 | 26158 | sec_epoch_time_taken_sum | `End of epoch 20 / 20 Time Taken: 928 sec` |
| Hayao | 20 | 14517 | sec_epoch_time_taken_sum | `End of epoch 20 / 20 Time Taken: 593 sec` |
| monet | 20 | 12680 | sec_epoch_time_taken_sum | `End of epoch 20 / 20 Time Taken: 877 sec` |
| photo | 20 | 9129 | sec_epoch_time_taken_sum | `End of epoch 20 / 20 Time Taken: 416 sec` |
| vangogh | 20 | 14120 | sec_epoch_time_taken_sum | `End of epoch 20 / 20 Time Taken: 637 sec` |

`infer_5x5\summary.json` and `infer_val_clean_5x5\summary.json` include
timestamps and metrics, but no inference wall-time field. The inference time
columns are therefore left blank in the CSV.

## Policy

All rows remain `delete_whitelist=no`.

Reason:

- `raw_results` is mixed tracked/ignored content. Deleting the directory would
  remove tracked HTML plus ignored native web images.
- `raw_results_val` is tracked repository content. Deleting it would remove
  3755 tracked files.
- `infer_5x5` and `infer_val_clean_5x5` are useful downstream metric evidence,
  but their existence does not authorize deleting tracked raw web outputs.

If the owner wants this space cleaned later, the next step is a separate
tracked-artifact migration or untracking plan with exact paths and post-delete
verification. It should not be folded into checkpoint cleanup.

Row-level CSV:

- `manual_local_cut_raw_tracked_file_policy_20260605.csv`
