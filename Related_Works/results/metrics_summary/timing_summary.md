# Timing Summary

Source: `run_511/outputs/*/summary.json`

| Run | Train | Train sec | Batch | Max iter | Epochs | Infer | Infer sec | Images | Sec / image | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `adain_4g_bg` | ok | 6.045 | 4 | 100 |  |  |  |  |  |  |
| `adain_4g_real` |  |  |  |  |  | ok | 5.736 | 750 | 0.007648 | Inference succeeded but visual output is invalid; timing still recorded. |
| `adain_7g_full` |  |  |  |  |  | ok | 0.269 | 10 | 0.026900 |  |
| `adain_7g_v32k` | ok | 9220.393 | 8 | 32000 |  | ok | 9.281 | 750 | 0.012375 | Full train + strict 750 inference timing available. |
| `adain_7g_vgg19` | ok | 262.780 | 4 | 2000 |  | ok | 9.098 | 750 | 0.012131 | Full train + strict 750 inference timing available. |
| `adain_smoke` | ok | 1.082 | 4 | 10 |  |  |  |  |  |  |
| `adain_smoke2` | ok | 1.560 | 4 | 20 |  |  |  |  |  |  |
| `adain_vgg19_smoke` | ok | 7.060 | 4 | 50 |  | ok | 0.230 | 15 | 0.015333 |  |
| `aesfa_timing_probe` | failed | 3.947 | 1 | 1 |  |  |  |  |  |  |
| `cast_smoke` | failed | 8.554 | 1 |  | 1 |  |  |  |  |  |
| `cast_smoke2` | ok | 26.654 | 1 |  | 1 | failed | 3.759 | 0 |  |  |
| `cast_smoke3` | ok | 29.366 | 1 |  | 1 | failed | 4.032 | 0 |  | Training is a 1-epoch smoke run; inference failed in this smoke config. |
| `samst_750_strict` |  |  |  |  |  | ok | 39.826 | 750 | 0.053101 | Only strict full-750 inference time is recorded here; training time was not preserved in `summary.json`. |
| `samst_timing_probe` | ok | 67.687 | 1 |  |  |  |  |  |  |  |
| `styleid_750_strict` |  |  |  |  |  | ok | 603.316 | 750 | 0.804421 | Inference timing is not a fair full-750 run: `photo` was actually generated (~603s), other targets were reused/copied. |
| `stytr2_smoke` | failed | 2.558 | 1 | 1 |  |  |  |  |  |  |
| `stytr2_smoke2` | failed | 4.412 | 1 | 1 |  |  |  |  |  |  |
| `stytr2_smoke3` | failed | 5.230 | 1 | 1 |  |  |  |  |  |  |
| `stytr2_smoke4` | failed | 5.469 | 1 | 1 |  |  |  |  |  |  |
| `stytr2_smoke5` | ok | 372.048 | 1 | 1 |  | failed | 7.055 | 0 |  |  |
| `stytr2_smoke6` | ok | 59.250 | 1 | 1 |  | ok | 35.810 | 5 | 7.162000 | Training is a 1-iter smoke run (`max_iter=1`), not a full epoch. |

## Readable Highlights

- `adain_7g_v32k`: train `9220.393s`, infer `9.281s / 750 = 0.012375s per image`.
- `adain_7g_vgg19`: train `262.780s`, infer `9.098s / 750 = 0.012131s per image`.
- `samst_750_strict`: infer `39.826s / 750 = 0.053101s per image`; train time not preserved in current summary file.
- `styleid_750_strict`: recorded infer `603.316s`, but this is not a fair full-750 timing because only `photo` was actually generated in this strict run.
- `stytr2_smoke6`: smoke train `59.250s`, smoke infer `35.810s / 5 = 7.162000s per image`.
- `cast_smoke3`: smoke train `29.366s` for `1` epoch, infer failed in this config.
