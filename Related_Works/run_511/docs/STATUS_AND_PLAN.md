# run_511 Status And AAAI Plan

Updated: 2026-05-12

## Paper Scope

Main paper should stay compact:

- Table 1 quality: `AdaIN / StyTr2 / AesPA-Net / AesFA / CAST / StyleID / SaMST / Ours`
- Table 2 efficiency: `SaMST / CAST / StyleID / Ours`
- Figure 1 time-to-quality: `CycleGAN / FastCUT / SaMST / Ours`
- Table 3 ablation: six key variants
- Table 4 user study: `Ours` vs `CAST / StyleID / SaMST / StyTr2`

Do not add extra baselines to the main table. Put `AdaAttN / EFDM / ArtBank / InST / DiffuseIT / DiffStyle` in supplement or related work unless the paper story changes.

## Completed Local Runs

| Method | Run folder | Train status | Inference status | Eval status | Notes |
| --- | --- | --- | --- | --- | --- |
| `AdaIN` | `outputs/adain_7g_v32k` | `ok`, 32000 iter, 9220.4s, batch 8 | `ok`, exact protocol 750, 9.3s | `ok`, refreshed | Best current AdaIN run |
| `AdaIN` | `outputs/adain_7g_vgg19` | `ok`, 2000 iter, 262.8s, batch 4 | `ok`, exact protocol 750, 9.1s | `ok`, refreshed | Shorter run, weaker than v32k |
| `AdaIN` | `outputs/adain_4g_real` | unknown/separate checkpoint | `ok`, exact protocol 750, 5.7s | `invalid` | LPIPS is near 1.0; do not use |
| `SaMST` | `outputs/samst_750_strict` | `photo` target trained locally, 631.1s; other target ckpts migrated into run_511 | `ok`, exact protocol 750, 39.8s | `ok`, refreshed | Wrapper fixed to use target style output instead of `style0` identity |
| `StyleID` | `outputs/styleid_750_strict` | training-free | `ok`, exact protocol 750, 603.3s incremental infer | `ok`, refreshed | Strict manifest repaired by reusing 600 matched files and generating missing `photo` target |
| `StyTR-2` | `outputs/stytr2_smoke6` | `ok`, 1 iter, 59.3s | `ok`, 5 images | not needed | Smoke only |
| `StyTR-2` | `outputs/stytr2_750` | incomplete/interrupted | no 750 images | none | 48000 iter run was too slow; later 1000 iter start has no completion summary |

## Protocol-750 Screening Metrics

Strict reference manifest: `SchrodingerBridge/exp/pareto_probe_4/S-add__K-3_C-2_W-10_Col-15/full_eval/epoch_0001/images`.

Full report: `run_511/protocol750_eval_report.md`.

Metric protocol: SB-match (`CLIP-style` uses the mean target-style reference prototype; `LPIPS` uses VGG-LPIPS).
Visual diagnostics:

- `run_511/diagnostic_samst_contact.jpg`
- `run_511/diagnostic_samst_random25.jpg`
- `run_511/diagnostic_samst_stats.md`

| Run | Status | Images | Ref match | LPIPS down | CLIP-style up | CLIP-content up |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ours_k1_c0_w20_col0_epoch_0007` | `ok` | 750 | 750 | 0.4587 | 0.7041 | 0.8043 |
| `styleid_750_strict` | `ok` | 750 | 750 | 0.7497 | 0.7597 | 0.5519 |
| `samst_750_strict` | `ok` | 750 | 750 | 0.4664 | 0.7194 | 0.8193 |
| `adain_7g_v32k` | `ok` | 750 | 750 | 0.6298 | 0.7130 | 0.6990 |
| `adain_7g_vgg19` | `ok` | 750 | 750 | 0.6870 | 0.6930 | 0.5991 |
| `adain_4g_real` | `invalid` | 750 | 750 | 0.8490 | 0.6308 | 0.5297 |
| `samst_750_refmatch` | `partial` | 450 | 450 | 0.4104 | 0.7558 | 0.8355 |
| `styleid_750_refmatch` | `partial` | 600 | 600 | 0.7908 | 0.7933 | 0.5444 |

Interpretation:

- `adain_7g_v32k` is the only usable AdaIN row right now.
- `adain_4g_real` is not a valid baseline result despite matching filenames.
- `StyleID` is protocol-complete, but current settings preserve content poorly.
- `SaMST` is protocol-complete after fixing the wrapper to avoid the `style0` identity output.
- SaMST keeps strong global structure but has visible pointillist/grain artifacts. Do not let it win by CLIP/LPIPS alone.

## Anti-Hack Evaluation Guard

To avoid noisy or texture-heavy outputs gaming CLIP/LPIPS:

- Report SB-match metrics plus guard metrics: `SSIM-Y`, `Edge-F1`, and `HF ratio`.
- Add DINO/CFSD before paper tables; these are better structure-sensitive metrics than CLIP-content.
- Keep qualitative grids mandatory for every table row; SaMST-style grain is visible even when structure scores are strong.
- User study should include an explicit artifact/noise preference question, not only content/style preference.

## Immediate Fixes

1. Re-run `StyTR-2` with a realistic profile or use official pretrained weights; scratch `48000` iter is not viable on the local GPU.
2. Run `CAST` smoke/full after confirming `run_511/repos/cast` has required weights and inference script works.

## Next Execution Order

1. Validate `AdaIN` row as completed and copy `adain_7g_v32k` into the final current-protocol table.
2. Finish `CAST` smoke and then 750 inference.
3. Decide `StyTR-2`: short local training for engineering table or official pretrained weights for paper table.
