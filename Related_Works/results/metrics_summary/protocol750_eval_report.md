# Protocol-750 Evaluation Report

Reference manifest: `SchrodingerBridge/exp/pareto_probe_4/S-add__K-3_C-2_W-10_Col-15/full_eval/epoch_0001/images`

Metric protocol: SB-match (`CLIP-style = cos(CLIP(gen), mean target-style reference prototype)`, `LPIPS = VGG-LPIPS`).

| Method | Run | Status | Images | Ref match | Missing | Extra | LPIPS down | CLIP-style up | CLIP-content up | SSIM-Y up | Edge-F1 up | HF ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours epoch_0007 | `ours_k1_c0_w20_col0_epoch_0007` | ok | 750 | 750 | 0 | 0 | 0.4587 | 0.7041 | 0.8043 | 0.4545 | 0.311 | 0.857 |
| StyleID strict | `styleid_750_strict` | ok | 750 | 750 | 0 | 0 | 0.7497 | 0.7597 | 0.5519 | 0.1466 | 0.1954 | 1.3291 |
| SaMST strict | `samst_750_strict` | ok | 750 | 750 | 0 | 0 | 0.4664 | 0.7194 | 0.8193 | 0.652 | 0.5162 | 0.5864 |
| AdaIN v32k | `adain_7g_v32k` | ok | 750 | 750 | 0 | 0 | 0.6298 | 0.713 | 0.699 | 0.3246 | 0.0167 | 0.1935 |
| AdaIN vgg19 | `adain_7g_vgg19` | ok | 750 | 750 | 0 | 0 | 0.687 | 0.693 | 0.5991 |  |  |  |
| AdaIN bad | `adain_4g_real` | invalid | 750 | 750 | 0 | 0 | 0.849 | 0.6308 | 0.5297 |  |  |  |
| SaMST refmatch | `samst_750_refmatch` | partial | 450 | 450 | 300 | 0 | 0.4104 | 0.7558 | 0.8355 |  |  |  |
| StyleID refmatch | `styleid_750_refmatch` | partial | 600 | 600 | 150 | 0 | 0.7908 | 0.7933 | 0.5444 |  |  |  |

## Notes

- `ok` rows exactly match the 750-image manifest and are table-ready for current screening metrics.
- `partial` rows are evaluated only on files that overlap the manifest, so they are useful for diagnosis but not main-table ready.
- `invalid` rows have exact 750 files but clearly broken/too weak behavior and should not be used.
- Guard metrics are no-download sanity checks. They do not replace DINO/CFSD/user study, but help catch CLIP/LPIPS blind spots.
- SaMST has strong structure metrics, but visual inspection shows heavy pointillist/grain artifacts; keep qualitative grids and user study in the decision loop.
