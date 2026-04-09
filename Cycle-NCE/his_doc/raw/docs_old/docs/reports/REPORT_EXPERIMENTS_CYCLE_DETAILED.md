# Experiments-Cycle Detailed Report

- Generated: 2026-02-13 15:56:59
- Source JSON: `G:/GitHub/Latent_Style/Cycle-NCE/docs/experiments_cycle/data/runs_detailed.json`

## 1) Structure Overview

- Total runs indexed: `53`
- Runs with full_eval: `52`
- Runs with parseable best style: `35`
- Strict comparable runs: `25`
- Runs with summary_history rounds: `8`
- Runs with src snapshots: `18`

### Family Breakdown

| family | runs | with_metric | strict | mean_style | best_style | best_run |
|---|---:|---:|---:|---:|---:|---|
| adacut | 5 | 2 | 0 | 0.455957 | 0.470728 | adacut_overfit50-lightonly |
| experiments_misc | 1 | 0 | 0 | - | - | - |
| full-300 | 1 | 0 | 0 | - | - | - |
| full_250 | 2 | 2 | 0 | 0.000000 | 0.000000 | full_250_strong-style |
| full_300 | 5 | 4 | 4 | 0.495998 | 0.551533 | full_300_distill_low_only_v1 |
| full_other | 1 | 1 | 1 | 0.475696 | 0.475696 | full_strong_style |
| other | 2 | 1 | 0 | 0.441478 | 0.441478 | 50-no-distill |
| overfit50 | 28 | 25 | 20 | 0.501126 | 0.593255 | overfit50-upscale |
| small-exp | 8 | 0 | 0 | - | - | - |

### Data Tier Counts

| tier | count | definition |
|---|---:|---|
| A_strict | 25 | style metric + complete matrix + LPIPS>0 |
| B_partial | 10 | style metric exists but not strict comparable |
| C_incomplete | 18 | no parseable style metric |

## 2) Run-By-Run Comparison (All Runs)

| # | run | path | family | tier | best_style | cls | lpips | eval_count | strict_rank | family_rank | record |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | overfit50-upscale | `experiments/overfit50-upscale` | overfit50 | B_partial | 0.593255 | 0.930000 | 0.000000 | 50 | - | 1/25 | [doc](experiments_cycle_records/experiments__overfit50-upscale.md) |
| 2 | full_300_distill_low_only_v1 | `full_300_distill_low_only_v1` | full_300 | A_strict | 0.551533 | 1.000000 | 0.697327 | 30 | 1 | 1/4 | [doc](experiments_cycle_records/full_300_distill_low_only_v1.md) |
| 3 | overfit50-style-force-balance-v1 | `experiments/overfit50-style-force-balance-v1` | overfit50 | A_strict | 0.549101 | 0.890000 | 0.764001 | 50 | 2 | 2/25 | [doc](experiments_cycle_records/experiments__overfit50-style-force-balance-v1.md) |
| 4 | overfit50-style-distill-struct-v1-20 | `experiments/overfit50-style-distill-struct-v1-20` | overfit50 | A_strict | 0.547901 | 0.970000 | 0.670641 | 50 | 3 | 3/25 | [doc](experiments_cycle_records/experiments__overfit50-style-distill-struct-v1-20.md) |
| 5 | overfit50-style-force-schedule | `experiments/overfit50-style-force-schedule` | overfit50 | A_strict | 0.543730 | 0.260000 | 0.582351 | 50 | 4 | 4/25 | [doc](experiments_cycle_records/experiments__overfit50-style-force-schedule.md) |
| 6 | overfit50-style-distill-struct-v2 | `experiments/overfit50-style-distill-struct-v2` | overfit50 | A_strict | 0.534868 | 0.880000 | 0.645370 | 50 | 5 | 5/25 | [doc](experiments_cycle_records/experiments__overfit50-style-distill-struct-v2.md) |
| 7 | overfit50-strong_structure | `overfit50-strong_structure` | overfit50 | A_strict | 0.534638 | 0.120000 | 0.581129 | 50 | 6 | 6/25 | [doc](experiments_cycle_records/overfit50-strong_structure.md) |
| 8 | overfit50-style-distill-struct-v2 | `overfit50-style-distill-struct-v2` | overfit50 | A_strict | 0.528368 | 0.850000 | 0.551213 | 50 | 7 | 7/25 | [doc](experiments_cycle_records/overfit50-style-distill-struct-v2.md) |
| 9 | overfit50-v5-mse-sharp-style_back | `overfit50-v5-mse-sharp-style_back` | overfit50 | A_strict | 0.526751 | 1.000000 | 0.606507 | 50 | 8 | 8/25 | [doc](experiments_cycle_records/overfit50-v5-mse-sharp-style_back.md) |
| 10 | overfit50-style-force-balance-v1-cycle4 | `experiments/overfit50-style-force-balance-v1-cycle4` | overfit50 | A_strict | 0.521914 | 0.520000 | 0.675383 | 50 | 9 | 9/25 | [doc](experiments_cycle_records/experiments__overfit50-style-force-balance-v1-cycle4.md) |
| 11 | overfit50-distill_low_only | `overfit50-distill_low_only` | overfit50 | A_strict | 0.518469 | 0.930000 | 0.545553 | 50 | 10 | 10/25 | [doc](experiments_cycle_records/overfit50-distill_low_only.md) |
| 12 | overfit50-style-distill-struct-v3 | `overfit50-style-distill-struct-v3` | overfit50 | A_strict | 0.516456 | 0.670000 | 0.510532 | 50 | 11 | 11/25 | [doc](experiments_cycle_records/overfit50-style-distill-struct-v3.md) |
| 13 | overfit50-strok-style | `overfit50-strok-style` | overfit50 | A_strict | 0.516445 | 0.930000 | 0.386163 | 50 | 12 | 12/25 | [doc](experiments_cycle_records/overfit50-strok-style.md) |
| 14 | full_300-map16+32 | `full_300-map16+32` | full_300 | A_strict | 0.509921 | 0.780000 | 0.424217 | 50 | 13 | 2/4 | [doc](experiments_cycle_records/full_300-map16-32.md) |
| 15 | overfit50-upscale-balance-v2-re-0.1cycle | `experiments/overfit50-upscale-balance-v2-re-0.1cycle` | overfit50 | A_strict | 0.506272 | 0.230000 | 0.434735 | 50 | 14 | 13/25 | [doc](experiments_cycle_records/experiments__overfit50-upscale-balance-v2-re-0.1cycle.md) |
| 16 | overfit50-upscale-balance-v2 | `experiments/overfit50-upscale-balance-v2` | overfit50 | A_strict | 0.504675 | 0.500000 | 0.441309 | 50 | 15 | 14/25 | [doc](experiments_cycle_records/experiments__overfit50-upscale-balance-v2.md) |
| 17 | overfit50-style-distill-struct-v4 | `overfit50-style-distill-struct-v4` | overfit50 | A_strict | 0.499004 | 0.550000 | 0.433740 | 50 | 16 | 15/25 | [doc](experiments_cycle_records/overfit50-style-distill-struct-v4.md) |
| 18 | overfit50-style-distill-struct-v4-mse | `overfit50-style-distill-struct-v4-mse` | overfit50 | A_strict | 0.497539 | 0.640000 | 0.437998 | 50 | 17 | 16/25 | [doc](experiments_cycle_records/overfit50-style-distill-struct-v4-mse.md) |
| 19 | overfit50-upscale-styleid-v1 | `experiments/overfit50-upscale-styleid-v1` | overfit50 | A_strict | 0.482200 | 0.450000 | 0.366808 | 50 | 18 | 17/25 | [doc](experiments_cycle_records/experiments__overfit50-upscale-styleid-v1.md) |
| 20 | overfit50-no-idt | `experiments/overfit50-no-idt` | overfit50 | B_partial | 0.478453 | 0.430000 | 0.000000 | 50 | - | 18/25 | [doc](experiments_cycle_records/experiments__overfit50-no-idt.md) |
| 21 | full_strong_style | `full_strong_style` | full_other | A_strict | 0.475696 | 0.110000 | 0.325611 | 50 | 19 | 1/1 | [doc](experiments_cycle_records/full_strong_style.md) |
| 22 | adacut_overfit50-lightonly | `adacut_overfit50-lightonly` | adacut | B_partial | 0.470728 | 0.320000 | 0.000000 | 50 | - | 1/2 | [doc](experiments_cycle_records/adacut_overfit50-lightonly.md) |
| 23 | full_300_gridfix_v2 | `full_300_gridfix_v2` | full_300 | A_strict | 0.462860 | 0.370000 | 0.324233 | 30 | 20 | 3/4 | [doc](experiments_cycle_records/full_300_gridfix_v2.md) |
| 24 | full_300_strong-style-v1 | `full_300_strong-style-v1` | full_300 | A_strict | 0.459679 | 0.170000 | 0.295542 | 50 | 21 | 4/4 | [doc](experiments_cycle_records/full_300_strong-style-v1.md) |
| 25 | overfit50-80-10-0.5 | `experiments/overfit50-80-10-0.5` | overfit50 | B_partial | 0.452397 | 0.320000 | 0.000000 | 50 | - | 19/25 | [doc](experiments_cycle_records/experiments__overfit50-80-10-0.5.md) |
| 26 | overfit50-upscale-balance-v2-re | `experiments/overfit50-upscale-balance-v2-re` | overfit50 | A_strict | 0.450307 | 0.100000 | 0.265898 | 50 | 22 | 20/25 | [doc](experiments_cycle_records/experiments__overfit50-upscale-balance-v2-re.md) |
| 27 | overfit50 | `experiments/overfit50` | overfit50 | B_partial | 0.449813 | 0.260000 | 0.000000 | 50 | - | 21/25 | [doc](experiments_cycle_records/experiments__overfit50.md) |
| 28 | overfit50-upscale-styleid-v2 | `experiments/overfit50-upscale-styleid-v2` | overfit50 | B_partial | 0.446567 | 0.080000 | 0.000000 | 50 | - | 22/25 | [doc](experiments_cycle_records/experiments__overfit50-upscale-styleid-v2.md) |
| 29 | overfit50-upscale-struct | `experiments/overfit50-upscale-struct` | overfit50 | A_strict | 0.444692 | 0.120000 | 0.254958 | 50 | 23 | 23/25 | [doc](experiments_cycle_records/experiments__overfit50-upscale-struct.md) |
| 30 | overfit50-style-force | `experiments/overfit50-style-force` | overfit50 | A_strict | 0.443286 | 0.070000 | 0.257564 | 50 | 24 | 24/25 | [doc](experiments_cycle_records/experiments__overfit50-style-force.md) |
| 31 | 50-no-distill | `50-no-distill` | other | B_partial | 0.441478 | 0.080000 | 0.000000 | 50 | - | 1/1 | [doc](experiments_cycle_records/50-no-distill.md) |
| 32 | adacut_overfit50 | `adacut_overfit50` | adacut | B_partial | 0.441186 | 0.100000 | 0.000000 | 50 | - | 2/2 | [doc](experiments_cycle_records/adacut_overfit50.md) |
| 33 | overfit50-v5-mse-sharp | `overfit50-v5-mse-sharp` | overfit50 | A_strict | 0.441041 | 0.080000 | 0.241780 | 50 | 25 | 25/25 | [doc](experiments_cycle_records/overfit50-v5-mse-sharp.md) |
| 34 | full_250_strong-style | `experiments/full_250_strong-style` | full_250 | B_partial | 0.000000 | 0.280000 | 0.000000 | 50 | - | 1/2 | [doc](experiments_cycle_records/experiments__full_250_strong-style.md) |
| 35 | full_250_strong-style | `full_250_strong-style` | full_250 | B_partial | 0.000000 | 0.460000 | 0.000000 | 50 | - | 2/2 | [doc](experiments_cycle_records/full_250_strong-style.md) |
| 36 | adacut | `adacut` | adacut | C_incomplete | - | - | - | - | - | - | [doc](experiments_cycle_records/adacut.md) |
| 37 | adacut_overfit | `adacut_overfit` | adacut | C_incomplete | - | - | - | - | - | - | [doc](experiments_cycle_records/adacut_overfit.md) |
| 38 | adacut_overfit0 | `adacut_overfit0` | adacut | C_incomplete | - | - | - | - | - | - | [doc](experiments_cycle_records/adacut_overfit0.md) |
| 39 | main-style-distill-struct-v1 | `experiments/main-style-distill-struct-v1` | experiments_misc | C_incomplete | - | - | - | - | - | - | [doc](experiments_cycle_records/experiments__main-style-distill-struct-v1.md) |
| 40 | full-300-3060-313 | `full-300-3060-313` | full-300 | C_incomplete | - | - | - | - | - | - | [doc](experiments_cycle_records/full-300-3060-313.md) |
| 41 | full_300_strong-style-v2 | `full_300_strong-style-v2` | full_300 | C_incomplete | - | - | - | - | - | - | [doc](experiments_cycle_records/full_300_strong-style-v2.md) |
| 42 | baseline_50e | `ablation50_repro_cwdfix/baseline_50e` | other | C_incomplete | - | - | - | - | - | - | [doc](experiments_cycle_records/ablation50_repro_cwdfix__baseline_50e.md) |
| 43 | overfit50-clipstyle-probe-v1 | `overfit50-clipstyle-probe-v1` | overfit50 | C_incomplete | - | - | - | - | - | - | [doc](experiments_cycle_records/overfit50-clipstyle-probe-v1.md) |
| 44 | overfit50-clipstyle-probe-v2 | `overfit50-clipstyle-probe-v2` | overfit50 | C_incomplete | - | - | - | - | - | - | [doc](experiments_cycle_records/overfit50-clipstyle-probe-v2.md) |
| 45 | overfit50-clipstyle-probe-v3 | `overfit50-clipstyle-probe-v3` | overfit50 | C_incomplete | - | - | - | - | - | - | [doc](experiments_cycle_records/overfit50-clipstyle-probe-v3.md) |
| 46 | small-exp-overfit50_e12_hires6_hifeat_v1-overfit50_e12_hires6_hifeat_v1-bd128-dsp1-hp0p22-whf3p0-wprob1p0-wproto0p2-wcyc8p0-20260209_145326 | `experiments/small-exp-overfit50_e12_hires6_hifeat_v1-overfit50_e12_hires6_hifeat_v1-bd128-dsp1-hp0p22-whf3p0-wprob1p0-wproto0p2-wcyc8p0-20260209_145326` | small-exp | C_incomplete | - | - | - | - | - | - | [doc](experiments_cycle_records/experiments__small-exp-overfit50_e12_hires6_hifeat_v1-overfit50_e12_hires6_hifeat_v1-bd128-dsp1-hp0p22-whf3p0-wprob1p0-wproto0p2-wcyc8p0-20260209_145326.md) |
| 47 | small-exp-overfit50_e12_hires6_hifeat_v1-overfit50_e12_hires6_hifeat_v1-bd128-dsp1-hp0p22-whf3p0-wprob1p0-wproto0p2-wcyc8p0-20260209_150950 | `experiments/small-exp-overfit50_e12_hires6_hifeat_v1-overfit50_e12_hires6_hifeat_v1-bd128-dsp1-hp0p22-whf3p0-wprob1p0-wproto0p2-wcyc8p0-20260209_150950` | small-exp | C_incomplete | - | - | - | - | - | - | [doc](experiments_cycle_records/experiments__small-exp-overfit50_e12_hires6_hifeat_v1-overfit50_e12_hires6_hifeat_v1-bd128-dsp1-hp0p22-whf3p0-wprob1p0-wproto0p2-wcyc8p0-20260209_150950.md) |
| 48 | small-exp-overfit50_e13_hires6_spatialproto_v1-overfit50_e13_hires6_spatialproto_v1-bd128-dsp1-hp0p2-whf3p0-wprob0p8-wproto0p2-wcyc8p0-20260209_151448 | `experiments/small-exp-overfit50_e13_hires6_spatialproto_v1-overfit50_e13_hires6_spatialproto_v1-bd128-dsp1-hp0p2-whf3p0-wprob0p8-wproto0p2-wcyc8p0-20260209_151448` | small-exp | C_incomplete | - | - | - | - | - | - | [doc](experiments_cycle_records/experiments__small-exp-overfit50_e13_hires6_spatialproto_v1-overfit50_e13_hires6_spatialproto_v1-bd128-dsp1-hp0p2-whf3p0-wprob0p8-wproto0p2-wcyc8p0-20260209_151448.md) |
| 49 | small-exp-overfit50_e14_hires6_weakcls_v1-overfit50_e14_hires6_weakcls_v1-bd128-dsp1-hp0p18-whf3p4-wprob0p35-wproto0p2-wcyc8p0-20260209_151947 | `experiments/small-exp-overfit50_e14_hires6_weakcls_v1-overfit50_e14_hires6_weakcls_v1-bd128-dsp1-hp0p18-whf3p4-wprob0p35-wproto0p2-wcyc8p0-20260209_151947` | small-exp | C_incomplete | - | - | - | - | - | - | [doc](experiments_cycle_records/experiments__small-exp-overfit50_e14_hires6_weakcls_v1-overfit50_e14_hires6_weakcls_v1-bd128-dsp1-hp0p18-whf3p4-wprob0p35-wproto0p2-wcyc8p0-20260209_151947.md) |
| 50 | small-exp-overfit50_e15_style_only_from_smoke | `experiments/small-exp-overfit50_e15_style_only_from_smoke` | small-exp | C_incomplete | - | - | - | - | - | - | [doc](experiments_cycle_records/experiments__small-exp-overfit50_e15_style_only_from_smoke.md) |
| 51 | small-exp-overfit50_e16_style_only_flowboost | `experiments/small-exp-overfit50_e16_style_only_flowboost` | small-exp | C_incomplete | - | - | - | - | - | - | [doc](experiments_cycle_records/experiments__small-exp-overfit50_e16_style_only_flowboost.md) |
| 52 | small-exp-overfit50_e17_style_forcepath | `experiments/small-exp-overfit50_e17_style_forcepath` | small-exp | C_incomplete | - | - | - | - | - | - | [doc](experiments_cycle_records/experiments__small-exp-overfit50_e17_style_forcepath.md) |
| 53 | small-exp-smoke_e12_skipfusion_v2-overfit50_e12_hires6_hifeat_v1-bd128-dsp1-hp0p22-whf3p0-wprob1p0-wproto0p2-wcyc8p0-20260209_150800 | `experiments/small-exp-smoke_e12_skipfusion_v2-overfit50_e12_hires6_hifeat_v1-bd128-dsp1-hp0p22-whf3p0-wprob1p0-wproto0p2-wcyc8p0-20260209_150800` | small-exp | C_incomplete | - | - | - | - | - | - | [doc](experiments_cycle_records/experiments__small-exp-smoke_e12_skipfusion_v2-overfit50_e12_hires6_hifeat_v1-bd128-dsp1-hp0p22-whf3p0-wprob1p0-wproto0p2-wcyc8p0-20260209_150800.md) |

## 3) Cross-Run Findings

- Current strict best: `full_300_distill_low_only_v1` at `0.551533`.
- Strict runs with style>=0.54: `4/25`.
- Strict runs with LPIPS>=0.60: `6/25`.
- Correlation(style, LPIPS) on strict runs: `0.939530`.
- Correlation(style, classifier_acc) on strict runs: `0.731278`.

### Balanced Candidates (style / cls / lpips)

| run | path | best_style | cls | lpips |
|---|---|---:|---:|---:|
| overfit50-style-distill-struct-v2 | `overfit50-style-distill-struct-v2` | 0.528368 | 0.850000 | 0.551213 |
| overfit50-distill_low_only | `overfit50-distill_low_only` | 0.518469 | 0.930000 | 0.545553 |
| overfit50-strok-style | `overfit50-strok-style` | 0.516445 | 0.930000 | 0.386163 |
| full_300-map16+32 | `full_300-map16+32` | 0.509921 | 0.780000 | 0.424217 |

## 4) Final Summary

- The historical pool is large, but strict-comparable evidence is much smaller than total runs.
- High style scores often co-occur with higher LPIPS; style/content balance is still the main bottleneck.
- Per-run record docs are generated for all runs in `docs/reports/experiments_cycle_records/` for direct audit.
