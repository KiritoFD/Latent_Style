# CSV Statistical Appendix

Reviewed on `2026-05-16`.

This appendix records the main grouped statistics extracted from the existing CSV evidence.

## 1. Destructive Ablation: Final Eval Joined with Final Training Log

Final joined rows:

| run | clip_style | clip_content | LPIPS | terminal_swd | kinetic_energy | semantic_k_abs | plan_entropy | velocity_abs | endpoint_abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D2_no_kinetic | 0.7159 | 0.6624 | 0.6375 | 0.000000 | 0.000000 | 0.632902 | 0.626274 | 0.055395 | 16.388847 |
| D4_conv_body_no_global_attn | 0.7022 | 0.8020 | 0.4594 | 0.000000 | 0.259521 | 0.391358 | 0.619435 | 0.055539 | 16.402470 |
| D6_disable_spatial_prior | 0.7022 | 0.8033 | 0.4589 | 0.000000 | 0.259125 | 0.391081 | 0.619546 | 0.055566 | 16.440404 |
| D9_l2_ot_cost | 0.7016 | 0.8021 | 0.4589 | 0.000000 | 0.259589 | 0.391475 | 0.619369 | 0.052418 | 16.407830 |
| D0_full_correct_7ep | 0.7014 | 0.8022 | 0.4593 | 0.000000 | 0.259121 | 0.391096 | 0.619378 | 0.049207 | 16.376984 |
| D7_no_residual_path | 0.7013 | 0.8025 | 0.4592 | 0.000000 | 0.259613 | 0.391462 | 0.619395 | 0.054443 | 16.398286 |
| D11_single_terminal_step | 0.7012 | 0.8032 | 0.4585 | 0.000000 | 0.259292 | 0.391204 | 0.619313 | 0.050277 | 16.377209 |
| D10_micro_hf_swd_trap | 0.6989 | 0.7772 | 0.4863 | 0.000000 | 0.163779 | 0.287971 | 0.626606 | 0.050692 | 16.417772 |
| D5_disable_skip_routing | 0.6951 | 0.8057 | 0.4727 | 0.000000 | 0.259741 | 0.390465 | 0.615051 | 0.056776 | 15.962208 |
| D8_strong_color_loss | 0.6923 | 0.6629 | 0.5675 | 3.685236 | 0.389935 | 0.483859 | 0.601152 | 0.057587 | 16.463803 |
| D3_no_swd_no_kinetic | 0.6884 | 0.8527 | 0.3938 | 0.000000 | 0.000000 | 0.175236 | 0.680413 | 0.053502 | 17.245457 |
| D1_no_terminal_swd | 0.6708 | 0.8989 | 0.3490 | 0.000000 | 0.000011 | 0.002468 | 0.636142 | 0.052058 | 16.407844 |

Correlations inside destructive ablations:

- `corr(clip_style, semantic_k_abs) = +0.8845`
- `corr(clip_style, lpips) = +0.7202`
- `corr(clip_style, kinetic_energy) = +0.2483`
- `corr(clip_style, plan_entropy) = -0.3378`
- `corr(clip_style, clip_content) = -0.6177`

Correlations to LPIPS inside destructive ablations:

- `corr(lpips, clip_content) = -0.9747`
- `corr(lpips, semantic_k_abs) = +0.8979`
- `corr(lpips, clip_style) = +0.7202`
- `corr(lpips, plan_entropy) = -0.4733`

## 2. `experiments_root`: Best Style by Experiment

| experiment_id | best_epoch | style_best | content_at_best | LPIPS_at_best |
|---|---|---:|---:|---:|
| 06_anchor_skip_only | epoch_0020 | 0.736300 | 0.594711 | 0.852838 |
| 07_anchor_hybrid_all | epoch_0060 | 0.718600 | 0.643252 | 0.687591 |
| 02_omf_swd_30 | epoch_0160 | 0.704228 | 0.707709 | 0.594692 |
| 03_omf_swd_45 | epoch_0140 | 0.702877 | 0.701714 | 0.593510 |
| 01_omf_swd_15 | epoch_0160 | 0.693792 | 0.734436 | 0.559026 |
| 05_anchor_ot_mse_only | epoch_0020 | 0.690874 | 0.610321 | 0.728009 |
| 04_anchor_kin_only | epoch_0140 | 0.690376 | 0.773014 | 0.517405 |

Correlations inside `experiments_root`:

- `corr(clip_style, content_lpips) = +0.6335`
- `corr(clip_style, clip_content) = -0.5121`

## 3. `weight_sweep_40`: Best-per-Experiment by K Family

Best-per-experiment aggregate:

| K family | mean best style | max best style | min best style | mean best content | mean best LPIPS |
|---|---:|---:|---:|---:|---:|
| K1 | 0.710957 | 0.716126 | 0.708478 | 0.799625 | 0.462262 |
| K2 | 0.706426 | 0.708210 | 0.704192 | 0.836888 | 0.420191 |

Correlations across all rows in `weight_sweep_40_all_epochs.csv`:

- `corr(all_clip_style, all_content_lpips) = +0.7760`
- `corr(all_clip_style, k_value) = -0.5130`
- `corr(all_clip_style, all_clip_content) = -0.7780`

Top K1 runs:

| experiment_id | epoch | style | content | LPIPS |
|---|---|---:|---:|---:|
| K1_r00_balanced_default | epoch_0008 | 0.716126 | 0.798365 | 0.460504 |
| K1_r04_cezanne_strong | epoch_0008 | 0.712916 | 0.789810 | 0.472741 |
| K1_r01_uniform_unbalanced | epoch_0007 | 0.711949 | 0.801998 | 0.462487 |
| K1_r09_photo_content_low | epoch_0008 | 0.711893 | 0.786327 | 0.477389 |
| K1_r12_hayao_cezanne | epoch_0008 | 0.711844 | 0.801215 | 0.459683 |

Top K2 runs:

| experiment_id | epoch | style | content | LPIPS |
|---|---|---:|---:|---:|
| K2_r04_cezanne_strong | epoch_0008 | 0.708210 | 0.830547 | 0.427283 |
| K2_r16_photo_hayao_content_art_target | epoch_0007 | 0.707257 | 0.838977 | 0.422141 |
| K2_r08_photo_content_high | epoch_0007 | 0.707043 | 0.839632 | 0.422678 |
| K2_r00_balanced_default | epoch_0008 | 0.706900 | 0.839888 | 0.413140 |
| K2_r11_photo_target_some | epoch_0008 | 0.706652 | 0.834804 | 0.419050 |

## 4. Step Size Sweep

| run | style | content |
|---|---:|---:|
| step_1p5 | 0.716197 | 0.808753 |
| step_2p0 | 0.716126 | 0.808645 |
| step_1p25 | 0.716120 | 0.808726 |
| base_epoch7 | 0.716114 | 0.808575 |

The sweep is effectively flat.

## 5. Step Count Sweep

| run | style | content | LPIPS |
|---|---:|---:|---:|
| steps_01 | 0.715977 | 0.808622 | 0.451390 |
| steps_04 | 0.716029 | 0.808607 | 0.451416 |
| steps_08 | 0.715928 | 0.808500 | 0.451408 |
| steps_12 | 0.716167 | 0.808688 | 0.451406 |
| steps_16 | 0.716105 | 0.808645 | 0.451392 |

This sweep is also effectively flat.

## 6. Residual Scale Sweep

| run | style | content |
|---|---:|---:|
| residual_1p25 | 0.721854 | 0.763490 |
| residual_1p5 | 0.720807 | 0.721171 |
| base_epoch7 | 0.716114 | 0.808575 |
| residual_2p0 | 0.706930 | 0.655791 |

This sweep is not flat. It shows a clear overshoot curve.

## 7. Theory Switch Validation: Delta vs `T0_k2_baseline`

Reference row:

- `T0_k2_baseline`: `epoch_0003`, `style 0.703216`, `content 0.859817`, `LPIPS 0.397394`

Delta table:

| run | best_epoch | d_style | d_content | d_LPIPS |
|---|---|---:|---:|---:|
| T1_sinkhorn_routing | epoch_0003 | -0.003864 | +0.007326 | -0.010369 |
| T2_entropy_gate_2p5 | epoch_0003 | -0.001561 | +0.005538 | -0.007025 |
| T3_entropy_gate_5p0 | epoch_0003 | -0.001588 | +0.006331 | -0.009715 |
| T4_sinkhorn_entropy | epoch_0003 | -0.003917 | +0.011286 | -0.012170 |
| T5_color_soft_w2 | epoch_0003 | +0.002219 | -0.034347 | +0.033941 |
| T6_color_gumbel_w2 | epoch_0003 | +0.002093 | -0.026158 | +0.026706 |
| T7_all_switches_mild | epoch_0003 | -0.001436 | -0.001946 | +0.000884 |

## 8. High-Tension Sweep: Best by Experiment

| experiment_id | best_epoch | style | content | LPIPS |
|---|---|---:|---:|---:|
| g2_swd_nuke | epoch_0030 | 0.655134 | 0.846339 | 0.378047 |
| g1_high_tension_base | epoch_0080 | 0.639189 | 0.879374 | 0.334654 |

Correlations inside this sweep:

- `corr(clip_style, content_lpips) = +0.9900`
- `corr(clip_style, clip_content) = -0.9923`

## 9. Orthogonal Phase Sweep: Best by Experiment

Top best-per-experiment rows:

| experiment_id | best_epoch | style | content | LPIPS |
|---|---|---:|---:|---:|
| g3_gravity_black_hole | epoch_0040 | 0.667729 | 0.803358 | 0.436491 |
| g1_absolute_release | epoch_0060 | 0.662420 | 0.806725 | 0.421744 |
| g6_structure_amnesty | epoch_0020 | 0.657839 | 0.852262 | 0.374793 |
| g7_flesh_stripping | epoch_0040 | 0.656204 | 0.838446 | 0.391182 |
| g0_universe_center | epoch_0030 | 0.650187 | 0.862086 | 0.363568 |

Correlations inside this sweep:

- `corr(clip_style, content_lpips) = +0.9777`
- `corr(clip_style, clip_content) = -0.9545`
