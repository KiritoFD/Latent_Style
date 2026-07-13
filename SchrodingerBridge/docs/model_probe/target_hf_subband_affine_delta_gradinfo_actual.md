# Gradient and Information-Flow Probe

Config: `configs\exp_probe_target_hf_subband_affine_delta_ft6.json`
Checkpoint: `exp\model_probe\target_hf_subband_affine_delta_ft6\epoch_0006.pt`
Device: `cuda`
Batch size: 4
Load info: `{'missing': 0, 'unexpected': 0}`

## Objective Focus

- `structure_aligned_target`: `True`
- `ll_partial_style_enabled`: `True`
- `ll_partial_alpha`: `0.3`
- `spectral_w_ll`: `0.3`
- `spectral_w_lh`: `1.0`
- `spectral_w_hl`: `1.0`
- `spectral_w_hh`: `2.0`
- `train_hf_stat_loss_enabled_in_config`: `False`
- `probe_hf_stat_loss_enabled`: `False`
- `probe_hf_stat_weight`: `2.0`
- `target_latent_hf_subband_fusion_enabled`: `True`
- `style_cross_attention_enabled`: `True`
- `cfg_dropout_prob`: `0.0`

## Group Gradient Cosines

## Residual Output Activation Gradients

### loss

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.083434e-02 | 8.675641e-05 | 1.224779e-03 |
| hl | 7.369781e-02 | 9.959699e-05 | 1.351424e-03 |
| hh | 1.591285e-01 | 1.697144e-04 | 1.066524e-03 |

### loss_fm_hf_total

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.083434e-02 | 8.675641e-05 | 1.224779e-03 |
| hl | 7.369781e-02 | 9.959699e-05 | 1.351424e-03 |
| hh | 1.591285e-01 | 1.697144e-04 | 1.066524e-03 |

### loss_stat

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.083434e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.369781e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.591285e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_lh

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.083434e-02 | 8.675641e-05 | 1.224779e-03 |
| hl | 7.369781e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.591285e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_stat_lh

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.083434e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.369781e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.591285e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_hl

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.083434e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.369781e-02 | 9.959699e-05 | 1.351424e-03 |
| hh | 1.591285e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_stat_hl

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.083434e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.369781e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.591285e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_hh

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.083434e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.369781e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.591285e-01 | 1.697144e-04 | 1.066524e-03 |

### loss_stat_hh

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.083434e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.369781e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.591285e-01 | 0.000000e+00 | 0.000000e+00 |

## Style-Latent Band Information Flow

### full target style_latent vs content condition

| output band | delta/base | delta rms |
|---|---:|---:|
| ll | 0.000000e+00 | 0.000000e+00 |
| lh | 1.747478e-01 | 2.718563e-02 |
| hl | 2.363286e-01 | 4.108424e-02 |
| hh | 1.430707e-01 | 1.585276e-02 |

### full target condition direction alignment

| output band | delta/desired | cos(delta, desired) | projection | orthogonal fraction | MSE improvement |
|---|---:|---:|---:|---:|---:|
| lh | 3.817609e-02 | 6.906373e-02 | 2.622321e-03 | 9.974470e-01 | 3.936167e-03 |
| hl | 5.023180e-02 | 6.521869e-02 | 3.481902e-03 | 9.974071e-01 | 4.871202e-03 |
| hh | 2.278694e-02 | 4.347223e-02 | 1.057836e-03 | 9.988303e-01 | 1.566557e-03 |

### single target condition band

| input band | output band | delta/base |
|---|---|---:|
| ll | ll | 0.000000e+00 |
| ll | lh | 4.524249e-06 |
| ll | hl | 6.236170e-06 |
| ll | hh | 1.224463e-05 |
| lh | ll | 0.000000e+00 |
| lh | lh | 1.747477e-01 |
| lh | hl | 3.877075e-06 |
| lh | hh | 1.606688e-05 |
| hl | ll | 0.000000e+00 |
| hl | lh | 4.279852e-06 |
| hl | hl | 2.363286e-01 |
| hl | hh | 1.566639e-05 |
| hh | ll | 0.000000e+00 |
| hh | lh | 3.644043e-06 |
| hh | hl | 5.075094e-06 |
| hh | hh | 1.430707e-01 |

### single target condition band direction alignment

| input band | output band | delta/desired | cos(delta, desired) | projection | MSE improvement |
|---|---|---:|---:|---:|---:|
| ll | lh | 9.883850e-07 | -3.128331e-03 | -2.812062e-09 | 0.000000e+00 |
| ll | hl | 1.325502e-06 | 6.858318e-03 | 1.316380e-08 | 0.000000e+00 |
| ll | hh | 1.950208e-06 | 2.434464e-03 | 6.789628e-09 | 0.000000e+00 |
| lh | lh | 3.817606e-02 | 6.906353e-02 | 2.622321e-03 | 3.936167e-03 |
| lh | hl | 8.240746e-07 | 8.023703e-03 | 9.115020e-09 | 0.000000e+00 |
| lh | hh | 2.558978e-06 | 2.145948e-03 | 1.048325e-08 | 0.000000e+00 |
| hl | lh | 9.349930e-07 | -9.432148e-03 | -5.223902e-09 | 0.000000e+00 |
| hl | hl | 5.023180e-02 | 6.521877e-02 | 3.481903e-03 | 4.871202e-03 |
| hl | hh | 2.495193e-06 | -3.678813e-03 | -2.677251e-09 | 0.000000e+00 |
| hh | lh | 7.960919e-07 | -1.160205e-02 | -1.175106e-08 | 0.000000e+00 |
| hh | hl | 1.078714e-06 | 2.233358e-02 | 2.126974e-08 | 8.910193e-08 |
| hh | hh | 2.278694e-02 | 4.347262e-02 | 1.057844e-03 | 1.566619e-03 |

### route interventions

| intervention | output band | delta/base |
|---|---|---:|
| target_hf_residual_contribution | ll | 0.000000e+00 |
| target_hf_residual_contribution | lh | 4.529285e-01 |
| target_hf_residual_contribution | hl | 4.215733e-01 |
| target_hf_residual_contribution | hh | 1.081515e+00 |
| cfg_unconditional_delta_from_full | ll | 0.000000e+00 |
| cfg_unconditional_delta_from_full | lh | 4.571394e-01 |
| cfg_unconditional_delta_from_full | hl | 4.199216e-01 |
| cfg_unconditional_delta_from_full | hh | 1.423083e+00 |

## Input Band Gradient Split

### full_shared

#### loss

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.840696e-05 | 2.473291e-02 |
| content | lh | 1.645994e-04 | 1.642749e-01 |
| content | hl | 1.670485e-04 | 2.118364e-01 |
| content | hh | 3.339759e-04 | 5.991348e-01 |
| target_style_shared | ll | 1.879459e-06 | 1.938919e-04 |
| target_style_shared | lh | 1.771454e-04 | 1.637403e-01 |
| target_style_shared | hl | 1.744240e-04 | 2.166764e-01 |
| target_style_shared | hh | 3.450064e-04 | 6.193689e-01 |

#### loss_fm_hf_total

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.840795e-05 | 2.476807e-02 |
| content | lh | 1.643120e-04 | 1.639229e-01 |
| content | hl | 1.668890e-04 | 2.117179e-01 |
| content | hh | 3.338717e-04 | 5.995702e-01 |
| target_style_shared | ll | 5.907575e-07 | 1.914011e-05 |
| target_style_shared | lh | 1.772871e-04 | 1.638635e-01 |
| target_style_shared | hl | 1.744746e-04 | 2.166186e-01 |
| target_style_shared | hh | 3.451830e-04 | 6.194782e-01 |

#### loss_fm_spectral_lh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.205634e-05 | 2.708420e-02 |
| content | lh | 1.616110e-04 | 9.627549e-01 |
| content | hl | 1.019342e-05 | 4.795285e-03 |
| content | hh | 1.266537e-05 | 5.238284e-03 |
| target_style_shared | ll | 1.529224e-07 | 7.987521e-06 |
| target_style_shared | lh | 1.745973e-04 | 9.897974e-01 |
| target_style_shared | hl | 1.041932e-05 | 4.811206e-03 |
| target_style_shared | hh | 1.274028e-05 | 5.255676e-03 |

#### loss_fm_spectral_hl

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.602969e-05 | 3.625387e-02 |
| content | lh | 1.575988e-05 | 6.932638e-03 |
| content | hl | 1.648440e-04 | 9.495974e-01 |
| content | hh | 1.696855e-05 | 7.119699e-03 |
| target_style_shared | ll | 2.516108e-07 | 1.603713e-05 |
| target_style_shared | lh | 1.681991e-05 | 6.812669e-03 |
| target_style_shared | hl | 1.732086e-04 | 9.860800e-01 |
| target_style_shared | hh | 1.706890e-05 | 6.996493e-03 |

#### loss_fm_spectral_hh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.629293e-05 | 1.333830e-02 |
| content | lh | 2.139630e-05 | 4.550597e-03 |
| content | hl | 1.868491e-05 | 4.344839e-03 |
| content | hh | 3.332151e-04 | 9.777319e-01 |
| target_style_shared | ll | 2.604308e-07 | 5.986094e-06 |
| target_style_shared | lh | 2.283545e-05 | 4.375020e-03 |
| target_style_shared | hl | 1.909900e-05 | 4.177201e-03 |
| target_style_shared | hh | 3.442281e-04 | 9.914088e-01 |

### target_only

#### loss

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.840696e-05 | 2.473292e-02 |
| content | lh | 1.645994e-04 | 1.642749e-01 |
| content | hl | 1.670485e-04 | 2.118364e-01 |
| content | hh | 3.339759e-04 | 5.991348e-01 |
| target_style_target_path | ll | 1.879459e-06 | 1.936378e-04 |
| target_style_target_path | lh | 1.772814e-04 | 1.637769e-01 |
| target_style_target_path | hl | 1.743714e-04 | 2.162620e-01 |
| target_style_target_path | hh | 3.453380e-04 | 6.197470e-01 |

#### loss_fm_hf_total

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.840794e-05 | 2.476805e-02 |
| content | lh | 1.643120e-04 | 1.639229e-01 |
| content | hl | 1.668891e-04 | 2.117179e-01 |
| content | hh | 3.338717e-04 | 5.995701e-01 |
| target_style_target_path | ll | 5.907563e-07 | 1.911282e-05 |
| target_style_target_path | lh | 1.774497e-04 | 1.639309e-01 |
| target_style_target_path | hl | 1.744390e-04 | 2.162222e-01 |
| target_style_target_path | hh | 3.455204e-04 | 6.198073e-01 |

#### loss_fm_spectral_lh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.205633e-05 | 2.708416e-02 |
| content | lh | 1.616110e-04 | 9.627549e-01 |
| content | hl | 1.019343e-05 | 4.795291e-03 |
| content | hh | 1.266537e-05 | 5.238285e-03 |
| target_style_target_path | ll | 1.529219e-07 | 7.971041e-06 |
| target_style_target_path | lh | 1.747788e-04 | 9.898183e-01 |
| target_style_target_path | hl | 1.041933e-05 | 4.801321e-03 |
| target_style_target_path | hh | 1.274028e-05 | 5.244872e-03 |

#### loss_fm_spectral_hl

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.602969e-05 | 3.625386e-02 |
| content | lh | 1.575988e-05 | 6.932638e-03 |
| content | hl | 1.648440e-04 | 9.495974e-01 |
| content | hh | 1.696855e-05 | 7.119699e-03 |
| target_style_target_path | ll | 2.516111e-07 | 1.603635e-05 |
| target_style_target_path | lh | 1.681991e-05 | 6.812324e-03 |
| target_style_target_path | hl | 1.732131e-04 | 9.860807e-01 |
| target_style_target_path | hh | 1.706890e-05 | 6.996139e-03 |

#### loss_fm_spectral_hh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.629294e-05 | 1.333831e-02 |
| content | lh | 2.139629e-05 | 4.550593e-03 |
| content | hl | 1.868492e-05 | 4.344840e-03 |
| content | hh | 3.332152e-04 | 9.777319e-01 |
| target_style_target_path | ll | 2.604305e-07 | 5.975157e-06 |
| target_style_target_path | lh | 2.283545e-05 | 4.367035e-03 |
| target_style_target_path | hl | 1.909900e-05 | 4.169581e-03 |
| target_style_target_path | hh | 3.445453e-04 | 9.914245e-01 |

### condition_only

#### loss

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.840694e-05 | 2.473289e-02 |
| content | lh | 1.645994e-04 | 1.642749e-01 |
| content | hl | 1.670485e-04 | 2.118365e-01 |
| content | hh | 3.339759e-04 | 5.991348e-01 |
| target_style_condition_path | ll | 3.976215e-14 | 1.509659e-16 |
| target_style_condition_path | lh | 7.241264e-06 | 4.759594e-01 |
| target_style_condition_path | hl | 5.272387e-06 | 3.443968e-01 |
| target_style_condition_path | hh | 3.987485e-06 | 1.439258e-01 |

#### loss_fm_hf_total

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.840795e-05 | 2.476807e-02 |
| content | lh | 1.643120e-04 | 1.639229e-01 |
| content | hl | 1.668891e-04 | 2.117179e-01 |
| content | hh | 3.338717e-04 | 5.995702e-01 |
| target_style_condition_path | ll | 4.012827e-14 | 1.537588e-16 |
| target_style_condition_path | lh | 7.241264e-06 | 4.759594e-01 |
| target_style_condition_path | hl | 5.272387e-06 | 3.443968e-01 |
| target_style_condition_path | hh | 3.987486e-06 | 1.439258e-01 |

#### loss_fm_spectral_lh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.205635e-05 | 2.708422e-02 |
| content | lh | 1.616110e-04 | 9.627549e-01 |
| content | hl | 1.019342e-05 | 4.795283e-03 |
| content | hh | 1.266537e-05 | 5.238284e-03 |
| target_style_condition_path | ll | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | lh | 7.241264e-06 | 9.301943e-01 |
| target_style_condition_path | hl | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hh | 0.000000e+00 | 0.000000e+00 |

#### loss_fm_spectral_hl

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.602969e-05 | 3.625383e-02 |
| content | lh | 1.575987e-05 | 6.932634e-03 |
| content | hl | 1.648440e-04 | 9.495974e-01 |
| content | hh | 1.696855e-05 | 7.119700e-03 |
| target_style_condition_path | ll | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | lh | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hl | 5.272387e-06 | 9.060336e-01 |
| target_style_condition_path | hh | 0.000000e+00 | 0.000000e+00 |

#### loss_fm_spectral_hh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.629293e-05 | 1.333830e-02 |
| content | lh | 2.139630e-05 | 4.550595e-03 |
| content | hl | 1.868491e-05 | 4.344836e-03 |
| content | hh | 3.332151e-04 | 9.777319e-01 |
| target_style_condition_path | ll | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | lh | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hl | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hh | 3.987485e-06 | 8.011731e-01 |
