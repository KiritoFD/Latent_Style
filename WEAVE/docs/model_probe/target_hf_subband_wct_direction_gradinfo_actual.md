# Gradient and Information-Flow Probe

Config: `configs\exp_probe_target_hf_subband_wct_direction_ft6.json`
Checkpoint: `exp\model_probe\target_hf_subband_wct_direction_ft6\epoch_0006.pt`
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
| lh | 7.621025e-02 | 8.675455e-05 | 1.138358e-03 |
| hl | 7.147571e-02 | 9.944200e-05 | 1.391270e-03 |
| hh | 1.319290e-01 | 1.699872e-04 | 1.288475e-03 |

### loss_fm_hf_total

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.621025e-02 | 8.675455e-05 | 1.138358e-03 |
| hl | 7.147571e-02 | 9.944200e-05 | 1.391270e-03 |
| hh | 1.319290e-01 | 1.699872e-04 | 1.288475e-03 |

### loss_stat

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.621025e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.147571e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.319290e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_lh

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.621025e-02 | 8.675455e-05 | 1.138358e-03 |
| hl | 7.147571e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.319290e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_stat_lh

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.621025e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.147571e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.319290e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_hl

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.621025e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.147571e-02 | 9.944200e-05 | 1.391270e-03 |
| hh | 1.319290e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_stat_hl

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.621025e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.147571e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.319290e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_hh

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.621025e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.147571e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.319290e-01 | 1.699872e-04 | 1.288475e-03 |

### loss_stat_hh

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.621025e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.147571e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.319290e-01 | 0.000000e+00 | 0.000000e+00 |

## Style-Latent Band Information Flow

### full target style_latent vs content condition

| output band | delta/base | delta rms |
|---|---:|---:|
| ll | 0.000000e+00 | 0.000000e+00 |
| lh | 8.350426e-02 | 1.290452e-02 |
| hl | 1.086773e-01 | 1.866091e-02 |
| hh | 1.309602e-01 | 1.443744e-02 |

### full target condition direction alignment

| output band | delta/desired | cos(delta, desired) | projection | orthogonal fraction | MSE improvement |
|---|---:|---:|---:|---:|---:|
| lh | 1.812609e-02 | 1.098872e-01 | 1.902797e-03 | 9.937931e-01 | 3.474350e-03 |
| hl | 2.284543e-02 | 1.251716e-01 | 2.825249e-03 | 9.916717e-01 | 5.390902e-03 |
| hh | 2.070283e-02 | 9.349220e-02 | 1.928466e-03 | 9.953671e-01 | 3.146067e-03 |

### single target condition band

| input band | output band | delta/base |
|---|---|---:|
| ll | ll | 0.000000e+00 |
| ll | lh | 3.780843e-06 |
| ll | hl | 3.364397e-06 |
| ll | hh | 1.192143e-05 |
| lh | ll | 0.000000e+00 |
| lh | lh | 8.350423e-02 |
| lh | hl | 4.240374e-06 |
| lh | hh | 3.511969e-06 |
| hl | ll | 0.000000e+00 |
| hl | lh | 3.524133e-06 |
| hl | hl | 1.086773e-01 |
| hl | hh | 6.373137e-06 |
| hh | ll | 0.000000e+00 |
| hh | lh | 5.650353e-06 |
| hh | hl | 3.718089e-06 |
| hh | hh | 1.309602e-01 |

### single target condition band direction alignment

| input band | output band | delta/desired | cos(delta, desired) | projection | MSE improvement |
|---|---|---:|---:|---:|---:|
| ll | lh | 8.206994e-07 | -8.197147e-03 | -9.853872e-09 | 0.000000e+00 |
| ll | hl | 7.072416e-07 | -1.564145e-03 | -3.639686e-09 | 0.000000e+00 |
| ll | hh | 1.884598e-06 | -3.189225e-03 | 2.688927e-08 | 0.000000e+00 |
| lh | lh | 1.812608e-02 | 1.098869e-01 | 1.902792e-03 | 3.474350e-03 |
| lh | hl | 8.913836e-07 | -1.624533e-03 | -3.243334e-09 | 0.000000e+00 |
| lh | hh | 5.551895e-07 | -6.601716e-03 | -7.407421e-10 | -6.128146e-08 |
| hl | lh | 7.649759e-07 | -7.672736e-03 | -7.377218e-09 | 0.000000e+00 |
| hl | hl | 2.284543e-02 | 1.251712e-01 | 2.825248e-03 | 5.390902e-03 |
| hl | hh | 1.007497e-06 | -2.301524e-02 | -1.628437e-08 | -1.225629e-07 |
| hh | lh | 1.226510e-06 | -7.960429e-03 | -7.742415e-09 | 0.000000e+00 |
| hh | hl | 7.815924e-07 | 1.354372e-02 | 2.019416e-09 | -8.933321e-08 |
| hh | hh | 2.070284e-02 | 9.349163e-02 | 1.928467e-03 | 3.146067e-03 |

### route interventions

| intervention | output band | delta/base |
|---|---|---:|
| target_hf_residual_contribution | ll | 0.000000e+00 |
| target_hf_residual_contribution | lh | 4.724192e-01 |
| target_hf_residual_contribution | hl | 3.944144e-01 |
| target_hf_residual_contribution | hh | 1.078633e+00 |
| cfg_unconditional_delta_from_full | ll | 0.000000e+00 |
| cfg_unconditional_delta_from_full | lh | 4.933189e-01 |
| cfg_unconditional_delta_from_full | hl | 4.130554e-01 |
| cfg_unconditional_delta_from_full | hh | 1.162327e+00 |

## Input Band Gradient Split

### full_shared

#### loss

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.741666e-05 | 2.304131e-02 |
| content | lh | 1.644750e-04 | 1.640465e-01 |
| content | hl | 1.667677e-04 | 2.111503e-01 |
| content | hh | 3.346814e-04 | 6.017410e-01 |
| target_style_shared | ll | 1.887683e-06 | 1.956595e-04 |
| target_style_shared | lh | 1.772406e-04 | 1.639725e-01 |
| target_style_shared | hl | 1.734449e-04 | 2.143242e-01 |
| target_style_shared | hh | 3.455366e-04 | 6.214871e-01 |

#### loss_fm_hf_total

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.734974e-05 | 2.296403e-02 |
| content | lh | 1.642280e-04 | 1.638041e-01 |
| content | hl | 1.666279e-04 | 2.111186e-01 |
| content | hh | 3.345234e-04 | 6.020922e-01 |
| target_style_shared | ll | 5.364612e-07 | 1.579066e-05 |
| target_style_shared | lh | 1.773389e-04 | 1.640340e-01 |
| target_style_shared | hl | 1.734963e-04 | 2.142939e-01 |
| target_style_shared | hh | 3.457049e-04 | 6.216358e-01 |

#### loss_fm_spectral_lh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.189522e-05 | 2.637223e-02 |
| content | lh | 1.616975e-04 | 9.640454e-01 |
| content | hl | 9.972832e-06 | 4.591226e-03 |
| content | hh | 1.220255e-05 | 4.863754e-03 |
| target_style_shared | ll | 1.335619e-07 | 6.072034e-06 |
| target_style_shared | lh | 1.749538e-04 | 9.904155e-01 |
| target_style_shared | hl | 1.019384e-05 | 4.589341e-03 |
| target_style_shared | hh | 1.227472e-05 | 4.861756e-03 |

#### loss_fm_spectral_hl

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.560177e-05 | 3.446743e-02 |
| content | lh | 1.561228e-05 | 6.827827e-03 |
| content | hl | 1.647374e-04 | 9.517763e-01 |
| content | hh | 1.659200e-05 | 6.831666e-03 |
| target_style_shared | ll | 2.406946e-07 | 1.478404e-05 |
| target_style_shared | lh | 1.666239e-05 | 6.734987e-03 |
| target_style_shared | hl | 1.726025e-04 | 9.864160e-01 |
| target_style_shared | hh | 1.669013e-05 | 6.738774e-03 |

#### loss_fm_spectral_hh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.567693e-05 | 1.232761e-02 |
| content | lh | 2.035946e-05 | 4.113187e-03 |
| content | hl | 1.716955e-05 | 3.662388e-03 |
| content | hh | 3.338643e-04 | 9.798625e-01 |
| target_style_shared | ll | 2.388015e-07 | 5.015568e-06 |
| target_style_shared | lh | 2.172887e-05 | 3.947502e-03 |
| target_style_shared | hl | 1.755005e-05 | 3.514862e-03 |
| target_style_shared | hh | 3.450179e-04 | 9.924997e-01 |

### target_only

#### loss

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.741666e-05 | 2.304131e-02 |
| content | lh | 1.644750e-04 | 1.640464e-01 |
| content | hl | 1.667677e-04 | 2.111502e-01 |
| content | hh | 3.346814e-04 | 6.017410e-01 |
| target_style_target_path | ll | 1.887683e-06 | 1.954016e-04 |
| target_style_target_path | lh | 1.771016e-04 | 1.634995e-01 |
| target_style_target_path | hl | 1.736103e-04 | 2.144499e-01 |
| target_style_target_path | hh | 3.458613e-04 | 6.218347e-01 |

#### loss_fm_hf_total

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.734975e-05 | 2.296405e-02 |
| content | lh | 1.642280e-04 | 1.638041e-01 |
| content | hl | 1.666279e-04 | 2.111186e-01 |
| content | hh | 3.345234e-04 | 6.020922e-01 |
| target_style_target_path | ll | 5.364605e-07 | 1.576897e-05 |
| target_style_target_path | lh | 1.772087e-04 | 1.635685e-01 |
| target_style_target_path | hl | 1.736748e-04 | 2.144404e-01 |
| target_style_target_path | hh | 3.460309e-04 | 6.219548e-01 |

#### loss_fm_spectral_lh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.189522e-05 | 2.637223e-02 |
| content | lh | 1.616974e-04 | 9.640454e-01 |
| content | hl | 9.972832e-06 | 4.591227e-03 |
| content | hh | 1.220255e-05 | 4.863754e-03 |
| target_style_target_path | ll | 1.335621e-07 | 6.076349e-06 |
| target_style_target_path | lh | 1.748912e-04 | 9.904087e-01 |
| target_style_target_path | hl | 1.019384e-05 | 4.592592e-03 |
| target_style_target_path | hh | 1.227472e-05 | 4.865199e-03 |

#### loss_fm_spectral_hl

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.560177e-05 | 3.446743e-02 |
| content | lh | 1.561228e-05 | 6.827831e-03 |
| content | hl | 1.647374e-04 | 9.517763e-01 |
| content | hh | 1.659199e-05 | 6.831661e-03 |
| target_style_target_path | ll | 2.406943e-07 | 1.475124e-05 |
| target_style_target_path | lh | 1.666239e-05 | 6.720065e-03 |
| target_style_target_path | hl | 1.727968e-04 | 9.864461e-01 |
| target_style_target_path | hh | 1.669012e-05 | 6.723834e-03 |

#### loss_fm_spectral_hh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.567693e-05 | 1.232762e-02 |
| content | lh | 2.035947e-05 | 4.113189e-03 |
| content | hl | 1.716955e-05 | 3.662387e-03 |
| content | hh | 3.338643e-04 | 9.798625e-01 |
| target_style_target_path | ll | 2.388015e-07 | 5.006484e-06 |
| target_style_target_path | lh | 2.172888e-05 | 3.940356e-03 |
| target_style_target_path | hl | 1.755005e-05 | 3.508497e-03 |
| target_style_target_path | hh | 3.453330e-04 | 9.925133e-01 |

### condition_only

#### loss

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.741666e-05 | 2.304131e-02 |
| content | lh | 1.644750e-04 | 1.640465e-01 |
| content | hl | 1.667677e-04 | 2.111503e-01 |
| content | hh | 3.346814e-04 | 6.017410e-01 |
| target_style_condition_path | ll | 4.035352e-14 | 3.826608e-16 |
| target_style_condition_path | lh | 4.611764e-06 | 4.751025e-01 |
| target_style_condition_path | hl | 2.841623e-06 | 2.462008e-01 |
| target_style_condition_path | hh | 2.926555e-06 | 1.907945e-01 |

#### loss_fm_hf_total

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.734975e-05 | 2.296405e-02 |
| content | lh | 1.642280e-04 | 1.638041e-01 |
| content | hl | 1.666279e-04 | 2.111186e-01 |
| content | hh | 3.345234e-04 | 6.020923e-01 |
| target_style_condition_path | ll | 4.045279e-14 | 3.845460e-16 |
| target_style_condition_path | lh | 4.611764e-06 | 4.751025e-01 |
| target_style_condition_path | hl | 2.841623e-06 | 2.462008e-01 |
| target_style_condition_path | hh | 2.926555e-06 | 1.907945e-01 |

#### loss_fm_spectral_lh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.189522e-05 | 2.637223e-02 |
| content | lh | 1.616975e-04 | 9.640454e-01 |
| content | hl | 9.972834e-06 | 4.591228e-03 |
| content | hh | 1.220255e-05 | 4.863754e-03 |
| target_style_condition_path | ll | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | lh | 4.611764e-06 | 8.438696e-01 |
| target_style_condition_path | hl | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hh | 0.000000e+00 | 0.000000e+00 |

#### loss_fm_spectral_hl

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.560177e-05 | 3.446743e-02 |
| content | lh | 1.561228e-05 | 6.827826e-03 |
| content | hl | 1.647374e-04 | 9.517763e-01 |
| content | hh | 1.659200e-05 | 6.831668e-03 |
| target_style_condition_path | ll | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | lh | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hl | 2.841623e-06 | 7.369011e-01 |
| target_style_condition_path | hh | 0.000000e+00 | 0.000000e+00 |

#### loss_fm_spectral_hh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.567693e-05 | 1.232761e-02 |
| content | lh | 2.035947e-05 | 4.113189e-03 |
| content | hl | 1.716955e-05 | 3.662388e-03 |
| content | hh | 3.338643e-04 | 9.798625e-01 |
| target_style_condition_path | ll | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | lh | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hl | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hh | 2.926555e-06 | 6.845957e-01 |
