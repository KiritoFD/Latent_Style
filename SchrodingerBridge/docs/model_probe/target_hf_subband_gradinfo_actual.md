# Gradient and Information-Flow Probe

Config: `configs\exp_probe_target_hf_subband_ft6.json`
Checkpoint: `exp\model_probe\target_hf_subband_ft6\epoch_0006.pt`
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
| lh | 7.646076e-02 | 8.678105e-05 | 1.134975e-03 |
| hl | 7.166816e-02 | 9.953389e-05 | 1.388816e-03 |
| hh | 1.318640e-01 | 1.700700e-04 | 1.289738e-03 |

### loss_fm_hf_total

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.646076e-02 | 8.678105e-05 | 1.134975e-03 |
| hl | 7.166816e-02 | 9.953389e-05 | 1.388816e-03 |
| hh | 1.318640e-01 | 1.700700e-04 | 1.289738e-03 |

### loss_stat

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.646076e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.166816e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.318640e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_lh

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.646076e-02 | 8.678105e-05 | 1.134975e-03 |
| hl | 7.166816e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.318640e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_stat_lh

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.646076e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.166816e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.318640e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_hl

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.646076e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.166816e-02 | 9.953389e-05 | 1.388816e-03 |
| hh | 1.318640e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_stat_hl

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.646076e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.166816e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.318640e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_hh

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.646076e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.166816e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.318640e-01 | 1.700700e-04 | 1.289738e-03 |

### loss_stat_hh

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.646076e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.166816e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.318640e-01 | 0.000000e+00 | 0.000000e+00 |

## Style-Latent Band Information Flow

### full target style_latent vs content condition

| output band | delta/base | delta rms |
|---|---:|---:|
| ll | 0.000000e+00 | 0.000000e+00 |
| lh | 7.562427e-02 | 1.178426e-02 |
| hl | 9.722850e-02 | 1.684589e-02 |
| hh | 1.194318e-01 | 1.332076e-02 |

### full target condition direction alignment

| output band | delta/desired | cos(delta, desired) | projection | orthogonal fraction | MSE improvement |
|---|---:|---:|---:|---:|---:|
| lh | 1.656277e-02 | 5.378619e-02 | 9.247426e-04 | 9.983401e-01 | 1.630882e-03 |
| hl | 2.064347e-02 | 4.478419e-02 | 9.692420e-04 | 9.987984e-01 | 1.611934e-03 |
| hh | 1.911406e-02 | 3.159146e-02 | 6.351958e-04 | 9.993771e-01 | 8.682688e-04 |

### single target condition band

| input band | output band | delta/base |
|---|---|---:|
| ll | ll | 0.000000e+00 |
| ll | lh | 1.505623e-06 |
| ll | hl | 3.091421e-06 |
| ll | hh | 6.496664e-06 |
| lh | ll | 0.000000e+00 |
| lh | lh | 7.562435e-02 |
| lh | hl | 3.189200e-06 |
| lh | hh | 3.560569e-06 |
| hl | ll | 0.000000e+00 |
| hl | lh | 1.668525e-06 |
| hl | hl | 9.722853e-02 |
| hl | hh | 7.284633e-06 |
| hh | ll | 0.000000e+00 |
| hh | lh | 1.771336e-06 |
| hh | hl | 2.205374e-06 |
| hh | hh | 1.194318e-01 |

### single target condition band direction alignment

| input band | output band | delta/desired | cos(delta, desired) | projection | MSE improvement |
|---|---|---:|---:|---:|---:|
| ll | lh | 3.297525e-07 | -1.233979e-03 | 1.735832e-09 | 0.000000e+00 |
| ll | hl | 6.563677e-07 | -2.875139e-03 | -4.261147e-09 | 8.950713e-08 |
| ll | hh | 1.039736e-06 | 5.975739e-03 | -4.481679e-09 | 0.000000e+00 |
| lh | lh | 1.656279e-02 | 5.378646e-02 | 9.247478e-04 | 1.630882e-03 |
| lh | hl | 6.771281e-07 | -1.177254e-03 | 4.698180e-10 | 0.000000e+00 |
| lh | hh | 5.698392e-07 | 3.460600e-03 | 1.773999e-09 | 0.000000e+00 |
| hl | lh | 3.654303e-07 | 1.701762e-03 | 1.563926e-09 | 0.000000e+00 |
| hl | hl | 2.064347e-02 | 4.478396e-02 | 9.692370e-04 | 1.611934e-03 |
| hl | hh | 1.165844e-06 | -3.063414e-03 | -2.959971e-09 | 6.136175e-08 |
| hh | lh | 3.879474e-07 | 7.670434e-03 | 3.781762e-09 | 0.000000e+00 |
| hh | hl | 4.682431e-07 | 3.538665e-03 | 7.977594e-10 | 0.000000e+00 |
| hh | hh | 1.911406e-02 | 3.159128e-02 | 6.351930e-04 | 8.682688e-04 |

### route interventions

| intervention | output band | delta/base |
|---|---|---:|
| target_hf_residual_contribution | ll | 0.000000e+00 |
| target_hf_residual_contribution | lh | 4.739516e-01 |
| target_hf_residual_contribution | hl | 3.962925e-01 |
| target_hf_residual_contribution | hh | 1.081199e+00 |
| cfg_unconditional_delta_from_full | ll | 0.000000e+00 |
| cfg_unconditional_delta_from_full | lh | 4.921412e-01 |
| cfg_unconditional_delta_from_full | hl | 4.129803e-01 |
| cfg_unconditional_delta_from_full | hh | 1.164728e+00 |

## Input Band Gradient Split

### full_shared

#### loss

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.751048e-05 | 2.316063e-02 |
| content | lh | 1.646031e-04 | 1.640284e-01 |
| content | hl | 1.670093e-04 | 2.114096e-01 |
| content | hh | 3.348601e-04 | 6.013805e-01 |
| target_style_shared | ll | 1.881765e-06 | 1.942317e-04 |
| target_style_shared | lh | 1.771431e-04 | 1.636213e-01 |
| target_style_shared | hl | 1.736597e-04 | 2.146312e-01 |
| target_style_shared | hh | 3.457298e-04 | 6.215328e-01 |

#### loss_fm_hf_total

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.744820e-05 | 2.309202e-02 |
| content | lh | 1.643476e-04 | 1.637758e-01 |
| content | hl | 1.668665e-04 | 2.113790e-01 |
| content | hh | 3.346960e-04 | 6.017322e-01 |
| target_style_shared | ll | 5.459801e-07 | 1.633858e-05 |
| target_style_shared | lh | 1.772445e-04 | 1.636848e-01 |
| target_style_shared | hl | 1.737126e-04 | 2.145995e-01 |
| target_style_shared | hh | 3.459012e-04 | 6.216789e-01 |

#### loss_fm_spectral_lh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.194368e-05 | 2.655058e-02 |
| content | lh | 1.617922e-04 | 9.638321e-01 |
| content | hl | 1.000355e-05 | 4.613124e-03 |
| content | hh | 1.222762e-05 | 4.876962e-03 |
| target_style_shared | ll | 1.354055e-07 | 6.250180e-06 |
| target_style_shared | lh | 1.748171e-04 | 9.903526e-01 |
| target_style_shared | hl | 1.022524e-05 | 4.624581e-03 |
| target_style_shared | hh | 1.229993e-05 | 4.889074e-03 |

#### loss_fm_spectral_hl

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.567538e-05 | 3.468399e-02 |
| content | lh | 1.569804e-05 | 6.881342e-03 |
| content | hl | 1.649718e-04 | 9.514851e-01 |
| content | hh | 1.664417e-05 | 6.853078e-03 |
| target_style_shared | ll | 2.445845e-07 | 1.522833e-05 |
| target_style_shared | lh | 1.675392e-05 | 6.792496e-03 |
| target_style_shared | hl | 1.728072e-04 | 9.863325e-01 |
| target_style_shared | hh | 1.674261e-05 | 6.764597e-03 |

#### loss_fm_spectral_hh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.571182e-05 | 1.236851e-02 |
| content | lh | 2.042885e-05 | 4.136576e-03 |
| content | hl | 1.721852e-05 | 3.679133e-03 |
| content | hh | 3.340399e-04 | 9.797815e-01 |
| target_style_shared | ll | 2.427986e-07 | 5.179290e-06 |
| target_style_shared | lh | 2.180293e-05 | 3.970175e-03 |
| target_style_shared | hl | 1.760011e-05 | 3.531133e-03 |
| target_style_shared | hh | 3.451970e-04 | 9.924606e-01 |

### target_only

#### loss

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.751048e-05 | 2.316063e-02 |
| content | lh | 1.646031e-04 | 1.640284e-01 |
| content | hl | 1.670093e-04 | 2.114096e-01 |
| content | hh | 3.348601e-04 | 6.013805e-01 |
| target_style_target_path | ll | 1.881765e-06 | 1.940706e-04 |
| target_style_target_path | lh | 1.771318e-04 | 1.634648e-01 |
| target_style_target_path | hl | 1.736799e-04 | 2.145031e-01 |
| target_style_target_path | hh | 3.459524e-04 | 6.218175e-01 |

#### loss_fm_hf_total

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.744820e-05 | 2.309202e-02 |
| content | lh | 1.643476e-04 | 1.637758e-01 |
| content | hl | 1.668665e-04 | 2.113790e-01 |
| content | hh | 3.346960e-04 | 6.017322e-01 |
| target_style_target_path | ll | 5.459801e-07 | 1.632455e-05 |
| target_style_target_path | lh | 1.772375e-04 | 1.635314e-01 |
| target_style_target_path | hl | 1.737442e-04 | 2.144933e-01 |
| target_style_target_path | hh | 3.461221e-04 | 6.219386e-01 |

#### loss_fm_spectral_lh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.194368e-05 | 2.655058e-02 |
| content | lh | 1.617922e-04 | 9.638321e-01 |
| content | hl | 1.000355e-05 | 4.613123e-03 |
| content | hh | 1.222762e-05 | 4.876963e-03 |
| target_style_target_path | ll | 1.354055e-07 | 6.245527e-06 |
| target_style_target_path | lh | 1.748829e-04 | 9.903598e-01 |
| target_style_target_path | hl | 1.022524e-05 | 4.621137e-03 |
| target_style_target_path | hh | 1.229993e-05 | 4.885434e-03 |

#### loss_fm_spectral_hl

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.567538e-05 | 3.468398e-02 |
| content | lh | 1.569804e-05 | 6.881343e-03 |
| content | hl | 1.649718e-04 | 9.514851e-01 |
| content | hh | 1.664416e-05 | 6.853072e-03 |
| target_style_target_path | ll | 2.445844e-07 | 1.522061e-05 |
| target_style_target_path | lh | 1.675392e-05 | 6.789060e-03 |
| target_style_target_path | hl | 1.728515e-04 | 9.863394e-01 |
| target_style_target_path | hh | 1.674260e-05 | 6.761168e-03 |

#### loss_fm_spectral_hh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.571182e-05 | 1.236851e-02 |
| content | lh | 2.042885e-05 | 4.136576e-03 |
| content | hl | 1.721852e-05 | 3.679133e-03 |
| content | hh | 3.340399e-04 | 9.797815e-01 |
| target_style_target_path | ll | 2.427987e-07 | 5.172893e-06 |
| target_style_target_path | lh | 2.180293e-05 | 3.965269e-03 |
| target_style_target_path | hl | 1.760011e-05 | 3.526770e-03 |
| target_style_target_path | hh | 3.454121e-04 | 9.924700e-01 |

### condition_only

#### loss

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.751048e-05 | 2.316063e-02 |
| content | lh | 1.646031e-04 | 1.640284e-01 |
| content | hl | 1.670093e-04 | 2.114096e-01 |
| content | hh | 3.348601e-04 | 6.013805e-01 |
| target_style_condition_path | ll | 2.207242e-14 | 1.504311e-16 |
| target_style_condition_path | lh | 4.476792e-06 | 5.882659e-01 |
| target_style_condition_path | hl | 2.291315e-06 | 2.103356e-01 |
| target_style_condition_path | hh | 1.713052e-06 | 8.589742e-02 |

#### loss_fm_hf_total

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.744820e-05 | 2.309202e-02 |
| content | lh | 1.643476e-04 | 1.637758e-01 |
| content | hl | 1.668665e-04 | 2.113790e-01 |
| content | hh | 3.346960e-04 | 6.017322e-01 |
| target_style_condition_path | ll | 2.195325e-14 | 1.488112e-16 |
| target_style_condition_path | lh | 4.476792e-06 | 5.882659e-01 |
| target_style_condition_path | hl | 2.291315e-06 | 2.103356e-01 |
| target_style_condition_path | hh | 1.713052e-06 | 8.589742e-02 |

#### loss_fm_spectral_lh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.194368e-05 | 2.655058e-02 |
| content | lh | 1.617922e-04 | 9.638321e-01 |
| content | hl | 1.000355e-05 | 4.613124e-03 |
| content | hh | 1.222762e-05 | 4.876963e-03 |
| target_style_condition_path | ll | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | lh | 4.476792e-06 | 8.358816e-01 |
| target_style_condition_path | hl | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hh | 0.000000e+00 | 0.000000e+00 |

#### loss_fm_spectral_hl

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.567538e-05 | 3.468399e-02 |
| content | lh | 1.569804e-05 | 6.881340e-03 |
| content | hl | 1.649718e-04 | 9.514851e-01 |
| content | hh | 1.664417e-05 | 6.853078e-03 |
| target_style_condition_path | ll | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | lh | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hl | 2.291315e-06 | 6.455246e-01 |
| target_style_condition_path | hh | 0.000000e+00 | 0.000000e+00 |

#### loss_fm_spectral_hh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.571182e-05 | 1.236850e-02 |
| content | lh | 2.042885e-05 | 4.136575e-03 |
| content | hl | 1.721852e-05 | 3.679132e-03 |
| content | hh | 3.340399e-04 | 9.797815e-01 |
| target_style_condition_path | ll | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | lh | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hl | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hh | 1.713052e-06 | 4.265047e-01 |
