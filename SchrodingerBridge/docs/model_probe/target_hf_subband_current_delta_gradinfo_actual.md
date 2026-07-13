# Gradient and Information-Flow Probe

Config: `configs\exp_probe_target_hf_subband_current_delta_ft6.json`
Checkpoint: `exp\model_probe\target_hf_subband_current_delta_ft6\epoch_0006.pt`
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
| lh | 7.678349e-02 | 8.677880e-05 | 1.130175e-03 |
| hl | 7.201640e-02 | 9.953068e-05 | 1.382056e-03 |
| hh | 1.321801e-01 | 1.700690e-04 | 1.286646e-03 |

### loss_fm_hf_total

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.678349e-02 | 8.677880e-05 | 1.130175e-03 |
| hl | 7.201640e-02 | 9.953068e-05 | 1.382056e-03 |
| hh | 1.321801e-01 | 1.700690e-04 | 1.286646e-03 |

### loss_stat

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.678349e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.201640e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.321801e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_lh

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.678349e-02 | 8.677880e-05 | 1.130175e-03 |
| hl | 7.201640e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.321801e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_stat_lh

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.678349e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.201640e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.321801e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_hl

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.678349e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.201640e-02 | 9.953068e-05 | 1.382056e-03 |
| hh | 1.321801e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_stat_hl

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.678349e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.201640e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.321801e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_hh

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.678349e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.201640e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.321801e-01 | 1.700690e-04 | 1.286646e-03 |

### loss_stat_hh

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.678349e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.201640e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.321801e-01 | 0.000000e+00 | 0.000000e+00 |

## Style-Latent Band Information Flow

### full target style_latent vs content condition

| output band | delta/base | delta rms |
|---|---:|---:|
| ll | 0.000000e+00 | 0.000000e+00 |
| lh | 8.337503e-02 | 1.299922e-02 |
| hl | 1.054963e-01 | 1.828900e-02 |
| hh | 1.259722e-01 | 1.405205e-02 |

### single target condition band

| input band | output band | delta/base |
|---|---|---:|
| ll | ll | 0.000000e+00 |
| ll | lh | 3.402306e-06 |
| ll | hl | 3.756481e-06 |
| ll | hh | 7.640249e-06 |
| lh | ll | 0.000000e+00 |
| lh | lh | 8.337499e-02 |
| lh | hl | 1.840143e-06 |
| lh | hh | 1.110032e-05 |
| hl | ll | 0.000000e+00 |
| hl | lh | 2.906558e-06 |
| hl | hl | 1.054963e-01 |
| hl | hh | 8.125108e-06 |
| hh | ll | 0.000000e+00 |
| hh | lh | 6.788474e-06 |
| hh | hl | 2.698150e-06 |
| hh | hh | 1.259722e-01 |

### route interventions

| intervention | output band | delta/base |
|---|---|---:|
| target_hf_residual_contribution | ll | 0.000000e+00 |
| target_hf_residual_contribution | lh | 4.758825e-01 |
| target_hf_residual_contribution | hl | 3.980898e-01 |
| target_hf_residual_contribution | hh | 1.080357e+00 |
| cfg_unconditional_delta_from_full | ll | 0.000000e+00 |
| cfg_unconditional_delta_from_full | lh | 4.940306e-01 |
| cfg_unconditional_delta_from_full | hl | 4.146722e-01 |
| cfg_unconditional_delta_from_full | hh | 1.165533e+00 |

## Input Band Gradient Split

### full_shared

#### loss

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.751372e-05 | 2.318025e-02 |
| content | lh | 1.645525e-04 | 1.640277e-01 |
| content | hl | 1.669598e-04 | 2.114134e-01 |
| content | hh | 3.347516e-04 | 6.013577e-01 |
| target_style_shared | ll | 1.882879e-06 | 1.943503e-04 |
| target_style_shared | lh | 1.771993e-04 | 1.636313e-01 |
| target_style_shared | hl | 1.736999e-04 | 2.146077e-01 |
| target_style_shared | hh | 3.458324e-04 | 6.215461e-01 |

#### loss_fm_hf_total

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.745108e-05 | 2.311078e-02 |
| content | lh | 1.642992e-04 | 1.637779e-01 |
| content | hl | 1.668174e-04 | 2.113818e-01 |
| content | hh | 3.345886e-04 | 6.017085e-01 |
| target_style_shared | ll | 5.471268e-07 | 1.639806e-05 |
| target_style_shared | lh | 1.773000e-04 | 1.636952e-01 |
| target_style_shared | hl | 1.737512e-04 | 2.145742e-01 |
| target_style_shared | hh | 3.460026e-04 | 6.216937e-01 |

#### loss_fm_spectral_lh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.194077e-05 | 2.655404e-02 |
| content | lh | 1.617415e-04 | 9.638245e-01 |
| content | hl | 1.000188e-05 | 4.614439e-03 |
| content | hh | 1.222726e-05 | 4.879697e-03 |
| target_style_shared | ll | 1.353983e-07 | 6.246246e-06 |
| target_style_shared | lh | 1.748634e-04 | 9.903594e-01 |
| target_style_shared | hl | 1.022354e-05 | 4.620623e-03 |
| target_style_shared | hh | 1.229958e-05 | 4.886236e-03 |

#### loss_fm_spectral_hl

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.567244e-05 | 3.469152e-02 |
| content | lh | 1.569609e-05 | 6.883704e-03 |
| content | hl | 1.649221e-04 | 9.514755e-01 |
| content | hh | 1.663884e-05 | 6.852749e-03 |
| target_style_shared | ll | 2.446941e-07 | 1.523576e-05 |
| target_style_shared | lh | 1.675184e-05 | 6.788034e-03 |
| target_style_shared | hl | 1.728434e-04 | 9.863440e-01 |
| target_style_shared | hh | 1.673725e-05 | 6.757509e-03 |

#### loss_fm_spectral_hh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.571818e-05 | 1.238611e-02 |
| content | lh | 2.044383e-05 | 4.145184e-03 |
| content | hl | 1.723094e-05 | 3.686702e-03 |
| content | hh | 3.339318e-04 | 9.797477e-01 |
| target_style_shared | ll | 2.435618e-07 | 5.208772e-06 |
| target_style_shared | lh | 2.181892e-05 | 3.973609e-03 |
| target_style_shared | hl | 1.761281e-05 | 3.534104e-03 |
| target_style_shared | hh | 3.452997e-04 | 9.924542e-01 |

### target_only

#### loss

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.751372e-05 | 2.318025e-02 |
| content | lh | 1.645525e-04 | 1.640277e-01 |
| content | hl | 1.669598e-04 | 2.114134e-01 |
| content | hh | 3.347515e-04 | 6.013576e-01 |
| target_style_target_path | ll | 1.882879e-06 | 1.941815e-04 |
| target_style_target_path | lh | 1.771800e-04 | 1.634536e-01 |
| target_style_target_path | hl | 1.737214e-04 | 2.144742e-01 |
| target_style_target_path | hh | 3.460695e-04 | 6.218575e-01 |

#### loss_fm_hf_total

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.745109e-05 | 2.311079e-02 |
| content | lh | 1.642992e-04 | 1.637779e-01 |
| content | hl | 1.668174e-04 | 2.113818e-01 |
| content | hh | 3.345886e-04 | 6.017085e-01 |
| target_style_target_path | ll | 5.471273e-07 | 1.638331e-05 |
| target_style_target_path | lh | 1.772855e-04 | 1.635209e-01 |
| target_style_target_path | hl | 1.737852e-04 | 2.144648e-01 |
| target_style_target_path | hh | 3.462376e-04 | 6.219775e-01 |

#### loss_fm_spectral_lh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.194077e-05 | 2.655405e-02 |
| content | lh | 1.617415e-04 | 9.638245e-01 |
| content | hl | 1.000188e-05 | 4.614440e-03 |
| content | hh | 1.222726e-05 | 4.879697e-03 |
| target_style_target_path | ll | 1.353981e-07 | 6.241635e-06 |
| target_style_target_path | lh | 1.749284e-04 | 9.903665e-01 |
| target_style_target_path | hl | 1.022354e-05 | 4.617224e-03 |
| target_style_target_path | hh | 1.229958e-05 | 4.882642e-03 |

#### loss_fm_spectral_hl

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.567245e-05 | 3.469158e-02 |
| content | lh | 1.569611e-05 | 6.883721e-03 |
| content | hl | 1.649221e-04 | 9.514754e-01 |
| content | hh | 1.663884e-05 | 6.852745e-03 |
| target_style_target_path | ll | 2.446948e-07 | 1.522751e-05 |
| target_style_target_path | lh | 1.675186e-05 | 6.784333e-03 |
| target_style_target_path | hl | 1.728914e-04 | 9.863515e-01 |
| target_style_target_path | hh | 1.673724e-05 | 6.753805e-03 |

#### loss_fm_spectral_hh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.571817e-05 | 1.238610e-02 |
| content | lh | 2.044382e-05 | 4.145181e-03 |
| content | hl | 1.723094e-05 | 3.686701e-03 |
| content | hh | 3.339318e-04 | 9.797478e-01 |
| target_style_target_path | ll | 2.435617e-07 | 5.201901e-06 |
| target_style_target_path | lh | 2.181891e-05 | 3.968370e-03 |
| target_style_target_path | hl | 1.761281e-05 | 3.529447e-03 |
| target_style_target_path | hh | 3.455292e-04 | 9.924642e-01 |

### condition_only

#### loss

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.751373e-05 | 2.318026e-02 |
| content | lh | 1.645525e-04 | 1.640277e-01 |
| content | hl | 1.669598e-04 | 2.114134e-01 |
| content | hh | 3.347515e-04 | 6.013577e-01 |
| target_style_condition_path | ll | 2.467186e-14 | 1.634387e-16 |
| target_style_condition_path | lh | 4.864999e-06 | 6.041133e-01 |
| target_style_condition_path | hl | 2.504374e-06 | 2.185016e-01 |
| target_style_condition_path | hh | 1.738678e-06 | 7.694684e-02 |

#### loss_fm_hf_total

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.745109e-05 | 2.311079e-02 |
| content | lh | 1.642992e-04 | 1.637779e-01 |
| content | hl | 1.668174e-04 | 2.113818e-01 |
| content | hh | 3.345886e-04 | 6.017085e-01 |
| target_style_condition_path | ll | 2.432689e-14 | 1.589001e-16 |
| target_style_condition_path | lh | 4.864999e-06 | 6.041133e-01 |
| target_style_condition_path | hl | 2.504374e-06 | 2.185016e-01 |
| target_style_condition_path | hh | 1.738678e-06 | 7.694684e-02 |

#### loss_fm_spectral_lh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.194077e-05 | 2.655404e-02 |
| content | lh | 1.617415e-04 | 9.638245e-01 |
| content | hl | 1.000188e-05 | 4.614438e-03 |
| content | hh | 1.222726e-05 | 4.879698e-03 |
| target_style_condition_path | ll | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | lh | 4.864999e-06 | 8.574437e-01 |
| target_style_condition_path | hl | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hh | 0.000000e+00 | 0.000000e+00 |

#### loss_fm_spectral_hl

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.567244e-05 | 3.469155e-02 |
| content | lh | 1.569610e-05 | 6.883716e-03 |
| content | hl | 1.649221e-04 | 9.514755e-01 |
| content | hh | 1.663883e-05 | 6.852743e-03 |
| target_style_condition_path | ll | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | lh | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hl | 2.504374e-06 | 6.850871e-01 |
| target_style_condition_path | hh | 0.000000e+00 | 0.000000e+00 |

#### loss_fm_spectral_hh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.571818e-05 | 1.238610e-02 |
| content | lh | 2.044383e-05 | 4.145182e-03 |
| content | hl | 1.723095e-05 | 3.686703e-03 |
| content | hh | 3.339318e-04 | 9.797478e-01 |
| target_style_condition_path | ll | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | lh | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hl | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hh | 1.738678e-06 | 4.337841e-01 |
