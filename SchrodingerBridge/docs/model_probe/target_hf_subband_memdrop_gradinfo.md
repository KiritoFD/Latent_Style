# Gradient and Information-Flow Probe

Config: `configs\exp_probe_target_hf_subband_memdrop_ft6.json`
Checkpoint: `exp\model_probe\target_hf_subband_memdrop_ft6\epoch_0006.pt`
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
| lh | 7.813126e-02 | 8.686209e-05 | 1.111746e-03 |
| hl | 7.278859e-02 | 9.961838e-05 | 1.368599e-03 |
| hh | 1.380728e-01 | 1.702278e-04 | 1.232885e-03 |

### loss_fm_hf_total

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.813260e-02 | 8.687929e-05 | 1.111947e-03 |
| hl | 7.494088e-02 | 9.986140e-05 | 1.332536e-03 |
| hh | 1.330474e-01 | 1.707451e-04 | 1.283341e-03 |

### loss_stat

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.769872e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.108393e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.363710e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_lh

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.769872e-02 | 8.686922e-05 | 1.118026e-03 |
| hl | 7.108393e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.363710e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_stat_lh

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.813126e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.278859e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.380728e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_hl

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.675483e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.315724e-02 | 1.001284e-04 | 1.368674e-03 |
| hh | 1.336315e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_stat_hl

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.754241e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.301041e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.299548e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_hh

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.754106e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.079946e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.350953e-01 | 1.701757e-04 | 1.259672e-03 |

### loss_stat_hh

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.813260e-02 | 0.000000e+00 | 0.000000e+00 |
| hl | 7.494088e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.330474e-01 | 0.000000e+00 | 0.000000e+00 |

## Style-Latent Band Information Flow

### full target style_latent vs content condition

| output band | delta/base | delta rms |
|---|---:|---:|
| ll | 0.000000e+00 | 0.000000e+00 |
| lh | 1.008119e-01 | 1.534483e-02 |
| hl | 1.181068e-01 | 2.024030e-02 |
| hh | 1.223399e-01 | 1.366247e-02 |

### full target condition direction alignment

| output band | delta/desired | cos(delta, desired) | projection | orthogonal fraction | MSE improvement |
|---|---:|---:|---:|---:|---:|
| lh | 2.154282e-02 | 5.483997e-02 | 1.217096e-03 | 9.983281e-01 | 2.021581e-03 |
| hl | 2.477687e-02 | 4.859784e-02 | 1.271588e-03 | 9.986159e-01 | 2.029305e-03 |
| hh | 1.958581e-02 | 3.185172e-02 | 6.643466e-04 | 9.992977e-01 | 9.078431e-04 |

### single target condition band

| input band | output band | delta/base |
|---|---|---:|
| ll | ll | 0.000000e+00 |
| ll | lh | 3.414049e-06 |
| ll | hl | 2.227086e-06 |
| ll | hh | 5.047250e-06 |
| lh | ll | 0.000000e+00 |
| lh | lh | 1.008118e-01 |
| lh | hl | 2.643986e-06 |
| lh | hh | 7.970063e-06 |
| hl | ll | 0.000000e+00 |
| hl | lh | 4.742948e-06 |
| hl | hl | 1.181068e-01 |
| hl | hh | 8.308464e-06 |
| hh | ll | 0.000000e+00 |
| hh | lh | 4.819555e-06 |
| hh | hl | 2.720229e-06 |
| hh | hh | 1.223400e-01 |

### single target condition band direction alignment

| input band | output band | delta/desired | cos(delta, desired) | projection | MSE improvement |
|---|---|---:|---:|---:|---:|
| ll | lh | 7.295593e-07 | -2.459315e-03 | -3.475398e-10 | 2.349583e-07 |
| ll | hl | 4.672064e-07 | -1.100792e-02 | -8.210876e-09 | 0.000000e+00 |
| ll | hh | 8.080314e-07 | 1.335360e-03 | 1.614707e-09 | 6.124557e-08 |
| lh | lh | 2.154281e-02 | 5.483938e-02 | 1.217089e-03 | 2.021581e-03 |
| lh | hl | 5.546651e-07 | -5.444349e-03 | -5.740681e-09 | 0.000000e+00 |
| lh | hh | 1.275955e-06 | 8.137655e-03 | 1.483460e-08 | 0.000000e+00 |
| hl | lh | 1.013536e-06 | -1.019402e-02 | -8.460555e-09 | 0.000000e+00 |
| hl | hl | 2.477688e-02 | 4.859762e-02 | 1.271588e-03 | 2.029305e-03 |
| hl | hh | 1.330130e-06 | 2.003097e-03 | 7.017916e-09 | 0.000000e+00 |
| hh | lh | 1.029906e-06 | -2.857557e-03 | -4.865210e-09 | 1.174792e-07 |
| hh | hl | 5.706597e-07 | -1.383298e-02 | -5.953071e-09 | 0.000000e+00 |
| hh | hh | 1.958582e-02 | 3.185244e-02 | 6.643573e-04 | 9.079043e-04 |

### route interventions

| intervention | output band | delta/base |
|---|---|---:|
| target_hf_residual_contribution | ll | 0.000000e+00 |
| target_hf_residual_contribution | lh | 4.973317e-01 |
| target_hf_residual_contribution | hl | 4.065465e-01 |
| target_hf_residual_contribution | hh | 1.074072e+00 |
| cfg_unconditional_delta_from_full | ll | 0.000000e+00 |
| cfg_unconditional_delta_from_full | lh | 5.148676e-01 |
| cfg_unconditional_delta_from_full | hl | 4.242614e-01 |
| cfg_unconditional_delta_from_full | hh | 1.217135e+00 |

## Input Band Gradient Split

### full_shared

#### loss

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.714823e-05 | 2.395801e-02 |
| content | lh | 1.598267e-04 | 1.642690e-01 |
| content | hl | 1.625535e-04 | 2.127404e-01 |
| content | hh | 3.242643e-04 | 5.990104e-01 |
| target_style_shared | ll | 2.018044e-06 | 2.091731e-04 |
| target_style_shared | lh | 1.825350e-04 | 1.626818e-01 |
| target_style_shared | hl | 1.794259e-04 | 2.145458e-01 |
| target_style_shared | hh | 3.575709e-04 | 6.225440e-01 |

#### loss_fm_hf_total

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.674640e-05 | 2.265331e-02 |
| content | lh | 1.619173e-04 | 1.642391e-01 |
| content | hl | 1.643522e-04 | 2.118564e-01 |
| content | hh | 3.291437e-04 | 6.012295e-01 |
| target_style_shared | ll | 5.987118e-07 | 1.902875e-05 |
| target_style_shared | lh | 1.800326e-04 | 1.635607e-01 |
| target_style_shared | hl | 1.764868e-04 | 2.145383e-01 |
| target_style_shared | hh | 3.515273e-04 | 6.218622e-01 |

#### loss_fm_spectral_lh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.149161e-05 | 2.554098e-02 |
| content | lh | 1.588096e-04 | 9.649796e-01 |
| content | hl | 9.664808e-06 | 4.474572e-03 |
| content | hh | 1.198974e-05 | 4.872633e-03 |
| target_style_shared | ll | 1.493052e-07 | 7.322059e-06 |
| target_style_shared | lh | 1.781696e-04 | 9.911813e-01 |
| target_style_shared | hl | 9.878994e-06 | 4.159245e-03 |
| target_style_shared | hh | 1.206065e-05 | 4.529255e-03 |

#### loss_fm_spectral_hl

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.519541e-05 | 3.384152e-02 |
| content | lh | 1.528259e-05 | 6.771862e-03 |
| content | hl | 1.619935e-04 | 9.525982e-01 |
| content | hh | 1.613655e-05 | 6.688289e-03 |
| target_style_shared | ll | 2.478158e-07 | 1.511020e-05 |
| target_style_shared | lh | 1.631052e-05 | 6.222287e-03 |
| target_style_shared | hl | 1.758795e-04 | 9.875251e-01 |
| target_style_shared | hh | 1.623199e-05 | 6.145496e-03 |

#### loss_fm_spectral_hh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.524818e-05 | 1.203550e-02 |
| content | lh | 1.951425e-05 | 3.899605e-03 |
| content | hl | 1.703789e-05 | 3.721767e-03 |
| content | hh | 3.287252e-04 | 9.803077e-01 |
| target_style_shared | ll | 3.034191e-07 | 7.739397e-06 |
| target_style_shared | lh | 2.082681e-05 | 3.466321e-03 |
| target_style_shared | hl | 1.741547e-05 | 3.308243e-03 |
| target_style_shared | hh | 3.530238e-04 | 9.931863e-01 |

### target_only

#### loss

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.705920e-05 | 2.373698e-02 |
| content | lh | 1.600436e-04 | 1.642710e-01 |
| content | hl | 1.627845e-04 | 2.127703e-01 |
| content | hh | 3.247535e-04 | 5.991995e-01 |
| target_style_target_path | ll | 1.932076e-06 | 1.920691e-04 |
| target_style_target_path | lh | 1.824215e-04 | 1.627658e-01 |
| target_style_target_path | hl | 1.791295e-04 | 2.142144e-01 |
| target_style_target_path | hh | 3.573321e-04 | 6.228084e-01 |

#### loss_fm_hf_total

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.673920e-05 | 2.214317e-02 |
| content | lh | 1.634427e-04 | 1.636677e-01 |
| content | hl | 1.657504e-04 | 2.107374e-01 |
| content | hh | 3.334325e-04 | 6.034306e-01 |
| target_style_target_path | ll | 6.112042e-07 | 2.018151e-05 |
| target_style_target_path | lh | 1.776658e-04 | 1.621028e-01 |
| target_style_target_path | hl | 1.752536e-04 | 2.152882e-01 |
| target_style_target_path | hh | 3.486603e-04 | 6.225686e-01 |

#### loss_fm_spectral_lh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.142916e-05 | 2.582717e-02 |
| content | lh | 1.570328e-04 | 9.645357e-01 |
| content | hl | 9.631935e-06 | 4.543233e-03 |
| content | hh | 1.196272e-05 | 4.958797e-03 |
| target_style_target_path | ll | 1.522872e-07 | 7.445319e-06 |
| target_style_target_path | lh | 1.802401e-04 | 9.914279e-01 |
| target_style_target_path | hl | 9.845393e-06 | 4.037644e-03 |
| target_style_target_path | hh | 1.203347e-05 | 4.406962e-03 |

#### loss_fm_spectral_hl

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.527793e-05 | 3.425786e-02 |
| content | lh | 1.535359e-05 | 6.844475e-03 |
| content | hl | 1.618328e-04 | 9.520373e-01 |
| content | hh | 1.621153e-05 | 6.760018e-03 |
| target_style_target_path | ll | 2.611895e-07 | 1.673169e-05 |
| target_style_target_path | lh | 1.638630e-05 | 6.260257e-03 |
| target_style_target_path | hl | 1.761531e-04 | 9.874483e-01 |
| target_style_target_path | hh | 1.630741e-05 | 6.183009e-03 |

#### loss_fm_spectral_hh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.526346e-05 | 1.176923e-02 |
| content | lh | 1.942956e-05 | 3.772738e-03 |
| content | hl | 1.674360e-05 | 3.507754e-03 |
| content | hh | 3.328594e-04 | 9.809158e-01 |
| target_style_target_path | ll | 2.809602e-07 | 6.830333e-06 |
| target_style_target_path | lh | 2.073643e-05 | 3.536893e-03 |
| target_style_target_path | hl | 1.711467e-05 | 3.288474e-03 |
| target_style_target_path | hh | 3.479586e-04 | 9.931354e-01 |

### condition_only

#### loss

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.706441e-05 | 2.331718e-02 |
| content | lh | 1.615737e-04 | 1.644028e-01 |
| content | hl | 1.639788e-04 | 2.120038e-01 |
| content | hh | 3.280153e-04 | 6.002545e-01 |
| target_style_condition_path | ll | 2.650103e-14 | 1.653889e-16 |
| target_style_condition_path | lh | 5.006096e-06 | 5.610238e-01 |
| target_style_condition_path | hl | 2.828856e-06 | 2.445165e-01 |
| target_style_condition_path | hh | 2.182813e-06 | 1.063691e-01 |

#### loss_fm_hf_total

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.688263e-05 | 2.293276e-02 |
| content | lh | 1.618251e-04 | 1.643968e-01 |
| content | hl | 1.641871e-04 | 2.118752e-01 |
| content | hh | 3.286736e-04 | 6.007736e-01 |
| target_style_condition_path | ll | 2.778935e-14 | 1.670477e-16 |
| target_style_condition_path | lh | 5.347216e-06 | 5.879509e-01 |
| target_style_condition_path | hl | 2.892639e-06 | 2.348429e-01 |
| target_style_condition_path | hh | 2.166954e-06 | 9.629068e-02 |

#### loss_fm_spectral_lh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.149161e-05 | 2.554098e-02 |
| content | lh | 1.588096e-04 | 9.649796e-01 |
| content | hl | 9.664808e-06 | 4.474572e-03 |
| content | hh | 1.198974e-05 | 4.872633e-03 |
| target_style_condition_path | ll | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | lh | 5.006096e-06 | 8.642912e-01 |
| target_style_condition_path | hl | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hh | 0.000000e+00 | 0.000000e+00 |

#### loss_fm_spectral_hl

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.499648e-05 | 3.290219e-02 |
| content | lh | 1.479409e-05 | 6.334487e-03 |
| content | hl | 1.623115e-04 | 9.546282e-01 |
| content | hh | 1.534214e-05 | 6.035121e-03 |
| target_style_condition_path | ll | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | lh | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hl | 3.298282e-06 | 7.905057e-01 |
| target_style_condition_path | hh | 0.000000e+00 | 0.000000e+00 |

#### loss_fm_spectral_hh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.524818e-05 | 1.203550e-02 |
| content | lh | 1.951425e-05 | 3.899606e-03 |
| content | hl | 1.703789e-05 | 3.721767e-03 |
| content | hh | 3.287252e-04 | 9.803077e-01 |
| target_style_condition_path | ll | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | lh | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hl | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hh | 2.840319e-06 | 6.715389e-01 |
