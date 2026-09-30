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
- `probe_hf_stat_loss_enabled`: `True`
- `probe_hf_stat_weight`: `2.0`
- `target_latent_hf_subband_fusion_enabled`: `True`
- `style_cross_attention_enabled`: `True`
- `cfg_dropout_prob`: `0.0`

## Group Gradient Cosines

### fm_hf_vs_stat

| group | cosine | left norm | right norm |
|---|---:|---:|---:|
| time_proj | -0.817074 | 1.836477e-01 | 1.218590e+00 |
| head_ll | 0.000000 | 0.000000e+00 | 0.000000e+00 |
| target_hf_subband_fusion | 0.294258 | 8.182500e-02 | 8.960673e-01 |
| input_proj | 0.385305 | 5.614380e-01 | 4.384989e+00 |
| block3.cross_attn_q | 0.408498 | 2.790427e-02 | 3.402816e-01 |
| block3.ffn | 0.443269 | 1.406936e-01 | 1.241455e+00 |
| head_hh | 0.446988 | 3.725044e-01 | 5.290408e-01 |
| block0.adaln | 0.565729 | 8.062722e-02 | 9.938428e-01 |
| block2.ffn | 0.587905 | 1.662893e-01 | 1.859538e+00 |
| block0.ffn | 0.588869 | 2.919150e-01 | 2.897251e+00 |
| block3.cross_attn_kv | 0.627429 | 4.222536e-02 | 4.829834e-01 |
| block0.self_attn | 0.627838 | 1.724906e-01 | 1.749995e+00 |
| block1.ffn | 0.640733 | 2.157165e-01 | 2.523836e+00 |
| block3.self_attn | 0.669436 | 8.334056e-02 | 1.030164e+00 |
| head_lh | 0.676718 | 2.977279e-01 | 9.501547e-01 |
| head_hl | 0.705197 | 3.191353e-01 | 1.312705e+00 |
| block2.self_attn | 0.724726 | 9.552091e-02 | 1.228259e+00 |
| block1.adaln | 0.735461 | 6.154874e-02 | 9.832995e-01 |
| block2.cross_attn_kv | 0.759620 | 5.409693e-02 | 5.480999e-01 |
| block0.cross_attn_kv | 0.759999 | 3.851636e-02 | 4.352678e-01 |
| style_memory | 0.760177 | 4.533537e-02 | 3.690833e-01 |
| block1.self_attn | 0.767527 | 1.201660e-01 | 1.647287e+00 |
| block2.cross_attn_q | 0.767706 | 5.243789e-02 | 4.896793e-01 |
| block2.adaln | 0.781460 | 4.319503e-02 | 6.859607e-01 |
| block0.cross_attn_q | 0.806335 | 2.752254e-02 | 2.581959e-01 |
| style_conditioner.patch_proj | 0.846151 | 7.904832e-02 | 1.108290e+00 |
| block1.cross_attn_q | 0.857432 | 1.983508e-02 | 3.089212e-01 |
| block3.adaln | 0.860972 | 3.004896e-02 | 4.855212e-01 |
| block1.cross_attn_kv | 0.875583 | 2.768900e-02 | 3.963602e-01 |
| block3.cross_attn_out_gate | 0.903194 | 6.731903e-02 | 1.231843e+00 |
| block1.cross_attn_out_gate | 0.968898 | 5.528707e-02 | 1.042034e+00 |
| block2.cross_attn_out_gate | 0.981685 | 1.097635e-01 | 1.717682e+00 |
| block0.cross_attn_out_gate | 0.990462 | 1.601174e-01 | 1.502198e+00 |

### lh_mse_vs_lh_stat

| group | cosine | left norm | right norm |
|---|---:|---:|---:|
| head_ll | 0.000000 | 0.000000e+00 | 0.000000e+00 |
| head_hl | 0.000000 | 0.000000e+00 | 0.000000e+00 |
| head_hh | 0.000000 | 0.000000e+00 | 0.000000e+00 |
| block3.cross_attn_q | 0.240683 | 1.240170e-02 | 1.021161e-01 |
| block3.cross_attn_out_gate | 0.347380 | 1.594753e-02 | 3.270647e-01 |
| block3.ffn | 0.348770 | 7.224689e-02 | 4.389811e-01 |
| block3.cross_attn_kv | 0.387460 | 1.583834e-02 | 1.493372e-01 |
| block2.ffn | 0.416117 | 6.520832e-02 | 6.019490e-01 |
| style_memory | 0.480209 | 1.061157e-02 | 1.038414e-01 |
| target_hf_subband_fusion | 0.482566 | 3.865544e-02 | 2.118794e-01 |
| block1.ffn | 0.486146 | 7.015989e-02 | 7.915471e-01 |
| block2.adaln | 0.527908 | 1.352440e-02 | 2.026483e-01 |
| input_proj | 0.534508 | 1.504605e-01 | 1.219025e+00 |
| block2.self_attn | 0.541744 | 2.925292e-02 | 3.715009e-01 |
| block3.self_attn | 0.547147 | 3.737300e-02 | 3.238533e-01 |
| style_conditioner.patch_proj | 0.549556 | 1.827076e-02 | 3.136307e-01 |
| block0.ffn | 0.550590 | 8.887789e-02 | 8.814518e-01 |
| block2.cross_attn_kv | 0.590876 | 1.206674e-02 | 1.707590e-01 |
| block2.cross_attn_q | 0.609914 | 1.152460e-02 | 1.630891e-01 |
| block3.adaln | 0.618992 | 1.277449e-02 | 1.519319e-01 |
| time_proj | 0.627725 | 7.697537e-02 | 2.151905e-01 |
| block1.cross_attn_q | 0.633979 | 4.690898e-03 | 8.345014e-02 |
| block1.self_attn | 0.642094 | 3.424101e-02 | 4.840969e-01 |
| block0.adaln | 0.665325 | 2.297424e-02 | 2.861784e-01 |
| block1.cross_attn_kv | 0.667883 | 6.296205e-03 | 1.047940e-01 |
| head_lh | 0.676718 | 2.977279e-01 | 9.501547e-01 |
| block1.adaln | 0.724222 | 2.081516e-02 | 2.826097e-01 |
| block0.self_attn | 0.741271 | 4.719121e-02 | 4.811123e-01 |
| block2.cross_attn_out_gate | 0.745145 | 1.586531e-02 | 4.268695e-01 |
| block0.cross_attn_kv | 0.745253 | 8.440506e-03 | 1.246180e-01 |
| block0.cross_attn_q | 0.756586 | 4.199366e-03 | 7.550160e-02 |
| block1.cross_attn_out_gate | 0.815271 | 8.961142e-03 | 2.707178e-01 |
| block0.cross_attn_out_gate | 0.899536 | 1.370748e-02 | 4.423610e-01 |

### hl_mse_vs_hl_stat

| group | cosine | left norm | right norm |
|---|---:|---:|---:|
| time_proj | -0.857633 | 1.141331e-01 | 4.993496e-01 |
| head_ll | 0.000000 | 0.000000e+00 | 0.000000e+00 |
| head_lh | 0.000000 | 0.000000e+00 | 0.000000e+00 |
| head_hh | 0.000000 | 0.000000e+00 | 0.000000e+00 |
| block3.cross_attn_q | 0.025347 | 1.574588e-02 | 1.158309e-01 |
| input_proj | 0.068999 | 3.754887e-01 | 1.880596e+00 |
| block0.cross_attn_kv | 0.259569 | 1.649364e-02 | 1.651306e-01 |
| block0.adaln | 0.293799 | 4.549899e-02 | 3.812809e-01 |
| block0.self_attn | 0.302689 | 9.827656e-02 | 6.760988e-01 |
| block0.cross_attn_q | 0.354152 | 1.118889e-02 | 9.452158e-02 |
| block0.ffn | 0.377958 | 1.695165e-01 | 1.229651e+00 |
| block3.cross_attn_kv | 0.399835 | 2.395011e-02 | 1.929006e-01 |
| target_hf_subband_fusion | 0.470226 | 3.108727e-02 | 1.344113e-01 |
| block1.ffn | 0.490834 | 1.180895e-01 | 1.115661e+00 |
| block2.ffn | 0.491611 | 9.779698e-02 | 8.482724e-01 |
| block3.ffn | 0.509790 | 8.091678e-02 | 5.977356e-01 |
| block3.self_attn | 0.560780 | 4.719324e-02 | 4.397546e-01 |
| block1.adaln | 0.576647 | 3.045774e-02 | 3.879712e-01 |
| style_memory | 0.614096 | 1.938393e-02 | 1.336513e-01 |
| block1.self_attn | 0.628971 | 6.148102e-02 | 6.494758e-01 |
| block2.self_attn | 0.633387 | 4.935888e-02 | 5.038141e-01 |
| block3.cross_attn_out_gate | 0.649586 | 2.568385e-02 | 4.116722e-01 |
| style_conditioner.patch_proj | 0.682460 | 3.560863e-02 | 4.342543e-01 |
| head_hl | 0.705197 | 3.191353e-01 | 1.312705e+00 |
| block3.adaln | 0.725704 | 1.672599e-02 | 2.037365e-01 |
| block2.adaln | 0.777315 | 2.112366e-02 | 2.826623e-01 |
| block2.cross_attn_kv | 0.791314 | 2.943229e-02 | 2.823708e-01 |
| block1.cross_attn_kv | 0.808026 | 1.319850e-02 | 1.575470e-01 |
| block1.cross_attn_q | 0.808174 | 1.031185e-02 | 1.174806e-01 |
| block2.cross_attn_q | 0.820246 | 2.976410e-02 | 2.599475e-01 |
| block1.cross_attn_out_gate | 0.935648 | 2.185811e-02 | 3.811178e-01 |
| block0.cross_attn_out_gate | 0.967464 | 4.847065e-02 | 5.324930e-01 |
| block2.cross_attn_out_gate | 0.980575 | 6.099128e-02 | 7.629326e-01 |

### hh_mse_vs_hh_stat

| group | cosine | left norm | right norm |
|---|---:|---:|---:|
| time_proj | -0.938790 | 1.450611e-01 | 5.404597e-01 |
| head_ll | 0.000000 | 0.000000e+00 | 0.000000e+00 |
| head_lh | 0.000000 | 0.000000e+00 | 0.000000e+00 |
| head_hl | 0.000000 | 0.000000e+00 | 0.000000e+00 |
| block3.ffn | 0.244966 | 7.006178e-02 | 3.731386e-01 |
| target_hf_subband_fusion | 0.279716 | 6.507437e-02 | 8.602195e-01 |
| block0.adaln | 0.341992 | 3.577600e-02 | 3.380932e-01 |
| block1.adaln | 0.369750 | 2.672182e-02 | 3.214595e-01 |
| block2.ffn | 0.434010 | 8.716305e-02 | 5.631245e-01 |
| block0.ffn | 0.442813 | 1.255255e-01 | 9.424425e-01 |
| head_hh | 0.446988 | 3.725044e-01 | 5.290408e-01 |
| input_proj | 0.450965 | 2.167551e-01 | 1.515291e+00 |
| block1.ffn | 0.517461 | 1.049254e-01 | 7.913879e-01 |
| block3.self_attn | 0.524826 | 3.601757e-02 | 2.943460e-01 |
| block1.cross_attn_q | 0.546000 | 9.224495e-03 | 1.145724e-01 |
| block2.cross_attn_kv | 0.565037 | 2.419671e-02 | 1.699634e-01 |
| block2.self_attn | 0.571993 | 4.333435e-02 | 3.801564e-01 |
| block2.cross_attn_q | 0.572380 | 2.126029e-02 | 1.562586e-01 |
| block0.self_attn | 0.586073 | 6.325354e-02 | 6.124170e-01 |
| block2.adaln | 0.586758 | 1.845250e-02 | 2.081532e-01 |
| block1.self_attn | 0.588975 | 5.071181e-02 | 5.310467e-01 |
| block3.cross_attn_kv | 0.644156 | 2.581382e-02 | 1.618080e-01 |
| block3.adaln | 0.653395 | 1.215261e-02 | 1.371153e-01 |
| block1.cross_attn_kv | 0.695946 | 1.257470e-02 | 1.405157e-01 |
| block3.cross_attn_q | 0.745305 | 1.862288e-02 | 1.487299e-01 |
| block0.cross_attn_kv | 0.784366 | 2.381130e-02 | 1.503858e-01 |
| style_conditioner.patch_proj | 0.808204 | 4.406427e-02 | 3.893273e-01 |
| style_memory | 0.823733 | 2.111279e-02 | 1.392051e-01 |
| block0.cross_attn_q | 0.861476 | 1.730422e-02 | 9.119538e-02 |
| block3.cross_attn_out_gate | 0.920853 | 4.459593e-02 | 5.031148e-01 |
| block2.cross_attn_out_gate | 0.940569 | 3.873826e-02 | 5.354059e-01 |
| block1.cross_attn_out_gate | 0.962997 | 2.696609e-02 | 3.923001e-01 |
| block0.cross_attn_out_gate | 0.988023 | 1.004900e-01 | 5.285172e-01 |

## Residual Output Activation Gradients

### loss

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.646076e-02 | 1.683301e-04 | 2.201522e-03 |
| hl | 7.166816e-02 | 1.985419e-04 | 2.770294e-03 |
| hh | 1.318640e-01 | 2.269822e-04 | 1.721336e-03 |

### loss_fm_hf_total

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.646076e-02 | 8.678105e-05 | 1.134975e-03 |
| hl | 7.166816e-02 | 9.953389e-05 | 1.388816e-03 |
| hh | 1.318640e-01 | 1.700700e-04 | 1.289738e-03 |

### loss_stat

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.646076e-02 | 1.419980e-04 | 1.857136e-03 |
| hl | 7.166816e-02 | 1.650977e-04 | 2.303641e-03 |
| hh | 1.318640e-01 | 1.461410e-04 | 1.108271e-03 |

### loss_fm_spectral_lh

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.646076e-02 | 8.678105e-05 | 1.134975e-03 |
| hl | 7.166816e-02 | 0.000000e+00 | 0.000000e+00 |
| hh | 1.318640e-01 | 0.000000e+00 | 0.000000e+00 |

### loss_stat_lh

| band | output rms | grad rms | grad/output |
|---|---:|---:|---:|
| lh | 7.646076e-02 | 1.419980e-04 | 1.857136e-03 |
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
| hl | 7.166816e-02 | 1.650977e-04 | 2.303641e-03 |
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
| hh | 1.318640e-01 | 1.461410e-04 | 1.108271e-03 |

## Style-Latent Band Information Flow

### full target style_latent vs content condition

| output band | delta/base | delta rms |
|---|---:|---:|
| ll | 0.000000e+00 | 0.000000e+00 |
| lh | 7.562427e-02 | 1.178426e-02 |
| hl | 9.722850e-02 | 1.684589e-02 |
| hh | 1.194318e-01 | 1.332076e-02 |

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
| content | ll | 1.036214e-04 | 6.367541e-02 |
| content | lh | 4.399239e-04 | 2.270475e-01 |
| content | hl | 4.493017e-04 | 2.965092e-01 |
| content | hh | 6.302030e-04 | 4.127639e-01 |
| target_style_shared | ll | 6.658325e-06 | 4.978748e-04 |
| target_style_shared | lh | 4.765091e-04 | 2.424011e-01 |
| target_style_shared | hl | 4.672541e-04 | 3.181279e-01 |
| target_style_shared | hh | 6.421273e-04 | 4.389689e-01 |

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

#### loss_stat

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 9.634900e-05 | 1.474815e-01 |
| content | lh | 2.867980e-04 | 2.585140e-01 |
| content | hl | 2.904228e-04 | 3.318898e-01 |
| content | hh | 3.068182e-04 | 2.621039e-01 |
| target_style_shared | ll | 6.161123e-06 | 1.232628e-03 |
| target_style_shared | lh | 3.132587e-04 | 3.029149e-01 |
| target_style_shared | hl | 3.062129e-04 | 3.950615e-01 |
| target_style_shared | hh | 3.125840e-04 | 3.007788e-01 |

#### loss_fm_spectral_lh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.194368e-05 | 2.655058e-02 |
| content | lh | 1.617922e-04 | 9.638321e-01 |
| content | hl | 1.000355e-05 | 4.613124e-03 |
| content | hh | 1.222762e-05 | 4.876962e-03 |
| target_style_shared | ll | 1.354055e-07 | 6.250181e-06 |
| target_style_shared | lh | 1.748171e-04 | 9.903526e-01 |
| target_style_shared | hl | 1.022524e-05 | 4.624581e-03 |
| target_style_shared | hh | 1.229993e-05 | 4.889074e-03 |

#### loss_stat_lh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 3.823307e-05 | 9.028332e-02 |
| content | lh | 2.689574e-04 | 8.838635e-01 |
| content | hl | 2.899230e-05 | 1.285831e-02 |
| content | hh | 3.459240e-05 | 1.295266e-02 |
| target_style_shared | ll | 1.924013e-06 | 4.360962e-04 |
| target_style_shared | lh | 2.946979e-04 | 9.725737e-01 |
| target_style_shared | hl | 2.963481e-05 | 1.342382e-02 |
| target_style_shared | hh | 3.479698e-05 | 1.352231e-02 |

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

#### loss_stat_hl

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 5.042534e-05 | 1.113568e-01 |
| content | lh | 4.881395e-05 | 2.064410e-02 |
| content | hl | 2.800279e-04 | 8.505724e-01 |
| content | hh | 4.760917e-05 | 1.739677e-02 |
| target_style_shared | ll | 2.622730e-06 | 5.783331e-04 |
| target_style_shared | lh | 5.209726e-05 | 2.169215e-02 |
| target_style_shared | hl | 2.965622e-04 | 9.594181e-01 |
| target_style_shared | hh | 4.789074e-05 | 1.827997e-02 |

#### loss_fm_spectral_hh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.571182e-05 | 1.236851e-02 |
| content | lh | 2.042885e-05 | 4.136576e-03 |
| content | hl | 1.721852e-05 | 3.679133e-03 |
| content | hh | 3.340399e-04 | 9.797815e-01 |
| target_style_shared | ll | 2.427987e-07 | 5.179293e-06 |
| target_style_shared | lh | 2.180293e-05 | 3.970175e-03 |
| target_style_shared | hl | 1.760011e-05 | 3.531132e-03 |
| target_style_shared | hh | 3.451970e-04 | 9.924606e-01 |

#### loss_stat_hh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.830346e-05 | 4.844024e-02 |
| content | lh | 3.792035e-05 | 1.720129e-02 |
| content | hl | 3.108097e-05 | 1.446791e-02 |
| content | hh | 2.946188e-04 | 9.198492e-01 |
| target_style_shared | ll | 1.826623e-06 | 3.855190e-04 |
| target_style_shared | lh | 4.047094e-05 | 1.799022e-02 |
| target_style_shared | hl | 3.176977e-05 | 1.513148e-02 |
| target_style_shared | hh | 2.970400e-04 | 9.664496e-01 |

### target_only

#### loss

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.036214e-04 | 6.367536e-02 |
| content | lh | 4.399239e-04 | 2.270475e-01 |
| content | hl | 4.493017e-04 | 2.965092e-01 |
| content | hh | 6.302030e-04 | 4.127639e-01 |
| target_style_target_path | ll | 6.658319e-06 | 4.971147e-04 |
| target_style_target_path | lh | 4.765560e-04 | 2.420790e-01 |
| target_style_target_path | hl | 4.669957e-04 | 3.172915e-01 |
| target_style_target_path | hh | 6.434655e-04 | 4.401282e-01 |

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

#### loss_stat

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 9.634900e-05 | 1.474815e-01 |
| content | lh | 2.867980e-04 | 2.585140e-01 |
| content | hl | 2.904228e-04 | 3.318898e-01 |
| content | hh | 3.068182e-04 | 2.621039e-01 |
| target_style_target_path | ll | 6.161123e-06 | 1.230541e-03 |
| target_style_target_path | lh | 3.132433e-04 | 3.023722e-01 |
| target_style_target_path | hl | 3.059249e-04 | 3.936511e-01 |
| target_style_target_path | hh | 3.138642e-04 | 3.027341e-01 |

#### loss_fm_spectral_lh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.194369e-05 | 2.655059e-02 |
| content | lh | 1.617922e-04 | 9.638321e-01 |
| content | hl | 1.000355e-05 | 4.613125e-03 |
| content | hh | 1.222762e-05 | 4.876962e-03 |
| target_style_target_path | ll | 1.354054e-07 | 6.245518e-06 |
| target_style_target_path | lh | 1.748829e-04 | 9.903598e-01 |
| target_style_target_path | hl | 1.022524e-05 | 4.621138e-03 |
| target_style_target_path | hh | 1.229993e-05 | 4.885433e-03 |

#### loss_stat_lh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 3.823307e-05 | 9.028334e-02 |
| content | lh | 2.689574e-04 | 8.838635e-01 |
| content | hl | 2.899229e-05 | 1.285831e-02 |
| content | hh | 3.459240e-05 | 1.295266e-02 |
| target_style_target_path | ll | 1.924013e-06 | 4.351867e-04 |
| target_style_target_path | lh | 2.950144e-04 | 9.726309e-01 |
| target_style_target_path | hl | 2.963480e-05 | 1.339582e-02 |
| target_style_target_path | hh | 3.479698e-05 | 1.349411e-02 |

#### loss_fm_spectral_hl

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.567538e-05 | 3.468399e-02 |
| content | lh | 1.569804e-05 | 6.881342e-03 |
| content | hl | 1.649718e-04 | 9.514851e-01 |
| content | hh | 1.664417e-05 | 6.853078e-03 |
| target_style_target_path | ll | 2.445845e-07 | 1.522062e-05 |
| target_style_target_path | lh | 1.675392e-05 | 6.789060e-03 |
| target_style_target_path | hl | 1.728515e-04 | 9.863394e-01 |
| target_style_target_path | hh | 1.674261e-05 | 6.761175e-03 |

#### loss_stat_hl

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 5.042534e-05 | 1.113568e-01 |
| content | lh | 4.881395e-05 | 2.064410e-02 |
| content | hl | 2.800279e-04 | 8.505724e-01 |
| content | hh | 4.760917e-05 | 1.739677e-02 |
| target_style_target_path | ll | 2.622730e-06 | 5.787334e-04 |
| target_style_target_path | lh | 5.209726e-05 | 2.170717e-02 |
| target_style_target_path | hl | 2.964553e-04 | 9.593900e-01 |
| target_style_target_path | hh | 4.789074e-05 | 1.829262e-02 |

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

#### loss_stat_hh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.830346e-05 | 4.844024e-02 |
| content | lh | 3.792035e-05 | 1.720129e-02 |
| content | hl | 3.108097e-05 | 1.446791e-02 |
| content | hh | 2.946188e-04 | 9.198492e-01 |
| target_style_target_path | ll | 1.826623e-06 | 3.825841e-04 |
| target_style_target_path | lh | 4.047094e-05 | 1.785327e-02 |
| target_style_target_path | hl | 3.176977e-05 | 1.501629e-02 |
| target_style_target_path | hh | 2.982166e-04 | 9.667050e-01 |

### condition_only

#### loss

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.036214e-04 | 6.367536e-02 |
| content | lh | 4.399239e-04 | 2.270475e-01 |
| content | hl | 4.493017e-04 | 2.965092e-01 |
| content | hh | 6.302029e-04 | 4.127639e-01 |
| target_style_condition_path | ll | 6.175457e-14 | 1.670427e-16 |
| target_style_condition_path | lh | 9.374343e-06 | 3.659089e-01 |
| target_style_condition_path | hl | 7.500045e-06 | 3.196852e-01 |
| target_style_condition_path | hh | 8.471851e-06 | 2.980213e-01 |

#### loss_fm_hf_total

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.744820e-05 | 2.309202e-02 |
| content | lh | 1.643476e-04 | 1.637758e-01 |
| content | hl | 1.668665e-04 | 2.113790e-01 |
| content | hh | 3.346960e-04 | 6.017322e-01 |
| target_style_condition_path | ll | 2.207278e-14 | 1.504360e-16 |
| target_style_condition_path | lh | 4.476792e-06 | 5.882659e-01 |
| target_style_condition_path | hl | 2.291315e-06 | 2.103356e-01 |
| target_style_condition_path | hh | 1.713052e-06 | 8.589742e-02 |

#### loss_stat

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 9.634900e-05 | 1.474815e-01 |
| content | lh | 2.867980e-04 | 2.585140e-01 |
| content | hl | 2.904228e-04 | 3.318898e-01 |
| content | hh | 3.068182e-04 | 2.621039e-01 |
| target_style_condition_path | ll | 5.236095e-14 | 2.137718e-16 |
| target_style_condition_path | lh | 5.407272e-06 | 2.167175e-01 |
| target_style_condition_path | hl | 5.955361e-06 | 3.588038e-01 |
| target_style_condition_path | hh | 7.313104e-06 | 3.953123e-01 |

#### loss_fm_spectral_lh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.194368e-05 | 2.655058e-02 |
| content | lh | 1.617922e-04 | 9.638321e-01 |
| content | hl | 1.000355e-05 | 4.613124e-03 |
| content | hh | 1.222762e-05 | 4.876962e-03 |
| target_style_condition_path | ll | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | lh | 4.476792e-06 | 8.358816e-01 |
| target_style_condition_path | hl | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hh | 0.000000e+00 | 0.000000e+00 |

#### loss_stat_lh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 3.823307e-05 | 9.028332e-02 |
| content | lh | 2.689574e-04 | 8.838635e-01 |
| content | hl | 2.899230e-05 | 1.285831e-02 |
| content | hh | 3.459240e-05 | 1.295266e-02 |
| target_style_condition_path | ll | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | lh | 5.407272e-06 | 8.813812e-01 |
| target_style_condition_path | hl | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hh | 0.000000e+00 | 0.000000e+00 |

#### loss_fm_spectral_hl

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.567538e-05 | 3.468399e-02 |
| content | lh | 1.569804e-05 | 6.881342e-03 |
| content | hl | 1.649718e-04 | 9.514851e-01 |
| content | hh | 1.664417e-05 | 6.853078e-03 |
| target_style_condition_path | ll | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | lh | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hl | 2.291315e-06 | 6.455246e-01 |
| target_style_condition_path | hh | 0.000000e+00 | 0.000000e+00 |

#### loss_stat_hl

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 5.042534e-05 | 1.113568e-01 |
| content | lh | 4.881395e-05 | 2.064410e-02 |
| content | hl | 2.800279e-04 | 8.505724e-01 |
| content | hh | 4.760917e-05 | 1.739677e-02 |
| target_style_condition_path | ll | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | lh | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hl | 5.955361e-06 | 9.248230e-01 |
| target_style_condition_path | hh | 0.000000e+00 | 0.000000e+00 |

#### loss_fm_spectral_hh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 1.571182e-05 | 1.236850e-02 |
| content | lh | 2.042885e-05 | 4.136576e-03 |
| content | hl | 1.721853e-05 | 3.679135e-03 |
| content | hh | 3.340399e-04 | 9.797815e-01 |
| target_style_condition_path | ll | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | lh | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hl | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hh | 1.713052e-06 | 4.265047e-01 |

#### loss_stat_hh

| tensor | band | grad/tensor | power share |
|---|---|---:|---:|
| content | ll | 2.830346e-05 | 4.844024e-02 |
| content | lh | 3.792035e-05 | 1.720129e-02 |
| content | hl | 3.108097e-05 | 1.446792e-02 |
| content | hh | 2.946188e-04 | 9.198492e-01 |
| target_style_condition_path | ll | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | lh | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hl | 0.000000e+00 | 0.000000e+00 |
| target_style_condition_path | hh | 7.313104e-06 | 9.312888e-01 |
