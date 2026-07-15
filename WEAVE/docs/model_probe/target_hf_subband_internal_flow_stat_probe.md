# Baseline Internal Flow Probe

Config: `configs\exp_probe_target_hf_subband_ft6.json`
Checkpoint: `exp\model_probe\target_hf_subband_ft6\epoch_0006.pt`
Device: `cuda`
Batches: 2, batch size: 4
Load info: `{'missing': 0, 'unexpected': 0}`

## Loss Components

| component | value | weighted value |
|---|---:|---:|
| loss | 3.048249 | 3.048249 |
| loss_fm_spectral_ll | 0.081340 | 0.024402 |
| loss_fm_spectral_lh | 0.452708 | 0.452708 |
| loss_fm_spectral_hl | 0.490531 | 0.490531 |
| loss_fm_spectral_hh | 0.413025 | 0.826050 |
| t_mean | 0.393818 | 0.393818 |
| flow | 1.793690 | 1.793690 |
| stat | 1.254559 | 1.254559 |
| fft | 0.000000 | 0.000000 |

## Parameter Gradient Groups

| group | grad norm | grad/param |
|---|---:|---:|
| input_proj | 5.916607e+00 | 1.366548e+00 |
| time_proj | 4.865978e+00 | 6.818743e-01 |
| block0.ffn | 3.724554e+00 | 3.603154e-01 |
| block1.ffn | 3.122298e+00 | 2.954943e-01 |
| block1.cross_attn_out_gate | 2.831890e+00 | 6.009920e-01 |
| block2.cross_attn_out_gate | 2.345099e+00 | 5.045603e-01 |
| block2.ffn | 2.221750e+00 | 2.050119e-01 |
| block1.self_attn | 2.210564e+00 | 2.363557e-01 |
| block0.self_attn | 2.207807e+00 | 2.375414e-01 |
| target_hf_subband_fusion | 1.847092e+00 | 5.948820e-02 |
| block0.adaln | 1.522484e+00 | 1.116677e+00 |
| style_conditioner.patch_proj | 1.511211e+00 | 7.223920e-02 |
| block1.adaln | 1.482973e+00 | 1.096845e+00 |
| block2.self_attn | 1.353789e+00 | 1.457463e-01 |
| block3.ffn | 1.320030e+00 | 1.187303e-01 |
| block3.self_attn | 8.999348e-01 | 9.594159e-02 |
| block1.cross_attn_kv | 8.918915e-01 | 1.340858e-01 |
| block2.adaln | 8.817148e-01 | 7.711810e-01 |
| head_lh | 8.443309e-01 | 9.758021e-02 |
| head_hh | 8.236850e-01 | 9.931095e-02 |
| head_hl | 7.602817e-01 | 8.799521e-02 |
| block3.cross_attn_kv | 6.448471e-01 | 9.667950e-02 |
| style_memory | 6.446065e-01 | 4.384077e-02 |
| block1.cross_attn_q | 6.420400e-01 | 1.338126e-01 |
| block2.cross_attn_kv | 6.419955e-01 | 9.656925e-02 |
| block3.cross_attn_out_gate | 5.792374e-01 | 1.245090e-01 |
| block3.adaln | 5.303717e-01 | 4.839643e-01 |
| block2.cross_attn_q | 5.039087e-01 | 1.061436e-01 |
| block0.cross_attn_out_gate | 4.573546e-01 | 9.909683e-02 |
| block0.cross_attn_kv | 4.445839e-01 | 6.662389e-02 |
| block3.cross_attn_q | 3.649071e-01 | 7.700944e-02 |
| head_ll | 3.333516e-01 | 4.128463e-02 |
| block0.cross_attn_q | 2.381036e-01 | 5.034255e-02 |

## Aggregated Gradient Paths

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 6.446065e-01 | 4.384077e-02 |
| style_patch_proj | 1.511211e+00 | 7.223920e-02 |
| target_hf_fusion | 1.847092e+00 | 5.948820e-02 |
| target_hf_subband | 1.847092e+00 | 5.948820e-02 |
| input_time | 7.660547e+00 | 9.177730e-01 |
| self_attn | 3.521879e+00 | 1.887591e-01 |
| cross_attn_q | 9.251972e-01 | 9.731790e-02 |
| cross_attn_kv | 1.349486e+00 | 1.013025e-01 |
| cross_attn_out_gate | 3.750170e+00 | 4.026433e-01 |
| adaln | 2.361330e+00 | 9.487116e-01 |
| ffn | 5.504516e+00 | 2.567746e-01 |
| head_ll | 3.333516e-01 | 4.128463e-02 |
| head_hf | 1.403346e+00 | 9.497916e-02 |

## Input Tensor Gradients

| tensor | tensor rms | grad rms | grad/tensor |
|---|---:|---:|---:|
| content | 8.835873e-01 | 1.913516e-04 | 2.165622e-04 |
| target_style | 8.353144e-01 | 1.814733e-04 | 2.172515e-04 |

## Per-Loss Gradient Paths


### loss

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 7.246742e-01 | 4.928631e-02 |
| style_patch_proj | 1.585968e+00 | 7.581273e-02 |
| target_hf_fusion | 1.009316e+00 | 3.250644e-02 |
| target_hf_subband | 1.009316e+00 | 3.250644e-02 |
| input_time | 1.119723e+01 | 1.341486e+00 |
| self_attn | 3.700394e+00 | 1.983268e-01 |
| cross_attn_q | 9.702657e-01 | 1.020585e-01 |
| cross_attn_kv | 1.244306e+00 | 9.340686e-02 |
| cross_attn_out_gate | 3.956843e+00 | 4.248330e-01 |
| adaln | 2.181861e+00 | 8.766062e-01 |
| ffn | 5.777195e+00 | 2.694945e-01 |
| head_ll | 3.969961e-01 | 4.916682e-02 |
| head_hf | 1.640313e+00 | 1.110172e-01 |

### loss_fm_hf_total

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 5.562816e-02 | 3.783365e-03 |
| style_patch_proj | 1.470439e-01 | 7.029020e-03 |
| target_hf_fusion | 8.816951e-02 | 2.839623e-03 |
| target_hf_subband | 8.816951e-02 | 2.839623e-03 |
| input_time | 2.327142e+00 | 2.788036e-01 |
| self_attn | 5.621446e-01 | 3.012878e-02 |
| cross_attn_q | 1.506065e-01 | 1.584171e-02 |
| cross_attn_kv | 1.638306e-01 | 1.229835e-02 |
| cross_attn_out_gate | 3.259555e-01 | 3.499675e-02 |
| adaln | 4.937473e-01 | 1.983729e-01 |
| ffn | 8.408526e-01 | 3.922407e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 4.714635e-01 | 3.190889e-02 |

### loss_fm_spectral_ll

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 2.558905e-02 | 1.740354e-03 |
| style_patch_proj | 6.072818e-02 | 2.902940e-03 |
| target_hf_fusion | 0.000000e+00 | 0.000000e+00 |
| target_hf_subband | 0.000000e+00 | 0.000000e+00 |
| input_time | 3.502796e-01 | 4.196531e-02 |
| self_attn | 1.488070e-01 | 7.975476e-03 |
| cross_attn_q | 4.290533e-02 | 4.513045e-03 |
| cross_attn_kv | 6.420114e-02 | 4.819416e-03 |
| cross_attn_out_gate | 9.253730e-02 | 9.935421e-03 |
| adaln | 6.317833e-02 | 2.538316e-02 |
| ffn | 2.658063e-01 | 1.239933e-02 |
| head_ll | 3.969961e-01 | 4.916682e-02 |
| head_hf | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_lh

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 7.790464e-03 | 5.298425e-04 |
| style_patch_proj | 2.478930e-02 | 1.184983e-03 |
| target_hf_fusion | 2.709018e-02 | 8.724775e-04 |
| target_hf_subband | 2.709018e-02 | 8.724775e-04 |
| input_time | 3.906989e-01 | 4.680774e-02 |
| self_attn | 9.437315e-02 | 5.058035e-03 |
| cross_attn_q | 2.784290e-02 | 2.928687e-03 |
| cross_attn_kv | 3.068355e-02 | 2.303336e-03 |
| cross_attn_out_gate | 4.708438e-02 | 5.055292e-03 |
| adaln | 8.184378e-02 | 3.288237e-02 |
| ffn | 1.757714e-01 | 8.199380e-03 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 1.857787e-01 | 1.257360e-02 |

### loss_fm_spectral_hl

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 1.884312e-02 | 1.281552e-03 |
| style_patch_proj | 5.068494e-02 | 2.422851e-03 |
| target_hf_fusion | 2.475956e-02 | 7.974165e-04 |
| target_hf_subband | 2.475956e-02 | 7.974165e-04 |
| input_time | 7.636177e-01 | 9.148534e-02 |
| self_attn | 1.899023e-01 | 1.017803e-02 |
| cross_attn_q | 6.017426e-02 | 6.329497e-03 |
| cross_attn_kv | 6.152846e-02 | 4.618785e-03 |
| cross_attn_out_gate | 1.433838e-01 | 1.539464e-02 |
| adaln | 1.624017e-01 | 6.524814e-02 |
| ffn | 3.080825e-01 | 1.437143e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 2.676075e-01 | 1.811181e-02 |

### loss_fm_spectral_hh

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 3.387612e-02 | 2.303972e-03 |
| style_patch_proj | 8.197638e-02 | 3.918650e-03 |
| target_hf_fusion | 8.016826e-02 | 2.581932e-03 |
| target_hf_subband | 8.016826e-02 | 2.581932e-03 |
| input_time | 1.234599e+00 | 1.479113e-01 |
| self_attn | 3.024323e-01 | 1.620920e-02 |
| cross_attn_q | 7.468963e-02 | 7.856313e-03 |
| cross_attn_kv | 8.636827e-02 | 6.483447e-03 |
| cross_attn_out_gate | 1.680220e-01 | 1.803996e-02 |
| adaln | 2.546077e-01 | 1.022937e-01 |
| ffn | 4.473827e-01 | 2.086950e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 3.408083e-01 | 2.306608e-02 |

### loss_stat

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 7.701349e-01 | 5.237816e-02 |
| style_patch_proj | 1.684057e+00 | 8.050159e-02 |
| target_hf_fusion | 1.053550e+00 | 3.393106e-02 |
| target_hf_subband | 1.053550e+00 | 3.393106e-02 |
| input_time | 1.234358e+01 | 1.478824e+00 |
| self_attn | 3.903098e+00 | 2.091910e-01 |
| cross_attn_q | 1.059940e+00 | 1.114909e-01 |
| cross_attn_kv | 1.317465e+00 | 9.889876e-02 |
| cross_attn_out_gate | 4.138028e+00 | 4.442863e-01 |
| adaln | 2.415644e+00 | 9.705332e-01 |
| ffn | 5.991967e+00 | 2.795132e-01 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 1.425446e+00 | 9.647488e-02 |

### loss_stat_lh

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 2.674357e-01 | 1.818875e-02 |
| style_patch_proj | 5.775160e-01 | 2.760653e-02 |
| target_hf_fusion | 2.137334e-01 | 6.883584e-03 |
| target_hf_subband | 2.137334e-01 | 6.883584e-03 |
| input_time | 4.834897e+00 | 5.792456e-01 |
| self_attn | 1.409850e+00 | 7.556252e-02 |
| cross_attn_q | 3.936691e-01 | 4.140853e-02 |
| cross_attn_kv | 4.769376e-01 | 3.580250e-02 |
| cross_attn_out_gate | 1.370783e+00 | 1.471764e-01 |
| adaln | 9.476474e-01 | 3.807363e-01 |
| ffn | 2.255743e+00 | 1.052259e-01 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 8.114963e-01 | 5.492248e-02 |

### loss_stat_hl

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 2.492135e-01 | 1.694943e-02 |
| style_patch_proj | 5.644372e-01 | 2.698133e-02 |
| target_hf_fusion | 1.265239e-01 | 4.074880e-03 |
| target_hf_subband | 1.265239e-01 | 4.074880e-03 |
| input_time | 3.684267e+00 | 4.413942e-01 |
| self_attn | 1.281758e+00 | 6.869728e-02 |
| cross_attn_q | 3.560020e-01 | 3.744647e-02 |
| cross_attn_kv | 4.614459e-01 | 3.463957e-02 |
| cross_attn_out_gate | 1.368350e+00 | 1.469152e-01 |
| adaln | 7.880209e-01 | 3.166032e-01 |
| ffn | 2.095028e+00 | 9.772880e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 9.997726e-01 | 6.766511e-02 |

### loss_stat_hh

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 2.578541e-01 | 1.753709e-02 |
| style_patch_proj | 5.555103e-01 | 2.655461e-02 |
| target_hf_fusion | 1.023854e+00 | 3.297466e-02 |
| target_hf_subband | 1.023854e+00 | 3.297466e-02 |
| input_time | 4.077244e+00 | 4.884749e-01 |
| self_attn | 1.275550e+00 | 6.836454e-02 |
| cross_attn_q | 3.453535e-01 | 3.632640e-02 |
| cross_attn_kv | 4.180446e-01 | 3.138155e-02 |
| cross_attn_out_gate | 1.409195e+00 | 1.513006e-01 |
| adaln | 7.251353e-01 | 2.913376e-01 |
| ffn | 1.895955e+00 | 8.844246e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 6.114115e-01 | 4.138064e-02 |

## Loss Gradient Cosines

| pair / norm | value |
|---|---:|
| cos_fm_hf_vs_stat | -5.288508e-01 |
| cos_lh_mse_vs_stat | -4.158416e-01 |
| cos_hl_mse_vs_stat | -2.955350e-01 |
| cos_hh_mse_vs_stat | -5.206170e-01 |
| grad_norm_loss_fm_hf_total | 2.663295e+00 |
| grad_norm_loss_stat | 1.535991e+01 |
| grad_norm_loss_fm_spectral_lh | 4.895988e-01 |
| grad_norm_loss_stat_lh | 5.893168e+00 |
| grad_norm_loss_fm_spectral_hl | 9.207633e-01 |
| grad_norm_loss_stat_hl | 4.881380e+00 |
| grad_norm_loss_fm_spectral_hh | 1.428791e+00 |
| grad_norm_loss_stat_hh | 5.141553e+00 |

## Activation Gradient Probes

| module | act rms | grad rms | grad/act |
|---|---:|---:|---:|
| style_conditioner.patch_proj | 5.277607e-01 | 8.536387e-05 | 1.617473e-04 |
| target_latent_hf_subband_encoder_lh | 1.855279e-01 | 3.893152e-04 | 2.098418e-03 |
| target_latent_hf_subband_proj_lh | 1.065237e+00 | 1.327753e-04 | 1.246439e-04 |
| target_latent_hf_subband_encoder_hl | 1.667283e-01 | 5.840343e-04 | 3.502910e-03 |
| target_latent_hf_subband_proj_hl | 1.104973e+00 | 1.672391e-04 | 1.513513e-04 |
| target_latent_hf_subband_encoder_hh | 2.321403e-01 | 1.015909e-03 | 4.376271e-03 |
| target_latent_hf_subband_proj_hh | 9.680135e-01 | 5.308778e-04 | 5.484199e-04 |
| time_proj | 7.447503e-01 | 2.206678e-02 | 2.962977e-02 |
| input_proj | 3.386531e-01 | 1.274067e-04 | 3.762160e-04 |
| block0.sa_qkv | 5.601763e-01 | 1.588914e-05 | 2.836453e-05 |
| block0.ca_q | 2.338242e-01 | 1.151713e-05 | 4.925551e-05 |
| block0.ca_k | 3.589712e-01 | 3.812008e-05 | 1.061926e-04 |
| block0.ca_v | 3.428147e-01 | 3.976137e-05 | 1.159850e-04 |
| block0.ca_out | 5.605895e-01 | 6.419282e-06 | 1.145095e-05 |
| block0.ffn | 2.199440e-01 | 9.371806e-05 | 4.260995e-04 |
| block0.residual | 3.471585e-01 | 9.371806e-05 | 2.699575e-04 |
| block1.sa_qkv | 6.159652e-01 | 1.032962e-05 | 1.676981e-05 |
| block1.ca_q | 2.778876e-01 | 1.698689e-05 | 6.112864e-05 |
| block1.ca_k | 3.491578e-01 | 4.810205e-05 | 1.377659e-04 |
| block1.ca_v | 3.562052e-01 | 4.351453e-05 | 1.221614e-04 |
| block1.ca_out | 1.531011e+00 | 4.441870e-06 | 2.901266e-06 |
| block1.ffn | 2.522669e-01 | 6.474440e-05 | 2.566504e-04 |
| block1.residual | 4.316189e-01 | 6.474440e-05 | 1.500036e-04 |
| block2.sa_qkv | 6.217987e-01 | 5.090731e-06 | 8.187105e-06 |
| block2.ca_q | 3.257592e-01 | 2.154076e-05 | 6.612481e-05 |
| block2.ca_k | 3.705012e-01 | 5.065742e-05 | 1.367267e-04 |
| block2.ca_v | 3.606393e-01 | 4.437706e-05 | 1.230511e-04 |
| block2.ca_out | 2.399943e+00 | 3.616190e-06 | 1.506781e-06 |
| block2.ffn | 3.247727e-01 | 4.891510e-05 | 1.506133e-04 |
| block2.residual | 5.028707e-01 | 4.891510e-05 | 9.727172e-05 |
| block3.sa_qkv | 6.658805e-01 | 2.938266e-06 | 4.412603e-06 |
| block3.ca_q | 3.794986e-01 | 1.624704e-05 | 4.281186e-05 |
| block3.ca_k | 3.660560e-01 | 2.980870e-05 | 8.143205e-05 |
| block3.ca_v | 3.390979e-01 | 3.750650e-05 | 1.106067e-04 |
| block3.ca_out | 1.975404e+00 | 2.673309e-06 | 1.353297e-06 |
| block3.ffn | 3.840438e-01 | 3.854708e-05 | 1.003716e-04 |
| block3.residual | 6.879177e-01 | 3.854708e-05 | 5.603443e-05 |
| head_ll | 2.213666e-01 | 1.035855e-05 | 4.679364e-05 |
| head_lh | 3.607318e-01 | 1.377991e-04 | 3.819987e-04 |
| head_hl | 3.866892e-01 | 1.473474e-04 | 3.810486e-04 |
| head_hh | 2.022344e-01 | 1.929389e-04 | 9.540362e-04 |

## Cross-Attention Debug

| block | style gate | delta abs | ca in std | ca out std |
|---:|---:|---:|---:|---:|
| 0 | 0.056831 | 0.023530 | 0.348458 | 0.953963 |
| 1 | 0.057300 | 0.050756 | 0.387770 | 1.841687 |
| 2 | 0.061292 | 0.063693 | 0.473164 | 2.477815 |
| 3 | 0.058250 | 0.081789 | 0.551689 | 4.048417 |

## Style Condition Sensitivity


### style_id_only_fixed_target_latent

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 1.166751e-01 | 1.478928e-01 | 7.889169e-01 |
| lh | 7.392255e-02 | 1.819398e-01 | 4.063023e-01 |
| hl | 7.796397e-02 | 1.789453e-01 | 4.356860e-01 |
| hh | 7.126693e-02 | 1.283604e-01 | 5.552098e-01 |

### target_style_latent_only_fixed_id

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 0.000000e+00 | 1.478928e-01 | 0.000000e+00 |
| lh | 1.136771e-02 | 1.819385e-01 | 6.248102e-02 |
| hl | 1.240185e-02 | 1.789445e-01 | 6.930555e-02 |
| hh | 9.962787e-03 | 1.283596e-01 | 7.761622e-02 |

### style_id_and_target_latent

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 1.190361e-01 | 1.478928e-01 | 8.048809e-01 |
| lh | 8.079249e-02 | 1.819385e-01 | 4.440648e-01 |
| hl | 8.500229e-02 | 1.789445e-01 | 4.750203e-01 |
| hh | 7.908037e-02 | 1.283596e-01 | 6.160846e-01 |
