# Baseline Internal Flow Probe

Config: `configs\exp_probe_target_hf_subband_mixer_ft6.json`
Checkpoint: `exp\model_probe\target_hf_subband_mixer_ft6\epoch_0006.pt`
Device: `cuda`
Batches: 2, batch size: 4
Load info: `{'missing': 0, 'unexpected': 0}`

## Loss Components

| component | value | weighted value |
|---|---:|---:|
| loss | 3.048003 | 3.048003 |
| loss_fm_spectral_ll | 0.081376 | 0.024413 |
| loss_fm_spectral_lh | 0.452695 | 0.452695 |
| loss_fm_spectral_hl | 0.490530 | 0.490530 |
| loss_fm_spectral_hh | 0.413017 | 0.826034 |
| t_mean | 0.393818 | 0.393818 |
| flow | 1.793672 | 1.793672 |
| stat | 1.254331 | 1.254331 |
| fft | 0.000000 | 0.000000 |

## Parameter Gradient Groups

| group | grad norm | grad/param |
|---|---:|---:|
| input_proj | 5.918451e+00 | 1.366958e+00 |
| time_proj | 4.869773e+00 | 6.824119e-01 |
| block0.ffn | 3.725389e+00 | 3.603972e-01 |
| block1.ffn | 3.122589e+00 | 2.955208e-01 |
| block1.cross_attn_out_gate | 2.839843e+00 | 6.026823e-01 |
| block2.cross_attn_out_gate | 2.348447e+00 | 5.052830e-01 |
| block2.ffn | 2.225018e+00 | 2.053125e-01 |
| block1.self_attn | 2.210760e+00 | 2.363830e-01 |
| block0.self_attn | 2.208330e+00 | 2.376017e-01 |
| target_hf_subband_fusion | 1.848105e+00 | 5.951899e-02 |
| block0.adaln | 1.524850e+00 | 1.118310e+00 |
| style_conditioner.patch_proj | 1.511591e+00 | 7.225726e-02 |
| block1.adaln | 1.485054e+00 | 1.098474e+00 |
| block2.self_attn | 1.354077e+00 | 1.457811e-01 |
| block3.ffn | 1.319493e+00 | 1.186810e-01 |
| block3.self_attn | 9.000367e-01 | 9.595533e-02 |
| block1.cross_attn_kv | 8.932143e-01 | 1.342849e-01 |
| block2.adaln | 8.834130e-01 | 7.727383e-01 |
| head_lh | 8.436911e-01 | 9.750484e-02 |
| head_hh | 8.213646e-01 | 9.903139e-02 |
| head_hl | 7.590406e-01 | 8.785097e-02 |
| block3.cross_attn_kv | 6.471282e-01 | 9.702102e-02 |
| style_memory | 6.445086e-01 | 4.383397e-02 |
| block1.cross_attn_q | 6.436393e-01 | 1.341463e-01 |
| block2.cross_attn_kv | 6.427213e-01 | 9.667870e-02 |
| block3.cross_attn_out_gate | 5.912252e-01 | 1.270858e-01 |
| block3.adaln | 5.308439e-01 | 4.843953e-01 |
| block2.cross_attn_q | 5.048815e-01 | 1.063483e-01 |
| block0.cross_attn_out_gate | 4.574676e-01 | 9.912198e-02 |
| block0.cross_attn_kv | 4.453489e-01 | 6.673819e-02 |
| block3.cross_attn_q | 3.665571e-01 | 7.735700e-02 |
| head_ll | 3.333138e-01 | 4.127998e-02 |
| block0.cross_attn_q | 2.385755e-01 | 5.044240e-02 |

## Aggregated Gradient Paths

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 6.445086e-01 | 4.383397e-02 |
| style_patch_proj | 1.511591e+00 | 7.225726e-02 |
| target_hf_fusion | 1.848105e+00 | 5.951899e-02 |
| target_hf_subband | 1.848105e+00 | 5.951899e-02 |
| input_time | 7.664382e+00 | 9.182353e-01 |
| self_attn | 3.522466e+00 | 1.887953e-01 |
| cross_attn_q | 9.276094e-01 | 9.757149e-02 |
| cross_attn_kv | 1.352048e+00 | 1.014946e-01 |
| cross_attn_out_gate | 3.760150e+00 | 4.037163e-01 |
| adaln | 2.364903e+00 | 9.501625e-01 |
| ffn | 5.506438e+00 | 2.568632e-01 |
| head_ll | 3.333138e-01 | 4.127998e-02 |
| head_hf | 1.400927e+00 | 9.481483e-02 |

## Input Tensor Gradients

| tensor | tensor rms | grad rms | grad/tensor |
|---|---:|---:|---:|
| content | 8.835873e-01 | 1.913642e-04 | 2.165765e-04 |
| target_style | 8.353144e-01 | 1.814565e-04 | 2.172314e-04 |

## Per-Loss Gradient Paths


### loss

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 7.254686e-01 | 4.934018e-02 |
| style_patch_proj | 1.587883e+00 | 7.590421e-02 |
| target_hf_fusion | 1.010483e+00 | 3.254304e-02 |
| target_hf_subband | 1.010483e+00 | 3.254304e-02 |
| input_time | 1.118992e+01 | 1.340614e+00 |
| self_attn | 3.701154e+00 | 1.983725e-01 |
| cross_attn_q | 9.715915e-01 | 1.021978e-01 |
| cross_attn_kv | 1.246034e+00 | 9.353646e-02 |
| cross_attn_out_gate | 3.962025e+00 | 4.253910e-01 |
| adaln | 2.184020e+00 | 8.774882e-01 |
| ffn | 5.778829e+00 | 2.695697e-01 |
| head_ll | 3.974331e-01 | 4.922098e-02 |
| head_hf | 1.642898e+00 | 1.111914e-01 |

### loss_fm_hf_total

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 5.583204e-02 | 3.797219e-03 |
| style_patch_proj | 1.473943e-01 | 7.045764e-03 |
| target_hf_fusion | 8.832723e-02 | 2.844615e-03 |
| target_hf_subband | 8.832723e-02 | 2.844615e-03 |
| input_time | 2.327516e+00 | 2.788493e-01 |
| self_attn | 5.624750e-01 | 3.014724e-02 |
| cross_attn_q | 1.507804e-01 | 1.585998e-02 |
| cross_attn_kv | 1.640743e-01 | 1.231662e-02 |
| cross_attn_out_gate | 3.272772e-01 | 3.513879e-02 |
| adaln | 4.945142e-01 | 1.986842e-01 |
| ffn | 8.411743e-01 | 3.923893e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 4.730646e-01 | 3.201704e-02 |

### loss_fm_spectral_ll

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 2.557834e-02 | 1.739620e-03 |
| style_patch_proj | 6.071826e-02 | 2.902463e-03 |
| target_hf_fusion | 0.000000e+00 | 0.000000e+00 |
| target_hf_subband | 0.000000e+00 | 0.000000e+00 |
| input_time | 3.498753e-01 | 4.191699e-02 |
| self_attn | 1.487453e-01 | 7.972374e-03 |
| cross_attn_q | 4.290420e-02 | 4.512920e-03 |
| cross_attn_kv | 6.421115e-02 | 4.820162e-03 |
| cross_attn_out_gate | 9.263728e-02 | 9.946192e-03 |
| adaln | 6.326808e-02 | 2.541964e-02 |
| ffn | 2.656881e-01 | 1.239376e-02 |
| head_ll | 3.974331e-01 | 4.922098e-02 |
| head_hf | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_lh

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 7.830721e-03 | 5.325788e-04 |
| style_patch_proj | 2.494977e-02 | 1.192653e-03 |
| target_hf_fusion | 2.759902e-02 | 8.888380e-04 |
| target_hf_subband | 2.759902e-02 | 8.888380e-04 |
| input_time | 3.909430e-01 | 4.683713e-02 |
| self_attn | 9.455705e-02 | 5.068019e-03 |
| cross_attn_q | 2.795137e-02 | 2.940092e-03 |
| cross_attn_kv | 3.082737e-02 | 2.314129e-03 |
| cross_attn_out_gate | 4.779019e-02 | 5.131092e-03 |
| adaln | 8.205736e-02 | 3.296873e-02 |
| ffn | 1.761251e-01 | 8.215848e-03 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 1.878983e-01 | 1.271697e-02 |

### loss_fm_spectral_hl

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 1.886212e-02 | 1.282841e-03 |
| style_patch_proj | 5.070423e-02 | 2.423771e-03 |
| target_hf_fusion | 2.484762e-02 | 8.002279e-04 |
| target_hf_subband | 2.484762e-02 | 8.002279e-04 |
| input_time | 7.637025e-01 | 9.149578e-02 |
| self_attn | 1.899695e-01 | 1.018189e-02 |
| cross_attn_q | 6.011884e-02 | 6.323658e-03 |
| cross_attn_kv | 6.154673e-02 | 4.620151e-03 |
| cross_attn_out_gate | 1.434239e-01 | 1.539900e-02 |
| adaln | 1.626520e-01 | 6.534977e-02 |
| ffn | 3.081183e-01 | 1.437304e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 2.693002e-01 | 1.822625e-02 |

### loss_fm_spectral_hh

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 3.396181e-02 | 2.309793e-03 |
| style_patch_proj | 8.211938e-02 | 3.925482e-03 |
| target_hf_fusion | 8.014285e-02 | 2.581034e-03 |
| target_hf_subband | 8.014285e-02 | 2.581034e-03 |
| input_time | 1.234430e+00 | 1.478915e-01 |
| self_attn | 3.025232e-01 | 1.621448e-02 |
| cross_attn_q | 7.479442e-02 | 7.867324e-03 |
| cross_attn_kv | 8.646970e-02 | 6.491052e-03 |
| cross_attn_out_gate | 1.684594e-01 | 1.808700e-02 |
| adaln | 2.549231e-01 | 1.024221e-01 |
| ffn | 4.474682e-01 | 2.087341e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 3.405315e-01 | 2.304719e-02 |

### loss_stat

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 7.711306e-01 | 5.244572e-02 |
| style_patch_proj | 1.686292e+00 | 8.060835e-02 |
| target_hf_fusion | 1.054566e+00 | 3.396275e-02 |
| target_hf_subband | 1.054566e+00 | 3.396275e-02 |
| input_time | 1.233508e+01 | 1.477811e+00 |
| self_attn | 3.903899e+00 | 2.092391e-01 |
| cross_attn_q | 1.061239e+00 | 1.116274e-01 |
| cross_attn_kv | 1.319191e+00 | 9.902819e-02 |
| cross_attn_out_gate | 4.144111e+00 | 4.449410e-01 |
| adaln | 2.417591e+00 | 9.713315e-01 |
| ffn | 5.993644e+00 | 2.795903e-01 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 1.426469e+00 | 9.654346e-02 |

### loss_stat_lh

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 2.676941e-01 | 1.820627e-02 |
| style_patch_proj | 5.780922e-01 | 2.763405e-02 |
| target_hf_fusion | 2.138932e-01 | 6.888521e-03 |
| target_hf_subband | 2.138932e-01 | 6.888521e-03 |
| input_time | 4.831004e+00 | 5.787809e-01 |
| self_attn | 1.409864e+00 | 7.556517e-02 |
| cross_attn_q | 3.939810e-01 | 4.144127e-02 |
| cross_attn_kv | 4.773971e-01 | 3.583694e-02 |
| cross_attn_out_gate | 1.372303e+00 | 1.473401e-01 |
| adaln | 9.482215e-01 | 3.809732e-01 |
| ffn | 2.256045e+00 | 1.052395e-01 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 8.122467e-01 | 5.497290e-02 |

### loss_stat_hl

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 2.496117e-01 | 1.697646e-02 |
| style_patch_proj | 5.653348e-01 | 2.702422e-02 |
| target_hf_fusion | 1.264251e-01 | 4.071572e-03 |
| target_hf_subband | 1.264251e-01 | 4.071572e-03 |
| input_time | 3.682192e+00 | 4.411469e-01 |
| self_attn | 1.282288e+00 | 6.872739e-02 |
| cross_attn_q | 3.566479e-01 | 3.751435e-02 |
| cross_attn_kv | 4.622540e-01 | 3.470019e-02 |
| cross_attn_out_gate | 1.370637e+00 | 1.471613e-01 |
| adaln | 7.889584e-01 | 3.169850e-01 |
| ffn | 2.095815e+00 | 9.776514e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 1.001230e+00 | 6.776328e-02 |

### loss_stat_hh

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 2.582065e-01 | 1.756100e-02 |
| style_patch_proj | 5.563155e-01 | 2.659307e-02 |
| target_hf_fusion | 1.024871e+00 | 3.300640e-02 |
| target_hf_subband | 1.024871e+00 | 3.300640e-02 |
| input_time | 4.075598e+00 | 4.882792e-01 |
| self_attn | 1.275964e+00 | 6.838844e-02 |
| cross_attn_q | 3.457688e-01 | 3.637003e-02 |
| cross_attn_kv | 4.186299e-01 | 3.142544e-02 |
| cross_attn_out_gate | 1.411476e+00 | 1.515460e-01 |
| adaln | 7.258431e-01 | 2.916268e-01 |
| ffn | 1.896717e+00 | 8.847768e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 6.104153e-01 | 4.131294e-02 |

## Loss Gradient Cosines

| pair / norm | value |
|---|---:|
| cos_fm_hf_vs_stat | -5.283868e-01 |
| cos_lh_mse_vs_stat | -4.154449e-01 |
| cos_hl_mse_vs_stat | -2.949835e-01 |
| cos_hh_mse_vs_stat | -5.196169e-01 |
| grad_norm_loss_fm_hf_total | 2.664486e+00 |
| grad_norm_loss_stat | 1.535622e+01 |
| grad_norm_loss_fm_spectral_lh | 4.909388e-01 |
| grad_norm_loss_stat_lh | 5.890699e+00 |
| grad_norm_loss_fm_spectral_hl | 9.215633e-01 |
| grad_norm_loss_stat_hl | 4.881639e+00 |
| grad_norm_loss_fm_spectral_hh | 1.428785e+00 |
| grad_norm_loss_stat_hh | 5.141422e+00 |

## Activation Gradient Probes

| module | act rms | grad rms | grad/act |
|---|---:|---:|---:|
| style_conditioner.patch_proj | 5.278172e-01 | 8.543648e-05 | 1.618675e-04 |
| target_latent_hf_subband_encoder_lh | 1.851343e-01 | 3.914629e-04 | 2.114481e-03 |
| target_latent_hf_subband_proj_lh | 1.065431e+00 | 1.328236e-04 | 1.246666e-04 |
| target_latent_hf_subband_encoder_hl | 1.667131e-01 | 5.903771e-04 | 3.541276e-03 |
| target_latent_hf_subband_proj_hl | 1.104815e+00 | 1.682514e-04 | 1.522891e-04 |
| target_latent_hf_subband_encoder_hh | 2.320141e-01 | 1.016683e-03 | 4.381986e-03 |
| target_latent_hf_subband_proj_hh | 9.672511e-01 | 5.301579e-04 | 5.481079e-04 |
| target_latent_hf_subband_mixer | 1.067058e+00 | 1.328761e-04 | 1.245256e-04 |
| time_proj | 7.456972e-01 | 2.207643e-02 | 2.960509e-02 |
| input_proj | 3.386488e-01 | 1.274101e-04 | 3.762308e-04 |
| block0.sa_qkv | 5.600878e-01 | 1.589120e-05 | 2.837269e-05 |
| block0.ca_q | 2.338390e-01 | 1.153494e-05 | 4.932854e-05 |
| block0.ca_k | 3.590414e-01 | 3.815915e-05 | 1.062806e-04 |
| block0.ca_v | 3.428681e-01 | 3.981534e-05 | 1.161243e-04 |
| block0.ca_out | 5.614316e-01 | 6.419746e-06 | 1.143460e-05 |
| block0.ffn | 2.199535e-01 | 9.373372e-05 | 4.261524e-04 |
| block0.residual | 3.471774e-01 | 9.373372e-05 | 2.699880e-04 |
| block1.sa_qkv | 6.158404e-01 | 1.033148e-05 | 1.677622e-05 |
| block1.ca_q | 2.779032e-01 | 1.700476e-05 | 6.118952e-05 |
| block1.ca_k | 3.491792e-01 | 4.818032e-05 | 1.379816e-04 |
| block1.ca_v | 3.562178e-01 | 4.347766e-05 | 1.220536e-04 |
| block1.ca_out | 1.531524e+00 | 4.442061e-06 | 2.900419e-06 |
| block1.ffn | 2.522541e-01 | 6.475094e-05 | 2.566894e-04 |
| block1.residual | 4.316496e-01 | 6.475094e-05 | 1.500081e-04 |
| block2.sa_qkv | 6.218110e-01 | 5.090994e-06 | 8.187365e-06 |
| block2.ca_q | 3.258011e-01 | 2.155001e-05 | 6.614468e-05 |
| block2.ca_k | 3.705205e-01 | 5.077409e-05 | 1.370345e-04 |
| block2.ca_v | 3.606594e-01 | 4.438396e-05 | 1.230634e-04 |
| block2.ca_out | 2.401517e+00 | 3.616392e-06 | 1.505878e-06 |
| block2.ffn | 3.247679e-01 | 4.892097e-05 | 1.506336e-04 |
| block2.residual | 5.028467e-01 | 4.892097e-05 | 9.728804e-05 |
| block3.sa_qkv | 6.659981e-01 | 2.937638e-06 | 4.410880e-06 |
| block3.ca_q | 3.795824e-01 | 1.626999e-05 | 4.286286e-05 |
| block3.ca_k | 3.661414e-01 | 2.986497e-05 | 8.156677e-05 |
| block3.ca_v | 3.391692e-01 | 3.758496e-05 | 1.108148e-04 |
| block3.ca_out | 1.979605e+00 | 2.672602e-06 | 1.350068e-06 |
| block3.ffn | 3.840921e-01 | 3.854855e-05 | 1.003628e-04 |
| block3.residual | 6.878993e-01 | 3.854855e-05 | 5.603807e-05 |
| head_ll | 2.214580e-01 | 1.036026e-05 | 4.678203e-05 |
| head_lh | 3.607630e-01 | 1.377824e-04 | 3.819194e-04 |
| head_hl | 3.868075e-01 | 1.473339e-04 | 3.808973e-04 |
| head_hh | 2.022677e-01 | 1.929262e-04 | 9.538164e-04 |

## Cross-Attention Debug

| block | style gate | delta abs | ca in std | ca out std |
|---:|---:|---:|---:|---:|
| 0 | 0.056828 | 0.023560 | 0.348473 | 0.954993 |
| 1 | 0.057299 | 0.050773 | 0.387810 | 1.842295 |
| 2 | 0.061289 | 0.063713 | 0.473204 | 2.479366 |
| 3 | 0.058237 | 0.081969 | 0.551715 | 4.059523 |

## Style Condition Sensitivity


### style_id_only_fixed_target_latent

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 1.167324e-01 | 1.479142e-01 | 7.891901e-01 |
| lh | 7.403953e-02 | 1.819872e-01 | 4.068391e-01 |
| hl | 7.807089e-02 | 1.790680e-01 | 4.359845e-01 |
| hh | 7.137478e-02 | 1.284452e-01 | 5.556829e-01 |

### target_style_latent_only_fixed_id

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 0.000000e+00 | 1.479141e-01 | 0.000000e+00 |
| lh | 1.140410e-02 | 1.819877e-01 | 6.266416e-02 |
| hl | 1.239786e-02 | 1.790683e-01 | 6.923533e-02 |
| hh | 9.972768e-03 | 1.284458e-01 | 7.764186e-02 |

### style_id_and_target_latent

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 1.191056e-01 | 1.479141e-01 | 8.052346e-01 |
| lh | 8.091870e-02 | 1.819877e-01 | 4.446384e-01 |
| hl | 8.511057e-02 | 1.790683e-01 | 4.752966e-01 |
| hh | 7.918494e-02 | 1.284458e-01 | 6.164855e-01 |
