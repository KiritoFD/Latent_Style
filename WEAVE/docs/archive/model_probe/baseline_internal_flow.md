# Baseline Internal Flow Probe

Config: `G:\GitHub\Latent_Style\SchrodingerBridge\configs\exp_brk_a_ll03_10ep.json`
Checkpoint: `G:\GitHub\Latent_Style\SchrodingerBridge\exp\dino_s_break\brk_a_ll03_10ep\epoch_0010.pt`
Device: `cuda`
Batches: 1, batch size: 4
Load info: `{'missing': 0, 'unexpected': 0}`

## Loss Components

| component | value | weighted value |
|---|---:|---:|
| loss | 0.946128 | 0.946128 |
| loss_fm_spectral_ll | 0.147822 | 0.044347 |
| loss_fm_spectral_lh | 0.417522 | 0.417522 |
| loss_fm_spectral_hl | 0.484260 | 0.484260 |
| loss_fm_spectral_hh | 0.000000 | 0.000000 |
| t_mean | 0.265435 | 0.265435 |
| flow | 0.946128 | 0.946128 |
| stat | 0.000000 | 0.000000 |
| fft | 0.000000 | 0.000000 |

## Parameter Gradient Groups

| group | grad norm | grad/param |
|---|---:|---:|
| time_proj | 1.190221e+00 | 1.673853e-01 |
| input_proj | 1.015692e+00 | 2.303369e-01 |
| block0.ffn | 4.420171e-01 | 4.255379e-02 |
| head_ll | 3.986451e-01 | 4.945000e-02 |
| head_hl | 3.127963e-01 | 3.685317e-02 |
| block1.ffn | 2.945558e-01 | 2.792238e-02 |
| block0.self_attn | 2.809640e-01 | 3.025181e-02 |
| block2.ffn | 2.614713e-01 | 2.433176e-02 |
| head_lh | 2.471632e-01 | 2.911037e-02 |
| block0.adaln | 2.098177e-01 | 1.664685e-01 |
| block1.self_attn | 1.984725e-01 | 2.123899e-02 |
| block1.adaln | 1.696622e-01 | 1.312331e-01 |
| block3.ffn | 1.562528e-01 | 1.431825e-02 |
| block2.self_attn | 1.502760e-01 | 1.620126e-02 |
| block0.cross_attn_out_gate | 1.458095e-01 | 3.163897e-02 |
| block3.self_attn | 9.208010e-02 | 9.831968e-03 |
| style_conditioner.patch_proj | 8.993658e-02 | 4.301339e-03 |
| block2.cross_attn_out_gate | 8.784519e-02 | 1.891932e-02 |
| block2.adaln | 8.075834e-02 | 7.334348e-02 |
| block3.cross_attn_kv | 5.912005e-02 | 8.874958e-03 |
| block1.cross_attn_out_gate | 5.346900e-02 | 1.136597e-02 |
| block0.cross_attn_kv | 5.030408e-02 | 7.550861e-03 |
| block3.cross_attn_out_gate | 4.733341e-02 | 1.018556e-02 |
| block3.adaln | 4.683593e-02 | 4.596341e-02 |
| block2.cross_attn_kv | 4.523447e-02 | 6.809335e-03 |
| style_memory | 4.047479e-02 | 2.778504e-03 |
| block2.cross_attn_q | 3.969207e-02 | 8.374338e-03 |
| block3.cross_attn_q | 3.376802e-02 | 7.134860e-03 |
| block1.cross_attn_kv | 3.282115e-02 | 4.945903e-03 |
| block0.cross_attn_q | 3.154301e-02 | 6.673571e-03 |
| block1.cross_attn_q | 1.905485e-02 | 3.978897e-03 |

## Aggregated Gradient Paths

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 4.047479e-02 | 2.778504e-03 |
| style_patch_proj | 8.993658e-02 | 4.301339e-03 |
| input_time | 1.564690e+00 | 1.870081e-01 |
| self_attn | 3.865148e-01 | 2.073940e-02 |
| cross_attn_q | 6.382624e-02 | 6.722677e-03 |
| cross_attn_kv | 9.565075e-02 | 7.191096e-03 |
| cross_attn_out_gate | 1.845984e-01 | 1.984543e-02 |
| adaln | 2.855245e-01 | 1.216290e-01 |
| ffn | 6.123107e-01 | 2.874517e-02 |
| head_ll | 3.986451e-01 | 4.945000e-02 |
| head_hf | 3.986618e-01 | 3.320688e-02 |

## Input Tensor Gradients

| tensor | tensor rms | grad rms | grad/tensor |
|---|---:|---:|---:|
| content | 9.508316e-01 | 5.109639e-05 | 5.373863e-05 |
| target_style | 9.403301e-01 | 5.806494e-05 | 6.174953e-05 |

## Per-Loss Gradient Paths


### loss

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 4.803721e-02 | 3.297647e-03 |
| style_patch_proj | 1.206649e-01 | 5.770963e-03 |
| input_time | 2.646636e+00 | 3.163198e-01 |
| self_attn | 6.445328e-01 | 3.458398e-02 |
| cross_attn_q | 9.733457e-02 | 1.025204e-02 |
| cross_attn_kv | 1.334463e-01 | 1.003259e-02 |
| cross_attn_out_gate | 2.491615e-01 | 2.678636e-02 |
| adaln | 4.953378e-01 | 2.110062e-01 |
| ffn | 9.490923e-01 | 4.455552e-02 |
| head_ll | 5.743970e-01 | 7.125118e-02 |
| head_hf | 3.974465e-01 | 3.310566e-02 |

### loss_fm_spectral_ll

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 3.672316e-02 | 2.520963e-03 |
| style_patch_proj | 7.716475e-02 | 3.690509e-03 |
| input_time | 5.075443e-01 | 6.066053e-02 |
| self_attn | 2.024639e-01 | 1.086369e-02 |
| cross_attn_q | 4.977741e-02 | 5.242945e-03 |
| cross_attn_kv | 7.938814e-02 | 5.968461e-03 |
| cross_attn_out_gate | 1.040843e-01 | 1.118968e-02 |
| adaln | 9.993975e-02 | 4.257277e-02 |
| ffn | 3.709507e-01 | 1.741443e-02 |
| head_ll | 5.743970e-01 | 7.125118e-02 |
| head_hf | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_lh

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 1.349530e-02 | 9.264223e-04 |
| style_patch_proj | 4.184414e-02 | 2.001253e-03 |
| input_time | 1.117495e+00 | 1.335605e-01 |
| self_attn | 2.642134e-01 | 1.417702e-02 |
| cross_attn_q | 4.393582e-02 | 4.627664e-03 |
| cross_attn_kv | 5.273279e-02 | 3.964491e-03 |
| cross_attn_out_gate | 1.059399e-01 | 1.138917e-02 |
| adaln | 2.139328e-01 | 9.113204e-02 |
| ffn | 3.893122e-01 | 1.827642e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 2.156088e-01 | 1.795933e-02 |

### loss_fm_spectral_hl

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 2.071954e-02 | 1.422350e-03 |
| style_patch_proj | 5.631711e-02 | 2.693442e-03 |
| input_time | 1.395607e+00 | 1.667998e-01 |
| self_attn | 3.367112e-01 | 1.806706e-02 |
| cross_attn_q | 5.402139e-02 | 5.689954e-03 |
| cross_attn_kv | 6.857834e-02 | 5.155772e-03 |
| cross_attn_out_gate | 1.104415e-01 | 1.187313e-02 |
| adaln | 2.634052e-01 | 1.122065e-01 |
| ffn | 4.948877e-01 | 2.323270e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 3.338811e-01 | 2.781092e-02 |

### loss_fm_spectral_hh

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 0.000000e+00 | 0.000000e+00 |
| style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| input_time | 0.000000e+00 | 0.000000e+00 |
| self_attn | 0.000000e+00 | 0.000000e+00 |
| cross_attn_q | 0.000000e+00 | 0.000000e+00 |
| cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| adaln | 0.000000e+00 | 0.000000e+00 |
| ffn | 0.000000e+00 | 0.000000e+00 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 0.000000e+00 | 0.000000e+00 |

## Activation Gradient Probes

| module | act rms | grad rms | grad/act |
|---|---:|---:|---:|
| style_conditioner.patch_proj | 5.144182e-01 | 9.209109e-06 | 1.790199e-05 |
| time_proj | 7.145193e-01 | 3.175260e-03 | 4.443911e-03 |
| input_proj | 3.964421e-01 | 2.858764e-05 | 7.211051e-05 |
| block0.sa_qkv | 5.619131e-01 | 2.093173e-06 | 3.725084e-06 |
| block0.ca_q | 2.719931e-01 | 2.936762e-06 | 1.079719e-05 |
| block0.ca_k | 3.483981e-01 | 4.512216e-06 | 1.295132e-05 |
| block0.ca_v | 3.393265e-01 | 4.513120e-06 | 1.330023e-05 |
| block0.ca_out | 7.646620e-01 | 1.570276e-06 | 2.053556e-06 |
| block0.ffn | 2.165377e-01 | 2.479703e-05 | 1.145160e-04 |
| block0.residual | 3.791066e-01 | 2.479703e-05 | 6.540911e-05 |
| block1.sa_qkv | 6.077619e-01 | 1.434783e-06 | 2.360765e-06 |
| block1.ca_q | 2.721366e-01 | 3.710263e-06 | 1.363383e-05 |
| block1.ca_k | 3.334200e-01 | 2.923581e-06 | 8.768462e-06 |
| block1.ca_v | 3.409174e-01 | 3.492597e-06 | 1.024470e-05 |
| block1.ca_out | 1.342214e+00 | 1.350942e-06 | 1.006503e-06 |
| block1.ffn | 2.340086e-01 | 2.121973e-05 | 9.067928e-05 |
| block1.residual | 4.287855e-01 | 2.121973e-05 | 4.948799e-05 |
| block2.sa_qkv | 6.106682e-01 | 9.336455e-07 | 1.528891e-06 |
| block2.ca_q | 3.112479e-01 | 5.983275e-06 | 1.922351e-05 |
| block2.ca_k | 3.667878e-01 | 3.964465e-06 | 1.080861e-05 |
| block2.ca_v | 3.516046e-01 | 4.392936e-06 | 1.249397e-05 |
| block2.ca_out | 2.073926e+00 | 1.262484e-06 | 6.087411e-07 |
| block2.ffn | 2.998257e-01 | 1.752505e-05 | 5.845077e-05 |
| block2.residual | 4.960403e-01 | 1.752505e-05 | 3.532988e-05 |
| block3.sa_qkv | 6.314420e-01 | 5.877455e-07 | 9.307988e-07 |
| block3.ca_q | 3.435922e-01 | 3.815872e-06 | 1.110582e-05 |
| block3.ca_k | 3.608010e-01 | 4.659658e-06 | 1.291476e-05 |
| block3.ca_v | 3.263640e-01 | 6.750146e-06 | 2.068287e-05 |
| block3.ca_out | 9.924486e-01 | 1.013671e-06 | 1.021384e-06 |
| block3.ffn | 3.616794e-01 | 1.416344e-05 | 3.916021e-05 |
| block3.residual | 6.306133e-01 | 1.416344e-05 | 2.245979e-05 |
| head_ll | 1.402982e-01 | 1.407993e-05 | 1.003572e-04 |
| head_lh | 3.569876e-01 | 7.887684e-05 | 2.209512e-04 |
| head_hl | 3.814369e-01 | 8.494723e-05 | 2.227032e-04 |

## Cross-Attention Debug

| block | style gate | delta abs | ca in std | ca out std |
|---:|---:|---:|---:|---:|
| 0 | 0.056254 | 0.025671 | 0.404460 | 1.201323 |
| 1 | 0.056904 | 0.047523 | 0.414803 | 1.966465 |
| 2 | 0.060976 | 0.079344 | 0.466686 | 3.307167 |
| 3 | 0.058741 | 0.033143 | 0.528898 | 1.739431 |

## Style Condition Sensitivity


### style_id_only_fixed_target_latent

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 1.227813e-01 | 2.189233e-01 | 5.608417e-01 |
| lh | 6.353138e-02 | 2.299886e-01 | 2.762370e-01 |
| hl | 7.131185e-02 | 2.743799e-01 | 2.599019e-01 |

### target_style_latent_only_fixed_id

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 0.000000e+00 | 2.189235e-01 | 0.000000e+00 |
| lh | 0.000000e+00 | 2.299884e-01 | 0.000000e+00 |
| hl | 0.000000e+00 | 2.743798e-01 | 0.000000e+00 |

### style_id_and_target_latent

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 1.279761e-01 | 2.189235e-01 | 5.845701e-01 |
| lh | 6.695493e-02 | 2.299884e-01 | 2.911230e-01 |
| hl | 7.464990e-02 | 2.743798e-01 | 2.720677e-01 |
