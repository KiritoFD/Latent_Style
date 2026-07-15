# Baseline Internal Flow Probe

Config: `SchrodingerBridge\configs\exp_probe_target_latent_fusion.json`
Checkpoint: `SchrodingerBridge\exp\dino_s_break\brk_a_ll03_10ep\epoch_0010.pt`
Device: `cuda`
Batches: 1, batch size: 4
Load info: `{'missing': 11, 'unexpected': 0}`

## Loss Components

| component | value | weighted value |
|---|---:|---:|
| loss | 0.945684 | 0.945684 |
| loss_fm_spectral_ll | 0.145030 | 0.043509 |
| loss_fm_spectral_lh | 0.417655 | 0.417655 |
| loss_fm_spectral_hl | 0.484520 | 0.484520 |
| loss_fm_spectral_hh | 0.000000 | 0.000000 |
| t_mean | 0.265435 | 0.265435 |
| flow | 0.945684 | 0.945684 |
| stat | 0.000000 | 0.000000 |
| fft | 0.000000 | 0.000000 |

## Parameter Gradient Groups

| group | grad norm | grad/param |
|---|---:|---:|
| time_proj | 1.184587e+00 | 1.665930e-01 |
| input_proj | 1.016132e+00 | 2.304367e-01 |
| block0.ffn | 4.414387e-01 | 4.249811e-02 |
| head_ll | 3.960846e-01 | 4.913238e-02 |
| head_hl | 3.068761e-01 | 3.615566e-02 |
| block1.ffn | 2.934472e-01 | 2.781729e-02 |
| block0.self_attn | 2.811652e-01 | 3.027347e-02 |
| block2.ffn | 2.576238e-01 | 2.397372e-02 |
| head_lh | 2.452269e-01 | 2.888231e-02 |
| block0.adaln | 2.092627e-01 | 1.660281e-01 |
| block1.self_attn | 1.983575e-01 | 2.122669e-02 |
| block1.adaln | 1.691161e-01 | 1.308107e-01 |
| block3.ffn | 1.541604e-01 | 1.412651e-02 |
| block0.cross_attn_out_gate | 1.514754e-01 | 3.286841e-02 |
| block2.self_attn | 1.497895e-01 | 1.614881e-02 |
| block3.self_attn | 9.102897e-02 | 9.719733e-03 |
| style_conditioner.patch_proj | 9.010669e-02 | 4.309474e-03 |
| block2.cross_attn_out_gate | 8.354621e-02 | 1.799344e-02 |
| block2.adaln | 8.035724e-02 | 7.297920e-02 |
| block3.cross_attn_kv | 5.769939e-02 | 8.661691e-03 |
| block1.cross_attn_out_gate | 5.697143e-02 | 1.211048e-02 |
| block0.cross_attn_kv | 5.134972e-02 | 7.707816e-03 |
| block3.cross_attn_out_gate | 4.671395e-02 | 1.005226e-02 |
| block3.adaln | 4.643848e-02 | 4.557336e-02 |
| block2.cross_attn_kv | 4.615726e-02 | 6.948247e-03 |
| block2.cross_attn_q | 4.047632e-02 | 8.539802e-03 |
| style_memory | 4.028628e-02 | 2.765563e-03 |
| block3.cross_attn_q | 3.324832e-02 | 7.025054e-03 |
| block1.cross_attn_kv | 3.322615e-02 | 5.006934e-03 |
| block0.cross_attn_q | 3.275003e-02 | 6.928940e-03 |
| target_latent_token_fusion | 2.120672e-02 | 1.530847e-03 |
| block1.cross_attn_q | 1.949295e-02 | 4.070377e-03 |

## Aggregated Gradient Paths

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 4.028628e-02 | 2.765563e-03 |
| style_patch_proj | 9.010669e-02 | 4.309474e-03 |
| target_latent_fusion | 2.120672e-02 | 1.530847e-03 |
| input_time | 1.560695e+00 | 1.865307e-01 |
| self_attn | 3.861641e-01 | 2.072058e-02 |
| cross_attn_q | 6.477903e-02 | 6.823033e-03 |
| cross_attn_kv | 9.591915e-02 | 7.211274e-03 |
| cross_attn_out_gate | 1.880231e-01 | 2.021361e-02 |
| adaln | 2.846137e-01 | 1.212410e-01 |
| ffn | 6.091920e-01 | 2.859876e-02 |
| head_ll | 3.960846e-01 | 4.913238e-02 |
| head_hf | 3.928221e-01 | 3.272046e-02 |

## Input Tensor Gradients

| tensor | tensor rms | grad rms | grad/tensor |
|---|---:|---:|---:|
| content | 9.508316e-01 | 5.110133e-05 | 5.374383e-05 |
| target_style | 9.403301e-01 | 5.807941e-05 | 6.176491e-05 |

## Per-Loss Gradient Paths


### loss

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 4.860931e-02 | 3.336920e-03 |
| style_patch_proj | 1.207323e-01 | 5.774185e-03 |
| target_latent_fusion | 2.910792e-02 | 2.101209e-03 |
| input_time | 2.587230e+00 | 3.092198e-01 |
| self_attn | 6.328375e-01 | 3.395644e-02 |
| cross_attn_q | 9.641588e-02 | 1.015527e-02 |
| cross_attn_kv | 1.328142e-01 | 9.985072e-03 |
| cross_attn_out_gate | 2.475921e-01 | 2.661764e-02 |
| adaln | 4.841869e-01 | 2.062561e-01 |
| ffn | 9.330019e-01 | 4.380015e-02 |
| head_ll | 5.722671e-01 | 7.098698e-02 |
| head_hf | 3.963697e-01 | 3.301596e-02 |

### loss_fm_spectral_ll

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 3.677954e-02 | 2.524833e-03 |
| style_patch_proj | 7.800810e-02 | 3.730843e-03 |
| target_latent_fusion | 2.194043e-02 | 1.583811e-03 |
| input_time | 5.180534e-01 | 6.191655e-02 |
| self_attn | 2.032760e-01 | 1.090727e-02 |
| cross_attn_q | 5.125599e-02 | 5.398680e-03 |
| cross_attn_kv | 8.053807e-02 | 6.054913e-03 |
| cross_attn_out_gate | 1.082323e-01 | 1.163562e-02 |
| adaln | 1.001631e-01 | 4.266790e-02 |
| ffn | 3.710965e-01 | 1.742127e-02 |
| head_ll | 5.722671e-01 | 7.098698e-02 |
| head_hf | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_lh

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 1.316719e-02 | 9.038980e-04 |
| style_patch_proj | 4.065169e-02 | 1.944222e-03 |
| target_latent_fusion | 8.597719e-03 | 6.206423e-04 |
| input_time | 1.088303e+00 | 1.300714e-01 |
| self_attn | 2.579927e-01 | 1.384323e-02 |
| cross_attn_q | 4.262664e-02 | 4.489770e-03 |
| cross_attn_kv | 5.126250e-02 | 3.853953e-03 |
| cross_attn_out_gate | 1.038976e-01 | 1.116961e-02 |
| adaln | 2.087880e-01 | 8.894042e-02 |
| ffn | 3.803339e-01 | 1.785493e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 2.182654e-01 | 1.818061e-02 |

### loss_fm_spectral_hl

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 2.098764e-02 | 1.440755e-03 |
| style_patch_proj | 5.630981e-02 | 2.693093e-03 |
| target_latent_fusion | 9.169798e-03 | 6.619389e-04 |
| input_time | 1.363433e+00 | 1.629543e-01 |
| self_attn | 3.302404e-01 | 1.771985e-02 |
| cross_attn_q | 5.325752e-02 | 5.609497e-03 |
| cross_attn_kv | 6.809356e-02 | 5.119325e-03 |
| cross_attn_out_gate | 1.092839e-01 | 1.174868e-02 |
| adaln | 2.577532e-01 | 1.097988e-01 |
| ffn | 4.861640e-01 | 2.282316e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 3.308612e-01 | 2.755938e-02 |

### loss_fm_spectral_hh

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 0.000000e+00 | 0.000000e+00 |
| style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| target_latent_fusion | 0.000000e+00 | 0.000000e+00 |
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
| style_conditioner.patch_proj | 5.144182e-01 | 9.195831e-06 | 1.787618e-05 |
| target_latent_tokenizer | 3.492821e-01 | 2.666251e-07 | 7.633516e-07 |
| target_latent_token_proj | 5.767168e-01 | 4.594088e-07 | 7.965933e-07 |
| time_proj | 7.145193e-01 | 3.165554e-03 | 4.430327e-03 |
| input_proj | 3.964421e-01 | 2.853236e-05 | 7.197105e-05 |
| block0.sa_qkv | 5.619131e-01 | 2.087433e-06 | 3.714868e-06 |
| block0.ca_q | 2.719931e-01 | 3.016557e-06 | 1.109056e-05 |
| block0.ca_k | 3.491163e-01 | 4.551673e-06 | 1.303770e-05 |
| block0.ca_v | 3.397054e-01 | 4.546381e-06 | 1.338331e-05 |
| block0.ca_out | 7.737971e-01 | 1.566595e-06 | 2.024555e-06 |
| block0.ffn | 2.165426e-01 | 2.475199e-05 | 1.143054e-04 |
| block0.residual | 3.791573e-01 | 2.475199e-05 | 6.528160e-05 |
| block1.sa_qkv | 6.079376e-01 | 1.430077e-06 | 2.352342e-06 |
| block1.ca_q | 2.722041e-01 | 3.714807e-06 | 1.364714e-05 |
| block1.ca_k | 3.337414e-01 | 2.927966e-06 | 8.773157e-06 |
| block1.ca_v | 3.419060e-01 | 3.503999e-06 | 1.024843e-05 |
| block1.ca_out | 1.335323e+00 | 1.348882e-06 | 1.010154e-06 |
| block1.ffn | 2.338771e-01 | 2.118856e-05 | 9.059698e-05 |
| block1.residual | 4.288402e-01 | 2.118856e-05 | 4.940898e-05 |
| block2.sa_qkv | 6.105841e-01 | 9.282861e-07 | 1.520325e-06 |
| block2.ca_q | 3.113034e-01 | 5.985058e-06 | 1.922580e-05 |
| block2.ca_k | 3.676789e-01 | 4.026418e-06 | 1.095091e-05 |
| block2.ca_v | 3.511659e-01 | 4.436959e-06 | 1.263494e-05 |
| block2.ca_out | 2.072365e+00 | 1.260538e-06 | 6.082607e-07 |
| block2.ffn | 2.995347e-01 | 1.750179e-05 | 5.842993e-05 |
| block2.residual | 4.966658e-01 | 1.750179e-05 | 3.523857e-05 |
| block3.sa_qkv | 6.318912e-01 | 5.813347e-07 | 9.199918e-07 |
| block3.ca_q | 3.442760e-01 | 3.823612e-06 | 1.110624e-05 |
| block3.ca_k | 3.609573e-01 | 4.564430e-06 | 1.264535e-05 |
| block3.ca_v | 3.267025e-01 | 6.562099e-06 | 2.008585e-05 |
| block3.ca_out | 9.876165e-01 | 1.012023e-06 | 1.024713e-06 |
| block3.ffn | 3.620027e-01 | 1.414543e-05 | 3.907548e-05 |
| block3.residual | 6.317467e-01 | 1.414543e-05 | 2.239098e-05 |
| head_ll | 1.409198e-01 | 1.394632e-05 | 9.896641e-05 |
| head_lh | 3.564698e-01 | 7.888943e-05 | 2.213075e-04 |
| head_hl | 3.804089e-01 | 8.497007e-05 | 2.233651e-04 |

## Cross-Attention Debug

| block | style gate | delta abs | ca in std | ca out std |
|---:|---:|---:|---:|---:|
| 0 | 0.056254 | 0.025895 | 0.404460 | 1.212356 |
| 1 | 0.056904 | 0.047318 | 0.414920 | 1.969060 |
| 2 | 0.060976 | 0.079426 | 0.466779 | 3.303508 |
| 3 | 0.058741 | 0.033105 | 0.529767 | 1.735703 |

## Style Condition Sensitivity


### style_id_only_fixed_target_latent

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 1.241906e-01 | 2.205411e-01 | 5.631178e-01 |
| lh | 6.492690e-02 | 2.301952e-01 | 2.820515e-01 |
| hl | 7.291040e-02 | 2.745202e-01 | 2.655921e-01 |

### target_style_latent_only_fixed_id

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 4.810931e-03 | 2.205409e-01 | 2.181424e-02 |
| lh | 4.350635e-03 | 2.301952e-01 | 1.889976e-02 |
| hl | 5.216731e-03 | 2.745200e-01 | 1.900310e-02 |

### style_id_and_target_latent

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 1.287806e-01 | 2.205409e-01 | 5.839310e-01 |
| lh | 6.522969e-02 | 2.301952e-01 | 2.833669e-01 |
| hl | 7.238684e-02 | 2.745200e-01 | 2.636851e-01 |
