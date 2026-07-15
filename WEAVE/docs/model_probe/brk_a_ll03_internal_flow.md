# Baseline Internal Flow Probe

Config: `configs\exp_brk_a_ll03_10ep.json`
Checkpoint: `exp\dino_s_break\brk_a_ll03_10ep\epoch_0010.pt`
Device: `cuda`
Batches: 3, batch size: 4
Load info: `{'missing': 0, 'unexpected': 0}`

## Loss Components

| component | value | weighted value |
|---|---:|---:|
| loss | 1.028419 | 1.028419 |
| loss_fm_spectral_ll | 0.104159 | 0.031248 |
| loss_fm_spectral_lh | 0.467689 | 0.467689 |
| loss_fm_spectral_hl | 0.529482 | 0.529482 |
| loss_fm_spectral_hh | 0.000000 | 0.000000 |
| t_mean | 0.402803 | 0.402803 |
| flow | 1.028419 | 1.028419 |
| stat | 0.000000 | 0.000000 |
| fft | 0.000000 | 0.000000 |

## Parameter Gradient Groups

| group | grad norm | grad/param |
|---|---:|---:|
| input_proj | 6.750467e-01 | 1.530860e-01 |
| head_ll | 4.168625e-01 | 5.170978e-02 |
| head_hl | 3.798499e-01 | 4.475332e-02 |
| block0.ffn | 2.125662e-01 | 2.046413e-02 |
| head_lh | 1.859748e-01 | 2.190372e-02 |
| block1.ffn | 1.824256e-01 | 1.729302e-02 |
| block2.ffn | 1.701848e-01 | 1.583690e-02 |
| block3.ffn | 1.397030e-01 | 1.280170e-02 |
| block0.cross_attn_out_gate | 1.349073e-01 | 2.927332e-02 |
| block0.self_attn | 1.264698e-01 | 1.361719e-02 |
| block1.self_attn | 1.033522e-01 | 1.105996e-02 |
| block2.self_attn | 8.020894e-02 | 8.647327e-03 |
| block3.cross_attn_out_gate | 8.001932e-02 | 1.721916e-02 |
| block3.self_attn | 6.965110e-02 | 7.437084e-03 |
| style_conditioner.patch_proj | 6.273118e-02 | 3.000204e-03 |
| time_proj | 5.241100e-02 | 7.370759e-03 |
| block0.cross_attn_kv | 5.077380e-02 | 7.621369e-03 |
| block0.adaln | 5.044435e-02 | 4.002234e-02 |
| block0.cross_attn_q | 4.327467e-02 | 9.155644e-03 |
| block1.adaln | 3.838767e-02 | 2.969272e-02 |
| block3.cross_attn_kv | 3.683288e-02 | 5.529262e-03 |
| block2.cross_attn_kv | 3.528586e-02 | 5.311729e-03 |
| block2.cross_attn_q | 3.020599e-02 | 6.372940e-03 |
| block2.cross_attn_out_gate | 2.966848e-02 | 6.389736e-03 |
| style_memory | 2.950728e-02 | 2.025609e-03 |
| block3.cross_attn_q | 2.788587e-02 | 5.892019e-03 |
| block2.adaln | 2.785861e-02 | 2.530076e-02 |
| block1.cross_attn_kv | 2.548921e-02 | 3.841035e-03 |
| block3.adaln | 2.398621e-02 | 2.353936e-02 |
| block1.cross_attn_out_gate | 2.386345e-02 | 5.072680e-03 |
| block1.cross_attn_q | 1.662208e-02 | 3.470904e-03 |

## Aggregated Gradient Paths

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 2.950728e-02 | 2.025609e-03 |
| style_patch_proj | 6.273118e-02 | 3.000204e-03 |
| input_time | 6.770783e-01 | 8.092284e-02 |
| self_attn | 1.948360e-01 | 1.045440e-02 |
| cross_attn_q | 6.195978e-02 | 6.526088e-03 |
| cross_attn_kv | 7.635072e-02 | 5.740105e-03 |
| cross_attn_out_gate | 1.614086e-01 | 1.735240e-02 |
| adaln | 7.327814e-02 | 3.121534e-02 |
| ffn | 3.562910e-01 | 1.672623e-02 |
| head_ll | 4.168625e-01 | 5.170978e-02 |
| head_hf | 4.229333e-01 | 3.522860e-02 |

## Input Tensor Gradients

| tensor | tensor rms | grad rms | grad/tensor |
|---|---:|---:|---:|
| content | 1.137552e+00 | 4.814984e-05 | 4.232761e-05 |
| target_style | 8.864332e-01 | 6.278435e-05 | 7.082807e-05 |

## Per-Loss Gradient Paths


### loss

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 4.074807e-02 | 2.797264e-03 |
| style_patch_proj | 1.007097e-01 | 4.816576e-03 |
| input_time | 7.592048e-01 | 9.073841e-02 |
| self_attn | 2.869242e-01 | 1.539561e-02 |
| cross_attn_q | 9.536193e-02 | 1.004426e-02 |
| cross_attn_kv | 1.137167e-01 | 8.549305e-03 |
| cross_attn_out_gate | 2.234337e-01 | 2.402046e-02 |
| adaln | 1.457542e-01 | 6.208902e-02 |
| ffn | 5.575100e-01 | 2.617253e-02 |
| head_ll | 4.726322e-01 | 5.862775e-02 |
| head_hf | 6.041126e-01 | 5.032009e-02 |

### loss_fm_spectral_ll

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 2.676041e-02 | 1.837043e-03 |
| style_patch_proj | 5.028134e-02 | 2.404773e-03 |
| input_time | 3.094325e-01 | 3.698266e-02 |
| self_attn | 1.493934e-01 | 8.016066e-03 |
| cross_attn_q | 3.648174e-02 | 3.842541e-03 |
| cross_attn_kv | 5.731870e-02 | 4.309263e-03 |
| cross_attn_out_gate | 8.012977e-02 | 8.614431e-03 |
| adaln | 5.480322e-02 | 2.334531e-02 |
| ffn | 2.634867e-01 | 1.236949e-02 |
| head_ll | 4.726322e-01 | 5.862775e-02 |
| head_hf | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_lh

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 1.885096e-02 | 1.294076e-03 |
| style_patch_proj | 3.047526e-02 | 1.457521e-03 |
| input_time | 4.777321e-01 | 5.709744e-02 |
| self_attn | 1.315345e-01 | 7.057804e-03 |
| cross_attn_q | 3.070419e-02 | 3.234005e-03 |
| cross_attn_kv | 3.622679e-02 | 2.723558e-03 |
| cross_attn_out_gate | 8.373371e-02 | 9.001877e-03 |
| adaln | 7.971196e-02 | 3.395605e-02 |
| ffn | 2.080177e-01 | 9.765474e-03 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 2.172303e-01 | 1.809439e-02 |

### loss_fm_spectral_hl

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 3.320497e-02 | 2.279447e-03 |
| style_patch_proj | 7.358881e-02 | 3.519485e-03 |
| input_time | 5.446037e-01 | 6.508978e-02 |
| self_attn | 1.977356e-01 | 1.060998e-02 |
| cross_attn_q | 7.900850e-02 | 8.321792e-03 |
| cross_attn_kv | 9.518507e-02 | 7.156086e-03 |
| cross_attn_out_gate | 1.233799e-01 | 1.326408e-02 |
| adaln | 9.736814e-02 | 4.147730e-02 |
| ffn | 4.419209e-01 | 2.074616e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 5.637048e-01 | 4.695428e-02 |

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
| style_conditioner.patch_proj | 5.080165e-01 | 6.948947e-06 | 1.367859e-05 |
| time_proj | 7.212912e-01 | 1.618836e-03 | 2.244359e-03 |
| input_proj | 4.082856e-01 | 3.055986e-05 | 7.484922e-05 |
| block0.sa_qkv | 5.665132e-01 | 1.236301e-06 | 2.182299e-06 |
| block0.ca_q | 2.787561e-01 | 2.958700e-06 | 1.061394e-05 |
| block0.ca_k | 3.417251e-01 | 3.434978e-06 | 1.005188e-05 |
| block0.ca_v | 3.324002e-01 | 3.337415e-06 | 1.004035e-05 |
| block0.ca_out | 8.657137e-01 | 1.699106e-06 | 1.962665e-06 |
| block0.ffn | 2.157104e-01 | 2.635589e-05 | 1.221818e-04 |
| block0.residual | 3.831311e-01 | 2.635589e-05 | 6.879079e-05 |
| block1.sa_qkv | 6.070875e-01 | 8.725222e-07 | 1.437226e-06 |
| block1.ca_q | 2.732284e-01 | 4.024875e-06 | 1.473081e-05 |
| block1.ca_k | 3.322582e-01 | 2.378094e-06 | 7.157367e-06 |
| block1.ca_v | 3.398889e-01 | 2.839704e-06 | 8.354803e-06 |
| block1.ca_out | 1.157623e+00 | 1.467990e-06 | 1.268107e-06 |
| block1.ffn | 2.366361e-01 | 2.291635e-05 | 9.684217e-05 |
| block1.residual | 4.288039e-01 | 2.291635e-05 | 5.344249e-05 |
| block2.sa_qkv | 6.160837e-01 | 6.724318e-07 | 1.091462e-06 |
| block2.ca_q | 3.086133e-01 | 6.101756e-06 | 1.977152e-05 |
| block2.ca_k | 3.598685e-01 | 3.219644e-06 | 8.946722e-06 |
| block2.ca_v | 3.465215e-01 | 3.669691e-06 | 1.059008e-05 |
| block2.ca_out | 1.665250e+00 | 1.375464e-06 | 8.259804e-07 |
| block2.ffn | 2.908537e-01 | 1.897441e-05 | 6.523697e-05 |
| block2.residual | 4.883835e-01 | 1.897441e-05 | 3.885146e-05 |
| block3.sa_qkv | 6.366589e-01 | 4.507002e-07 | 7.079146e-07 |
| block3.ca_q | 3.335755e-01 | 4.427505e-06 | 1.327287e-05 |
| block3.ca_k | 3.560586e-01 | 3.504595e-06 | 9.842748e-06 |
| block3.ca_v | 3.239613e-01 | 4.766250e-06 | 1.471241e-05 |
| block3.ca_out | 1.507482e+00 | 1.097483e-06 | 7.280237e-07 |
| block3.ffn | 3.483499e-01 | 1.525285e-05 | 4.378601e-05 |
| block3.residual | 6.115152e-01 | 1.525285e-05 | 2.494272e-05 |
| head_ll | 2.391732e-01 | 1.154930e-05 | 4.828843e-05 |
| head_lh | 3.586696e-01 | 8.334605e-05 | 2.323756e-04 |
| head_hl | 4.063478e-01 | 8.877848e-05 | 2.184791e-04 |

## Cross-Attention Debug

| block | style gate | delta abs | ca in std | ca out std |
|---:|---:|---:|---:|---:|
| 0 | 0.056254 | 0.047031 | 0.479143 | 2.449476 |
| 1 | 0.056904 | 0.043311 | 0.442625 | 1.682280 |
| 2 | 0.060976 | 0.062572 | 0.482886 | 2.422591 |
| 3 | 0.058741 | 0.044529 | 0.531766 | 3.084933 |

## Style Condition Sensitivity


### style_id_only_fixed_target_latent

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 1.173547e-01 | 3.044485e-01 | 3.854666e-01 |
| lh | 6.090168e-02 | 2.108465e-01 | 2.888437e-01 |
| hl | 6.615859e-02 | 2.875174e-01 | 2.301029e-01 |

### target_style_latent_only_fixed_id

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 0.000000e+00 | 3.044484e-01 | 0.000000e+00 |
| lh | 0.000000e+00 | 2.108462e-01 | 0.000000e+00 |
| hl | 0.000000e+00 | 2.875176e-01 | 0.000000e+00 |

### style_id_and_target_latent

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 1.277658e-01 | 3.044484e-01 | 4.196634e-01 |
| lh | 6.555824e-02 | 2.108462e-01 | 3.109292e-01 |
| hl | 7.111215e-02 | 2.875176e-01 | 2.473315e-01 |
