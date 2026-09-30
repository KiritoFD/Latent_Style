# Baseline Internal Flow Probe

Config: `I:\Github\Latent_Style\SchrodingerBridge\configs\exp_brk_a_ll03_10ep.json`
Checkpoint: `I:\Github\Latent_Style\SchrodingerBridge\exp\dino_s_break\brk_a_ll03_10ep\epoch_0010.pt`
Device: `cuda`
Batches: 1, batch size: 4
Load info: `{'missing': 0, 'unexpected': 0}`

## Loss Components

| component | value | weighted value |
|---|---:|---:|
| loss | 0.930701 | 0.930701 |
| loss_fm_spectral_ll | 0.100332 | 0.030099 |
| loss_fm_spectral_lh | 0.399380 | 0.399380 |
| loss_fm_spectral_hl | 0.501221 | 0.501221 |
| loss_fm_spectral_hh | 0.000000 | 0.000000 |
| t_mean | 0.265435 | 0.265435 |
| flow | 0.930701 | 0.930701 |
| stat | 0.000000 | 0.000000 |
| fft | 0.000000 | 0.000000 |

## Parameter Gradient Groups

| group | grad norm | grad/param |
|---|---:|---:|
| time_proj | 7.518819e-01 | 1.057400e-01 |
| head_ll | 6.240585e-01 | 7.741145e-02 |
| input_proj | 4.367734e-01 | 9.905076e-02 |
| block0.ffn | 3.192271e-01 | 3.073257e-02 |
| head_hl | 3.074506e-01 | 3.622334e-02 |
| block1.ffn | 2.767549e-01 | 2.623495e-02 |
| block2.ffn | 2.639303e-01 | 2.456059e-02 |
| head_lh | 2.407085e-01 | 2.835014e-02 |
| block3.ffn | 2.123876e-01 | 1.946217e-02 |
| block0.self_attn | 1.667366e-01 | 1.795278e-02 |
| block1.self_attn | 1.556506e-01 | 1.665653e-02 |
| block0.adaln | 1.342894e-01 | 1.065447e-01 |
| block2.self_attn | 1.210511e-01 | 1.305052e-02 |
| block2.cross_attn_out_gate | 1.199185e-01 | 2.582699e-02 |
| block1.adaln | 1.169107e-01 | 9.043000e-02 |
| block3.self_attn | 1.003535e-01 | 1.071537e-02 |
| style_conditioner.patch_proj | 7.486277e-02 | 3.580414e-03 |
| block2.cross_attn_kv | 6.643555e-02 | 1.000082e-02 |
| block2.adaln | 6.638273e-02 | 6.028777e-02 |
| block1.cross_attn_out_gate | 5.978187e-02 | 1.270790e-02 |
| block1.cross_attn_kv | 5.159490e-02 | 7.774969e-03 |
| block3.adaln | 4.192256e-02 | 4.114157e-02 |
| block3.cross_attn_kv | 4.036595e-02 | 6.059638e-03 |
| block2.cross_attn_q | 3.554523e-02 | 7.499427e-03 |
| block0.cross_attn_kv | 3.279564e-02 | 4.922768e-03 |
| block3.cross_attn_q | 3.236823e-02 | 6.839099e-03 |
| block0.cross_attn_out_gate | 3.161648e-02 | 6.860412e-03 |
| block3.cross_attn_out_gate | 3.147657e-02 | 6.773366e-03 |
| style_memory | 3.049321e-02 | 2.093291e-03 |
| block1.cross_attn_q | 1.666465e-02 | 3.479793e-03 |
| block0.cross_attn_q | 1.218918e-02 | 2.578871e-03 |

## Aggregated Gradient Paths

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 3.049321e-02 | 2.093291e-03 |
| style_patch_proj | 7.486277e-02 | 3.580414e-03 |
| input_time | 8.695386e-01 | 1.039253e-01 |
| self_attn | 2.770423e-01 | 1.486538e-02 |
| cross_attn_q | 5.232067e-02 | 5.510821e-03 |
| cross_attn_kv | 9.889732e-02 | 7.435175e-03 |
| cross_attn_out_gate | 1.412257e-01 | 1.518261e-02 |
| adaln | 1.945917e-01 | 8.289302e-02 |
| ffn | 5.415413e-01 | 2.542287e-02 |
| head_ll | 6.240585e-01 | 7.741145e-02 |
| head_hf | 3.904695e-01 | 3.252450e-02 |

## Input Tensor Gradients

| tensor | tensor rms | grad rms | grad/tensor |
|---|---:|---:|---:|
| content | 7.699862e-01 | 5.394683e-05 | 7.006207e-05 |
| target_style | 9.391144e-01 | 5.571954e-05 | 5.933201e-05 |

## Per-Loss Gradient Paths


### loss

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 3.895399e-02 | 2.674104e-03 |
| style_patch_proj | 8.366902e-02 | 4.001584e-03 |
| input_time | 5.846455e-01 | 6.987548e-02 |
| self_attn | 2.536812e-01 | 1.361188e-02 |
| cross_attn_q | 5.693251e-02 | 5.996576e-03 |
| cross_attn_kv | 8.910471e-02 | 6.698960e-03 |
| cross_attn_out_gate | 1.521292e-01 | 1.635480e-02 |
| adaln | 1.233725e-01 | 5.255475e-02 |
| ffn | 4.644318e-01 | 2.180294e-02 |
| head_ll | 6.135213e-01 | 7.610436e-02 |
| head_hf | 3.777933e-01 | 3.146862e-02 |

### loss_fm_spectral_ll

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 3.104092e-02 | 2.130890e-03 |
| style_patch_proj | 7.926639e-02 | 3.791023e-03 |
| input_time | 2.783820e-01 | 3.327158e-02 |
| self_attn | 2.019584e-01 | 1.083657e-02 |
| cross_attn_q | 4.662553e-02 | 4.910965e-03 |
| cross_attn_kv | 8.506590e-02 | 6.395319e-03 |
| cross_attn_out_gate | 9.503387e-02 | 1.021671e-02 |
| adaln | 1.040618e-01 | 4.432871e-02 |
| ffn | 3.754581e-01 | 1.762603e-02 |
| head_ll | 6.135213e-01 | 7.610436e-02 |
| head_hf | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_lh

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 9.446131e-03 | 6.484558e-04 |
| style_patch_proj | 1.773653e-02 | 8.482738e-04 |
| input_time | 1.718244e-01 | 2.053605e-02 |
| self_attn | 7.374871e-02 | 3.957167e-03 |
| cross_attn_q | 1.754515e-02 | 1.847992e-03 |
| cross_attn_kv | 2.252837e-02 | 1.693700e-03 |
| cross_attn_out_gate | 2.426997e-02 | 2.609168e-03 |
| adaln | 3.203130e-02 | 1.364483e-02 |
| ffn | 1.414726e-01 | 6.641490e-03 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 3.003941e-01 | 2.502159e-02 |

### loss_fm_spectral_hl

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 1.777928e-02 | 1.220508e-03 |
| style_patch_proj | 3.240324e-02 | 1.549729e-03 |
| input_time | 4.306361e-01 | 5.146864e-02 |
| self_attn | 1.386726e-01 | 7.440818e-03 |
| cross_attn_q | 2.975736e-02 | 3.134277e-03 |
| cross_attn_kv | 3.567816e-02 | 2.682311e-03 |
| cross_attn_out_gate | 6.936327e-02 | 7.456968e-03 |
| adaln | 6.701012e-02 | 2.854526e-02 |
| ffn | 2.454477e-01 | 1.152264e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 2.291094e-01 | 1.908387e-02 |

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
| style_conditioner.patch_proj | 5.101224e-01 | 6.677905e-06 | 1.309079e-05 |
| time_proj | 7.145193e-01 | 1.596998e-03 | 2.235066e-03 |
| input_proj | 3.534846e-01 | 2.524702e-05 | 7.142326e-05 |
| block0.sa_qkv | 5.626242e-01 | 1.170217e-06 | 2.079926e-06 |
| block0.ca_q | 2.435718e-01 | 2.016284e-06 | 8.277985e-06 |
| block0.ca_k | 3.423524e-01 | 2.143966e-06 | 6.262453e-06 |
| block0.ca_v | 3.332817e-01 | 2.708562e-06 | 8.126945e-06 |
| block0.ca_out | 5.057725e-01 | 1.393033e-06 | 2.754268e-06 |
| block0.ffn | 2.166001e-01 | 2.160329e-05 | 9.973815e-05 |
| block0.residual | 3.666156e-01 | 2.160329e-05 | 5.892627e-05 |
| block1.sa_qkv | 6.054226e-01 | 7.921346e-07 | 1.308400e-06 |
| block1.ca_q | 2.741105e-01 | 3.656203e-06 | 1.333843e-05 |
| block1.ca_k | 3.315137e-01 | 2.437955e-06 | 7.354009e-06 |
| block1.ca_v | 3.412292e-01 | 4.302286e-06 | 1.260820e-05 |
| block1.ca_out | 1.545193e+00 | 1.202425e-06 | 7.781716e-07 |
| block1.ffn | 2.357661e-01 | 1.878064e-05 | 7.965796e-05 |
| block1.residual | 4.400788e-01 | 1.878064e-05 | 4.267564e-05 |
| block2.sa_qkv | 6.050854e-01 | 5.868987e-07 | 9.699436e-07 |
| block2.ca_q | 3.168129e-01 | 5.201304e-06 | 1.641759e-05 |
| block2.ca_k | 3.626013e-01 | 4.056545e-06 | 1.118734e-05 |
| block2.ca_v | 3.473951e-01 | 5.880702e-06 | 1.692800e-05 |
| block2.ca_out | 2.178908e+00 | 1.121594e-06 | 5.147507e-07 |
| block2.ffn | 3.058530e-01 | 1.591610e-05 | 5.203839e-05 |
| block2.residual | 5.201476e-01 | 1.591610e-05 | 3.059920e-05 |
| block3.sa_qkv | 6.380225e-01 | 3.919119e-07 | 6.142603e-07 |
| block3.ca_q | 3.763691e-01 | 4.151189e-06 | 1.102957e-05 |
| block3.ca_k | 3.581694e-01 | 3.318826e-06 | 9.266080e-06 |
| block3.ca_v | 3.255992e-01 | 3.036549e-06 | 9.326034e-06 |
| block3.ca_out | 1.419878e+00 | 9.224803e-07 | 6.496900e-07 |
| block3.ffn | 3.520197e-01 | 1.323795e-05 | 3.760572e-05 |
| block3.residual | 6.572949e-01 | 1.323795e-05 | 2.014005e-05 |
| head_ll | 1.236178e-01 | 1.159979e-05 | 9.383591e-05 |
| head_lh | 3.246256e-01 | 7.714419e-05 | 2.376405e-04 |
| head_hl | 3.795511e-01 | 8.642211e-05 | 2.276956e-04 |

## Cross-Attention Debug

| block | style gate | delta abs | ca in std | ca out std |
|---:|---:|---:|---:|---:|
| 0 | 0.056254 | 0.018572 | 0.364045 | 0.791771 |
| 1 | 0.056904 | 0.055390 | 0.411821 | 2.288931 |
| 2 | 0.060976 | 0.079627 | 0.480475 | 3.487193 |
| 3 | 0.058741 | 0.048113 | 0.565067 | 2.425394 |

## Style Condition Sensitivity


### style_id_only_fixed_target_latent

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 1.298714e-01 | 2.476530e-01 | 5.244086e-01 |
| lh | 6.393281e-02 | 1.820686e-01 | 3.511469e-01 |
| hl | 7.012604e-02 | 2.229519e-01 | 3.145343e-01 |

### target_style_latent_only_fixed_id

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 0.000000e+00 | 2.476525e-01 | 0.000000e+00 |
| lh | 0.000000e+00 | 1.820664e-01 | 0.000000e+00 |
| hl | 0.000000e+00 | 2.229548e-01 | 0.000000e+00 |

### style_id_and_target_latent

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 1.418820e-01 | 2.476525e-01 | 5.729075e-01 |
| lh | 6.972610e-02 | 1.820664e-01 | 3.829708e-01 |
| hl | 7.652584e-02 | 2.229548e-01 | 3.432348e-01 |
