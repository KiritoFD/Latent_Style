# Baseline Internal Flow Probe

Config: `configs\exp_brk_a_ll03_10ep.json`
Checkpoint: `exp\dino_s_break\brk_a_ll03_10ep\epoch_0010.pt`
Device: `cuda`
Batches: 2, batch size: 4
Load info: `{'missing': 0, 'unexpected': 0}`

## Loss Components

| component | value | weighted value |
|---|---:|---:|
| loss | 1.870387 | 1.870387 |
| loss_fm_spectral_ll | 0.098540 | 0.029562 |
| loss_fm_spectral_lh | 0.480071 | 0.480071 |
| loss_fm_spectral_hl | 0.525415 | 0.525415 |
| loss_fm_spectral_hh | 0.000000 | 0.000000 |
| t_mean | 0.393818 | 0.393818 |
| flow | 1.035048 | 1.035048 |
| stat | 0.835339 | 0.835339 |
| fft | 0.000000 | 0.000000 |

## Parameter Gradient Groups

| group | grad norm | grad/param |
|---|---:|---:|
| input_proj | 2.691551e+00 | 6.103856e-01 |
| block0.ffn | 2.080973e+00 | 2.003390e-01 |
| block1.ffn | 1.846277e+00 | 1.750176e-01 |
| block1.cross_attn_out_gate | 1.419720e+00 | 3.017914e-01 |
| block2.ffn | 1.259558e+00 | 1.172107e-01 |
| block0.self_attn | 1.250951e+00 | 1.346918e-01 |
| block1.self_attn | 1.187630e+00 | 1.270910e-01 |
| block2.cross_attn_out_gate | 1.154439e+00 | 2.486329e-01 |
| head_lh | 1.124913e+00 | 1.324899e-01 |
| head_hl | 1.082249e+00 | 1.275089e-01 |
| style_conditioner.patch_proj | 9.083730e-01 | 4.344417e-02 |
| time_proj | 8.637470e-01 | 1.214721e-01 |
| block0.adaln | 7.988690e-01 | 6.338193e-01 |
| block2.self_attn | 7.970279e-01 | 8.592759e-02 |
| block0.cross_attn_out_gate | 7.862890e-01 | 1.706156e-01 |
| block3.ffn | 7.737377e-01 | 7.090157e-02 |
| block1.adaln | 7.244020e-01 | 5.603222e-01 |
| block3.cross_attn_out_gate | 6.146876e-01 | 1.322731e-01 |
| block1.cross_attn_kv | 4.958618e-01 | 7.472269e-02 |
| block2.adaln | 4.949150e-01 | 4.494742e-01 |
| block3.self_attn | 4.692647e-01 | 5.010633e-02 |
| block1.cross_attn_q | 3.724094e-01 | 7.776385e-02 |
| style_memory | 3.379216e-01 | 2.319756e-02 |
| head_ll | 3.308242e-01 | 4.103715e-02 |
| block2.cross_attn_kv | 3.290144e-01 | 4.952792e-02 |
| block3.adaln | 3.110759e-01 | 3.052808e-01 |
| block3.cross_attn_kv | 2.870042e-01 | 4.308437e-02 |
| block0.cross_attn_kv | 2.417465e-01 | 3.628720e-02 |
| block2.cross_attn_q | 2.331332e-01 | 4.918707e-02 |
| block3.cross_attn_q | 1.575338e-01 | 3.328538e-02 |
| block0.cross_attn_q | 1.384956e-01 | 2.930157e-02 |

## Aggregated Gradient Paths

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 3.379216e-01 | 2.319756e-02 |
| style_patch_proj | 9.083730e-01 | 4.344417e-02 |
| input_time | 2.826749e+00 | 3.378465e-01 |
| self_attn | 1.957245e+00 | 1.050207e-01 |
| cross_attn_q | 4.868652e-01 | 5.128044e-02 |
| cross_attn_kv | 7.035212e-01 | 5.289126e-02 |
| cross_attn_out_gate | 2.084328e+00 | 2.240778e-01 |
| adaln | 1.226645e+00 | 5.225318e-01 |
| ffn | 3.150292e+00 | 1.478917e-01 |
| head_ll | 3.308242e-01 | 4.103715e-02 |
| head_hf | 1.560991e+00 | 1.300241e-01 |

## Input Tensor Gradients

| tensor | tensor rms | grad rms | grad/tensor |
|---|---:|---:|---:|
| content | 7.737666e-01 | 1.346508e-04 | 1.740199e-04 |
| target_style | 8.157434e-01 | 1.365393e-04 | 1.673802e-04 |

## Per-Loss Gradient Paths


### loss

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 2.972504e-01 | 2.040558e-02 |
| style_patch_proj | 7.549493e-01 | 3.610647e-02 |
| input_time | 3.830499e+00 | 4.578125e-01 |
| self_attn | 2.037503e+00 | 1.093272e-01 |
| cross_attn_q | 4.407037e-01 | 4.641835e-02 |
| cross_attn_kv | 6.060056e-01 | 4.555996e-02 |
| cross_attn_out_gate | 1.665378e+00 | 1.790381e-01 |
| adaln | 1.220730e+00 | 5.200119e-01 |
| ffn | 3.488781e+00 | 1.637822e-01 |
| head_ll | 2.621922e-01 | 3.252368e-02 |
| head_hf | 1.588050e+00 | 1.322780e-01 |

### loss_fm_hf_total

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 2.338050e-02 | 1.605019e-03 |
| style_patch_proj | 6.548857e-02 | 3.132080e-03 |
| input_time | 1.076005e+00 | 1.286017e-01 |
| self_attn | 2.618531e-01 | 1.405036e-02 |
| cross_attn_q | 6.724159e-02 | 7.082408e-03 |
| cross_attn_kv | 8.091925e-02 | 6.083570e-03 |
| cross_attn_out_gate | 1.638692e-01 | 1.761693e-02 |
| adaln | 2.271219e-01 | 9.675039e-02 |
| ffn | 4.554681e-01 | 2.138213e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 2.701590e-01 | 2.250313e-02 |

### loss_fm_spectral_ll

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 1.961450e-02 | 1.346492e-03 |
| style_patch_proj | 2.627767e-02 | 1.256765e-03 |
| input_time | 1.224976e-01 | 1.464064e-02 |
| self_attn | 7.002151e-02 | 3.757175e-03 |
| cross_attn_q | 1.264952e-02 | 1.332347e-03 |
| cross_attn_kv | 2.744459e-02 | 2.063305e-03 |
| cross_attn_out_gate | 2.416185e-02 | 2.597544e-03 |
| adaln | 2.991595e-02 | 1.274373e-02 |
| ffn | 1.345257e-01 | 6.315361e-03 |
| head_ll | 2.621922e-01 | 3.252368e-02 |
| head_hf | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_lh

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 1.018844e-02 | 6.994140e-04 |
| style_patch_proj | 2.219812e-02 | 1.061655e-03 |
| input_time | 4.351713e-01 | 5.201068e-02 |
| self_attn | 1.160079e-01 | 6.224686e-03 |
| cross_attn_q | 2.200866e-02 | 2.318123e-03 |
| cross_attn_kv | 2.791228e-02 | 2.098466e-03 |
| cross_attn_out_gate | 3.348168e-02 | 3.599482e-03 |
| adaln | 8.722962e-02 | 3.715845e-02 |
| ffn | 1.988207e-01 | 9.333717e-03 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 1.994061e-01 | 1.660970e-02 |

### loss_fm_spectral_hl

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 1.851597e-02 | 1.271080e-03 |
| style_patch_proj | 4.971594e-02 | 2.377732e-03 |
| input_time | 6.896672e-01 | 8.242744e-02 |
| self_attn | 1.640329e-01 | 8.801585e-03 |
| cross_attn_q | 5.331744e-02 | 5.615808e-03 |
| cross_attn_kv | 6.057897e-02 | 4.554373e-03 |
| cross_attn_out_gate | 1.404273e-01 | 1.509678e-02 |
| adaln | 1.425936e-01 | 6.074264e-02 |
| ffn | 3.071303e-01 | 1.441836e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 1.822721e-01 | 1.518251e-02 |

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

### loss_stat

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 2.773605e-01 | 1.904018e-02 |
| style_patch_proj | 7.145073e-01 | 3.417228e-02 |
| input_time | 3.530872e+00 | 4.220017e-01 |
| self_attn | 1.977952e+00 | 1.061318e-01 |
| cross_attn_q | 4.184546e-01 | 4.407491e-02 |
| cross_attn_kv | 5.766089e-01 | 4.334990e-02 |
| cross_attn_out_gate | 1.601491e+00 | 1.721699e-01 |
| adaln | 1.167737e+00 | 4.974379e-01 |
| ffn | 3.412927e+00 | 1.602212e-01 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 1.543965e+00 | 1.286059e-01 |

### loss_stat_lh

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 1.278949e-01 | 8.779698e-03 |
| style_patch_proj | 3.276005e-01 | 1.566794e-02 |
| input_time | 1.784864e+00 | 2.133228e-01 |
| self_attn | 9.676447e-01 | 5.192133e-02 |
| cross_attn_q | 1.991920e-01 | 2.098045e-02 |
| cross_attn_kv | 2.692962e-01 | 2.024589e-02 |
| cross_attn_out_gate | 7.383655e-01 | 7.937873e-02 |
| adaln | 5.705797e-01 | 2.430580e-01 |
| ffn | 1.723273e+00 | 8.089974e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 1.077415e+00 | 8.974423e-02 |

### loss_stat_hl

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 1.514963e-01 | 1.039988e-02 |
| style_patch_proj | 3.936305e-01 | 1.882591e-02 |
| input_time | 1.875839e+00 | 2.241960e-01 |
| self_attn | 1.022953e+00 | 5.488906e-02 |
| cross_attn_q | 2.427666e-01 | 2.557008e-02 |
| cross_attn_kv | 3.261126e-01 | 2.451739e-02 |
| cross_attn_out_gate | 8.844078e-01 | 9.507916e-02 |
| adaln | 6.028935e-01 | 2.568232e-01 |
| ffn | 1.868825e+00 | 8.773276e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 1.105896e+00 | 9.211655e-02 |

### loss_stat_hh

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

## Loss Gradient Cosines

| pair / norm | value |
|---|---:|
| cos_fm_hf_vs_stat | 1.216368e-01 |
| cos_lh_mse_vs_stat | 2.803924e-02 |
| cos_hl_mse_vs_stat | 1.687923e-01 |
| grad_norm_loss_fm_hf_total | 1.267080e+00 |
| grad_norm_loss_stat | 5.951831e+00 |
| grad_norm_loss_fm_spectral_lh | 5.438133e-01 |
| grad_norm_loss_stat_lh | 3.059746e+00 |
| grad_norm_loss_fm_spectral_hl | 8.238317e-01 |
| grad_norm_loss_stat_hl | 3.280118e+00 |

## Activation Gradient Probes

| module | act rms | grad rms | grad/act |
|---|---:|---:|---:|
| style_conditioner.patch_proj | 5.069635e-01 | 4.590629e-05 | 9.055147e-05 |
| time_proj | 7.172062e-01 | 1.412910e-02 | 1.970019e-02 |
| input_proj | 3.735255e-01 | 8.096142e-05 | 2.167493e-04 |
| block0.sa_qkv | 5.669976e-01 | 1.056489e-05 | 1.863304e-05 |
| block0.ca_q | 2.548606e-01 | 8.156889e-06 | 3.200529e-05 |
| block0.ca_k | 3.414115e-01 | 1.784427e-05 | 5.226617e-05 |
| block0.ca_v | 3.319594e-01 | 1.938544e-05 | 5.839701e-05 |
| block0.ca_out | 5.745298e-01 | 4.139367e-06 | 7.204791e-06 |
| block0.ffn | 2.150492e-01 | 6.342312e-05 | 2.949237e-04 |
| block0.residual | 3.704220e-01 | 6.342312e-05 | 1.712185e-04 |
| block1.sa_qkv | 6.038094e-01 | 7.214958e-06 | 1.194907e-05 |
| block1.ca_q | 2.671836e-01 | 1.052562e-05 | 3.939470e-05 |
| block1.ca_k | 3.326304e-01 | 2.564012e-05 | 7.708289e-05 |
| block1.ca_v | 3.392187e-01 | 2.146157e-05 | 6.326764e-05 |
| block1.ca_out | 1.173106e+00 | 3.086128e-06 | 2.630733e-06 |
| block1.ffn | 2.369787e-01 | 4.540841e-05 | 1.916139e-04 |
| block1.residual | 4.241489e-01 | 4.540841e-05 | 1.070577e-04 |
| block2.sa_qkv | 6.167805e-01 | 3.809648e-06 | 6.176667e-06 |
| block2.ca_q | 3.050794e-01 | 1.222190e-05 | 4.006136e-05 |
| block2.ca_k | 3.585021e-01 | 2.250360e-05 | 6.277117e-05 |
| block2.ca_v | 3.460847e-01 | 2.328303e-05 | 6.727552e-05 |
| block2.ca_out | 1.700933e+00 | 2.511205e-06 | 1.476369e-06 |
| block2.ffn | 2.958802e-01 | 3.297413e-05 | 1.114442e-04 |
| block2.residual | 4.898091e-01 | 3.297413e-05 | 6.732036e-05 |
| block3.sa_qkv | 6.327144e-01 | 2.098814e-06 | 3.317159e-06 |
| block3.ca_q | 3.392908e-01 | 9.114482e-06 | 2.686333e-05 |
| block3.ca_k | 3.550032e-01 | 1.540813e-05 | 4.340280e-05 |
| block3.ca_v | 3.231423e-01 | 1.787037e-05 | 5.530186e-05 |
| block3.ca_out | 1.357510e+00 | 1.831872e-06 | 1.349436e-06 |
| block3.ffn | 3.506862e-01 | 2.565899e-05 | 7.316794e-05 |
| block3.residual | 6.120207e-01 | 2.565899e-05 | 4.192503e-05 |
| head_ll | 1.294623e-01 | 1.110380e-05 | 8.576863e-05 |
| head_lh | 3.754889e-01 | 1.408878e-04 | 3.752115e-04 |
| head_hl | 3.997441e-01 | 1.485030e-04 | 3.714952e-04 |

## Cross-Attention Debug

| block | style gate | delta abs | ca in std | ca out std |
|---:|---:|---:|---:|---:|
| 0 | 0.056254 | 0.014864 | 0.359909 | 0.638764 |
| 1 | 0.056904 | 0.039003 | 0.394760 | 1.516927 |
| 2 | 0.060976 | 0.051642 | 0.463046 | 2.055854 |
| 3 | 0.058741 | 0.061876 | 0.520433 | 3.052706 |

## Style Condition Sensitivity


### style_id_only_fixed_target_latent

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 1.212941e-01 | 9.485503e-02 | 1.278731e+00 |
| lh | 5.379479e-02 | 1.689727e-01 | 3.183639e-01 |
| hl | 5.653413e-02 | 1.622811e-01 | 3.483716e-01 |

### target_style_latent_only_fixed_id

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 0.000000e+00 | 9.485499e-02 | 0.000000e+00 |
| lh | 0.000000e+00 | 1.689727e-01 | 0.000000e+00 |
| hl | 0.000000e+00 | 1.622810e-01 | 0.000000e+00 |

### style_id_and_target_latent

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 1.188884e-01 | 9.485499e-02 | 1.253370e+00 |
| lh | 5.585253e-02 | 1.689727e-01 | 3.305417e-01 |
| hl | 5.932442e-02 | 1.622810e-01 | 3.655660e-01 |
