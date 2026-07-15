# Baseline Internal Flow Probe

Config: `I:\Github\Latent_Style\SchrodingerBridge\configs\exp_probe_target_latent_fusion.json`
Checkpoint: `I:\Github\Latent_Style\SchrodingerBridge\exp\dino_s_break\brk_a_ll03_10ep\epoch_0010.pt`
Device: `cuda`
Batches: 1, batch size: 4
Load info: `{'missing': 11, 'unexpected': 0}`

## Loss Components

| component | value | weighted value |
|---|---:|---:|
| loss | 0.930621 | 0.930621 |
| loss_fm_spectral_ll | 0.098692 | 0.029608 |
| loss_fm_spectral_lh | 0.399474 | 0.399474 |
| loss_fm_spectral_hl | 0.501540 | 0.501540 |
| loss_fm_spectral_hh | 0.000000 | 0.000000 |
| t_mean | 0.265435 | 0.265435 |
| flow | 0.930621 | 0.930621 |
| stat | 0.000000 | 0.000000 |
| fft | 0.000000 | 0.000000 |

## Parameter Gradient Groups

| group | grad norm | grad/param |
|---|---:|---:|
| time_proj | 7.546602e-01 | 1.061308e-01 |
| head_ll | 6.154697e-01 | 7.634605e-02 |
| input_proj | 4.438429e-01 | 1.006540e-01 |
| block0.ffn | 3.190084e-01 | 3.071151e-02 |
| head_hl | 3.094076e-01 | 3.645392e-02 |
| block1.ffn | 2.757996e-01 | 2.614440e-02 |
| block2.ffn | 2.614363e-01 | 2.432850e-02 |
| head_lh | 2.462741e-01 | 2.900566e-02 |
| block3.ffn | 2.110647e-01 | 1.934094e-02 |
| block0.self_attn | 1.674619e-01 | 1.803087e-02 |
| block1.self_attn | 1.557358e-01 | 1.666564e-02 |
| block0.adaln | 1.343071e-01 | 1.065587e-01 |
| block2.self_attn | 1.211340e-01 | 1.305946e-02 |
| block1.adaln | 1.170278e-01 | 9.052055e-02 |
| block2.cross_attn_out_gate | 1.095861e-01 | 2.360169e-02 |
| block3.self_attn | 9.964043e-02 | 1.063923e-02 |
| style_conditioner.patch_proj | 7.299206e-02 | 3.490944e-03 |
| block2.adaln | 6.579746e-02 | 5.975624e-02 |
| block2.cross_attn_kv | 6.464299e-02 | 9.730981e-03 |
| block1.cross_attn_out_gate | 5.781662e-02 | 1.229014e-02 |
| block1.cross_attn_kv | 4.921605e-02 | 7.416495e-03 |
| block3.adaln | 4.113685e-02 | 4.037051e-02 |
| block3.cross_attn_kv | 4.048279e-02 | 6.077177e-03 |
| block3.cross_attn_out_gate | 3.646428e-02 | 7.846660e-03 |
| block2.cross_attn_q | 3.429925e-02 | 7.236547e-03 |
| block3.cross_attn_q | 3.274909e-02 | 6.919571e-03 |
| block0.cross_attn_kv | 3.216298e-02 | 4.827804e-03 |
| style_memory | 2.986517e-02 | 2.050177e-03 |
| block0.cross_attn_out_gate | 2.967452e-02 | 6.439028e-03 |
| target_latent_token_fusion | 1.700474e-02 | 1.227519e-03 |
| block1.cross_attn_q | 1.651177e-02 | 3.447869e-03 |
| block0.cross_attn_q | 1.185311e-02 | 2.507768e-03 |

## Aggregated Gradient Paths

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 2.986517e-02 | 2.050177e-03 |
| style_patch_proj | 7.299206e-02 | 3.490944e-03 |
| target_latent_fusion | 1.700474e-02 | 1.227519e-03 |
| input_time | 8.755047e-01 | 1.046383e-01 |
| self_attn | 2.773063e-01 | 1.487955e-02 |
| cross_attn_q | 5.159531e-02 | 5.434421e-03 |
| cross_attn_kv | 9.630291e-02 | 7.240125e-03 |
| cross_attn_out_gate | 1.325221e-01 | 1.424692e-02 |
| adaln | 1.943076e-01 | 8.277200e-02 |
| ffn | 5.391930e-01 | 2.531263e-02 |
| head_ll | 6.154697e-01 | 7.634605e-02 |
| head_hf | 3.954542e-01 | 3.293971e-02 |

## Input Tensor Gradients

| tensor | tensor rms | grad rms | grad/tensor |
|---|---:|---:|---:|
| content | 7.699862e-01 | 5.384223e-05 | 6.992623e-05 |
| target_style | 9.391144e-01 | 5.580962e-05 | 5.942792e-05 |

## Per-Loss Gradient Paths


### loss

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 3.858310e-02 | 2.648644e-03 |
| style_patch_proj | 8.203598e-02 | 3.923482e-03 |
| target_latent_fusion | 1.918754e-02 | 1.385088e-03 |
| input_time | 5.777064e-01 | 6.904614e-02 |
| self_attn | 2.477333e-01 | 1.329273e-02 |
| cross_attn_q | 5.728897e-02 | 6.034121e-03 |
| cross_attn_kv | 8.649777e-02 | 6.502968e-03 |
| cross_attn_out_gate | 1.558987e-01 | 1.676005e-02 |
| adaln | 1.203764e-01 | 5.127845e-02 |
| ffn | 4.532678e-01 | 2.127884e-02 |
| head_ll | 5.947490e-01 | 7.377575e-02 |
| head_hf | 3.695106e-01 | 3.077871e-02 |

### loss_fm_spectral_ll

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 3.054451e-02 | 2.096813e-03 |
| style_patch_proj | 7.691901e-02 | 3.678756e-03 |
| target_latent_fusion | 2.294380e-02 | 1.656241e-03 |
| input_time | 2.681036e-01 | 3.204312e-02 |
| self_attn | 1.946932e-01 | 1.044674e-02 |
| cross_attn_q | 4.707168e-02 | 4.957956e-03 |
| cross_attn_kv | 8.212315e-02 | 6.174081e-03 |
| cross_attn_out_gate | 9.582099e-02 | 1.030133e-02 |
| adaln | 1.002494e-01 | 4.270468e-02 |
| ffn | 3.624421e-01 | 1.701499e-02 |
| head_ll | 5.947490e-01 | 7.377575e-02 |
| head_hf | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_lh

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 9.580225e-03 | 6.576611e-04 |
| style_patch_proj | 1.780068e-02 | 8.513415e-04 |
| target_latent_fusion | 5.945117e-03 | 4.291593e-04 |
| input_time | 1.696324e-01 | 2.027407e-02 |
| self_attn | 7.389246e-02 | 3.964880e-03 |
| cross_attn_q | 1.693555e-02 | 1.783784e-03 |
| cross_attn_kv | 2.189266e-02 | 1.645906e-03 |
| cross_attn_out_gate | 2.442709e-02 | 2.626059e-03 |
| adaln | 3.229641e-02 | 1.375777e-02 |
| ffn | 1.415649e-01 | 6.645820e-03 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 2.994694e-01 | 2.494457e-02 |

### loss_fm_spectral_hl

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 1.779134e-02 | 1.221336e-03 |
| style_patch_proj | 3.249915e-02 | 1.554316e-03 |
| target_latent_fusion | 8.774497e-03 | 6.334033e-04 |
| input_time | 4.248449e-01 | 5.077648e-02 |
| self_attn | 1.374865e-01 | 7.377175e-03 |
| cross_attn_q | 2.972479e-02 | 3.130847e-03 |
| cross_attn_kv | 3.515084e-02 | 2.642667e-03 |
| cross_attn_out_gate | 6.904318e-02 | 7.422556e-03 |
| adaln | 6.666820e-02 | 2.839961e-02 |
| ffn | 2.431222e-01 | 1.141347e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 2.164627e-01 | 1.803045e-02 |

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
| style_conditioner.patch_proj | 5.101224e-01 | 6.592684e-06 | 1.292373e-05 |
| target_latent_tokenizer | 3.511259e-01 | 1.915802e-07 | 5.456169e-07 |
| target_latent_token_proj | 5.607674e-01 | 3.293598e-07 | 5.873377e-07 |
| time_proj | 7.145193e-01 | 1.603616e-03 | 2.244328e-03 |
| input_proj | 3.534846e-01 | 2.506631e-05 | 7.091202e-05 |
| block0.sa_qkv | 5.626242e-01 | 1.171358e-06 | 2.081956e-06 |
| block0.ca_q | 2.435718e-01 | 1.973427e-06 | 8.102034e-06 |
| block0.ca_k | 3.412137e-01 | 2.142718e-06 | 6.279696e-06 |
| block0.ca_v | 3.321639e-01 | 2.711412e-06 | 8.162873e-06 |
| block0.ca_out | 4.965795e-01 | 1.382517e-06 | 2.784080e-06 |
| block0.ffn | 2.166260e-01 | 2.144087e-05 | 9.897642e-05 |
| block0.residual | 3.666486e-01 | 2.144087e-05 | 5.847797e-05 |
| block1.sa_qkv | 6.054220e-01 | 7.988108e-07 | 1.319428e-06 |
| block1.ca_q | 2.741123e-01 | 3.611898e-06 | 1.317671e-05 |
| block1.ca_k | 3.303309e-01 | 2.447239e-06 | 7.408448e-06 |
| block1.ca_v | 3.409624e-01 | 4.195855e-06 | 1.230592e-05 |
| block1.ca_out | 1.522068e+00 | 1.193287e-06 | 7.839906e-07 |
| block1.ffn | 2.358282e-01 | 1.865553e-05 | 7.910641e-05 |
| block1.residual | 4.400492e-01 | 1.865553e-05 | 4.239418e-05 |
| block2.sa_qkv | 6.052787e-01 | 5.864707e-07 | 9.689267e-07 |
| block2.ca_q | 3.169145e-01 | 5.100515e-06 | 1.609429e-05 |
| block2.ca_k | 3.623956e-01 | 3.939986e-06 | 1.087206e-05 |
| block2.ca_v | 3.462939e-01 | 5.848512e-06 | 1.688887e-05 |
| block2.ca_out | 2.137196e+00 | 1.114338e-06 | 5.214017e-07 |
| block2.ffn | 3.055510e-01 | 1.582861e-05 | 5.180351e-05 |
| block2.residual | 5.205542e-01 | 1.582861e-05 | 3.040724e-05 |
| block3.sa_qkv | 6.386534e-01 | 3.892871e-07 | 6.095436e-07 |
| block3.ca_q | 3.769294e-01 | 4.005663e-06 | 1.062709e-05 |
| block3.ca_k | 3.571080e-01 | 3.351314e-06 | 9.384594e-06 |
| block3.ca_v | 3.255419e-01 | 2.995013e-06 | 9.200083e-06 |
| block3.ca_out | 1.376062e+00 | 9.183444e-07 | 6.673716e-07 |
| block3.ffn | 3.523776e-01 | 1.318590e-05 | 3.741980e-05 |
| block3.residual | 6.585019e-01 | 1.318590e-05 | 2.002408e-05 |
| head_ll | 1.238182e-01 | 1.150463e-05 | 9.291554e-05 |
| head_lh | 3.238115e-01 | 7.715327e-05 | 2.382660e-04 |
| head_hl | 3.786447e-01 | 8.644954e-05 | 2.283131e-04 |

## Cross-Attention Debug

| block | style gate | delta abs | ca in std | ca out std |
|---:|---:|---:|---:|---:|
| 0 | 0.056254 | 0.018144 | 0.364045 | 0.778376 |
| 1 | 0.056904 | 0.054343 | 0.411875 | 2.266439 |
| 2 | 0.060976 | 0.078175 | 0.480668 | 3.424866 |
| 3 | 0.058741 | 0.046889 | 0.566244 | 2.350290 |

## Style Condition Sensitivity


### style_id_only_fixed_target_latent

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 1.271954e-01 | 2.428017e-01 | 5.238654e-01 |
| lh | 6.187029e-02 | 1.805979e-01 | 3.425859e-01 |
| hl | 6.796183e-02 | 2.208818e-01 | 3.076842e-01 |

### target_style_latent_only_fixed_id

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 2.680240e-03 | 2.428023e-01 | 1.103877e-02 |
| lh | 2.641246e-03 | 1.805978e-01 | 1.462502e-02 |
| hl | 3.094137e-03 | 2.208813e-01 | 1.400814e-02 |

### style_id_and_target_latent

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 1.399723e-01 | 2.428023e-01 | 5.764865e-01 |
| lh | 6.950974e-02 | 1.805978e-01 | 3.848870e-01 |
| hl | 7.665565e-02 | 2.208813e-01 | 3.470445e-01 |
