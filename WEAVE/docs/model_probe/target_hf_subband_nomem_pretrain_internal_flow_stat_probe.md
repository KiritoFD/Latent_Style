# Baseline Internal Flow Probe

Config: `configs\exp_probe_target_hf_subband_nomem_ft6.json`
Checkpoint: `exp\model_probe\target_hf_subband_ft6\epoch_0006.pt`
Device: `cuda`
Batches: 2, batch size: 4
Load info: `{'missing': 0, 'unexpected': 0}`

## Loss Components

| component | value | weighted value |
|---|---:|---:|
| loss | 3.605664 | 3.605664 |
| loss_fm_spectral_ll | 0.091409 | 0.027423 |
| loss_fm_spectral_lh | 0.472849 | 0.472849 |
| loss_fm_spectral_hl | 0.515963 | 0.515963 |
| loss_fm_spectral_hh | 0.436989 | 0.873979 |
| t_mean | 0.393818 | 0.393818 |
| flow | 1.890214 | 1.890214 |
| stat | 1.715450 | 1.715450 |
| fft | 0.000000 | 0.000000 |

## Parameter Gradient Groups

| group | grad norm | grad/param |
|---|---:|---:|
| input_proj | 6.254439e+00 | 1.444577e+00 |
| block0.ffn | 4.227972e+00 | 4.090164e-01 |
| block1.ffn | 4.049615e+00 | 3.832556e-01 |
| block2.ffn | 3.004746e+00 | 2.772629e-01 |
| block1.self_attn | 2.685603e+00 | 2.871473e-01 |
| block0.self_attn | 2.478252e+00 | 2.666391e-01 |
| head_lh | 2.209122e+00 | 2.553105e-01 |
| block2.self_attn | 1.917512e+00 | 2.064356e-01 |
| head_hl | 1.818363e+00 | 2.104578e-01 |
| target_hf_subband_fusion | 1.789971e+00 | 5.764854e-02 |
| block1.adaln | 1.743107e+00 | 1.289246e+00 |
| block3.ffn | 1.722732e+00 | 1.549514e-01 |
| head_hh | 1.715500e+00 | 2.068363e-01 |
| block0.adaln | 1.709402e+00 | 1.253774e+00 |
| block3.self_attn | 1.270025e+00 | 1.353967e-01 |
| block2.adaln | 1.225367e+00 | 1.071752e+00 |
| time_proj | 9.543723e-01 | 1.337371e-01 |
| block3.adaln | 6.549136e-01 | 5.976088e-01 |
| head_ll | 3.129357e-01 | 3.875618e-02 |
| style_memory | 0.000000e+00 | 0.000000e+00 |
| style_conditioner.patch_proj | 0.000000e+00 | 0.000000e+00 |
| block0.cross_attn_q | 0.000000e+00 | 0.000000e+00 |
| block0.cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| block0.cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| block1.cross_attn_q | 0.000000e+00 | 0.000000e+00 |
| block1.cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| block1.cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| block2.cross_attn_q | 0.000000e+00 | 0.000000e+00 |
| block2.cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| block2.cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| block3.cross_attn_q | 0.000000e+00 | 0.000000e+00 |
| block3.cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| block3.cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |

## Aggregated Gradient Paths

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 0.000000e+00 | 0.000000e+00 |
| style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| target_hf_fusion | 1.789971e+00 | 5.764854e-02 |
| target_hf_subband | 1.789971e+00 | 5.764854e-02 |
| input_time | 6.326834e+00 | 7.579873e-01 |
| self_attn | 4.317871e+00 | 2.314212e-01 |
| cross_attn_q | 0.000000e+00 | 0.000000e+00 |
| cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| adaln | 2.809076e+00 | 1.128603e+00 |
| ffn | 6.802311e+00 | 3.173141e-01 |
| head_ll | 3.129357e-01 | 3.875618e-02 |
| head_hf | 3.336106e+00 | 2.257893e-01 |

## Input Tensor Gradients

| tensor | tensor rms | grad rms | grad/tensor |
|---|---:|---:|---:|
| content | 8.835873e-01 | 2.089850e-04 | 2.365188e-04 |
| target_style | 8.353144e-01 | 2.070096e-04 | 2.478224e-04 |

## Per-Loss Gradient Paths


### loss

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 0.000000e+00 | 0.000000e+00 |
| style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| target_hf_fusion | 8.506683e-01 | 2.739697e-02 |
| target_hf_subband | 8.506683e-01 | 2.739697e-02 |
| input_time | 5.840287e+00 | 6.996964e-01 |
| self_attn | 2.430113e+00 | 1.302447e-01 |
| cross_attn_q | 0.000000e+00 | 0.000000e+00 |
| cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| adaln | 1.554009e+00 | 6.243543e-01 |
| ffn | 4.077921e+00 | 1.902268e-01 |
| head_ll | 2.581878e-01 | 3.197582e-02 |
| head_hf | 3.248180e+00 | 2.198385e-01 |

### loss_fm_hf_total

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 0.000000e+00 | 0.000000e+00 |
| style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| target_hf_fusion | 1.214205e-01 | 3.910517e-03 |
| target_hf_subband | 1.214205e-01 | 3.910517e-03 |
| input_time | 9.173781e-01 | 1.099066e-01 |
| self_attn | 2.270321e-01 | 1.216804e-02 |
| cross_attn_q | 0.000000e+00 | 0.000000e+00 |
| cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| adaln | 1.058475e-01 | 4.252636e-02 |
| ffn | 4.450182e-01 | 2.075920e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 1.064347e+00 | 7.203552e-02 |

### loss_fm_spectral_ll

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 0.000000e+00 | 0.000000e+00 |
| style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| target_hf_fusion | 0.000000e+00 | 0.000000e+00 |
| target_hf_subband | 0.000000e+00 | 0.000000e+00 |
| input_time | 5.047903e-01 | 6.047648e-02 |
| self_attn | 1.381332e-01 | 7.403403e-03 |
| cross_attn_q | 0.000000e+00 | 0.000000e+00 |
| cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| adaln | 5.008377e-02 | 2.012216e-02 |
| ffn | 2.059111e-01 | 9.605338e-03 |
| head_ll | 2.581878e-01 | 3.197582e-02 |
| head_hf | 0.000000e+00 | 0.000000e+00 |

### loss_fm_spectral_lh

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 0.000000e+00 | 0.000000e+00 |
| style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| target_hf_fusion | 6.295521e-02 | 2.027561e-03 |
| target_hf_subband | 6.295521e-02 | 2.027561e-03 |
| input_time | 3.716863e-01 | 4.452994e-02 |
| self_attn | 1.012413e-01 | 5.426144e-03 |
| cross_attn_q | 0.000000e+00 | 0.000000e+00 |
| cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| adaln | 4.977487e-02 | 1.999805e-02 |
| ffn | 1.917446e-01 | 8.944500e-03 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 5.659179e-01 | 3.830161e-02 |

### loss_fm_spectral_hl

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 0.000000e+00 | 0.000000e+00 |
| style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| target_hf_fusion | 7.103047e-02 | 2.287636e-03 |
| target_hf_subband | 7.103047e-02 | 2.287636e-03 |
| input_time | 3.700528e-01 | 4.433423e-02 |
| self_attn | 1.185619e-01 | 6.354458e-03 |
| cross_attn_q | 0.000000e+00 | 0.000000e+00 |
| cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| adaln | 5.218336e-02 | 2.096571e-02 |
| ffn | 2.450955e-01 | 1.143321e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 7.585944e-01 | 5.134205e-02 |

### loss_fm_spectral_hh

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 0.000000e+00 | 0.000000e+00 |
| style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| target_hf_fusion | 7.572478e-02 | 2.438823e-03 |
| target_hf_subband | 7.572478e-02 | 2.438823e-03 |
| input_time | 3.091837e-01 | 3.704181e-02 |
| self_attn | 8.005035e-02 | 4.290389e-03 |
| cross_attn_q | 0.000000e+00 | 0.000000e+00 |
| cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| adaln | 4.794049e-02 | 1.926105e-02 |
| ffn | 1.759860e-01 | 8.209390e-03 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 4.869347e-01 | 3.295599e-02 |

### loss_stat

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 0.000000e+00 | 0.000000e+00 |
| style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| target_hf_fusion | 8.232992e-01 | 2.651551e-02 |
| target_hf_subband | 8.232992e-01 | 2.651551e-02 |
| input_time | 5.090904e+00 | 6.099166e-01 |
| self_attn | 2.232636e+00 | 1.196607e-01 |
| cross_attn_q | 0.000000e+00 | 0.000000e+00 |
| cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| adaln | 1.455914e+00 | 5.849427e-01 |
| ffn | 3.770990e+00 | 1.759091e-01 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 2.355228e+00 | 1.594030e-01 |

### loss_stat_lh

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 0.000000e+00 | 0.000000e+00 |
| style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| target_hf_fusion | 2.656876e-01 | 8.556843e-03 |
| target_hf_subband | 2.656876e-01 | 8.556843e-03 |
| input_time | 1.575857e+00 | 1.887958e-01 |
| self_attn | 7.362341e-01 | 3.945930e-02 |
| cross_attn_q | 0.000000e+00 | 0.000000e+00 |
| cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| adaln | 4.580470e-01 | 1.840295e-01 |
| ffn | 1.444463e+00 | 6.738127e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 1.453364e+00 | 9.836439e-02 |

### loss_stat_hl

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 0.000000e+00 | 0.000000e+00 |
| style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| target_hf_fusion | 1.773036e-01 | 5.710312e-03 |
| target_hf_subband | 1.773036e-01 | 5.710312e-03 |
| input_time | 1.908221e+00 | 2.286147e-01 |
| self_attn | 8.714415e-01 | 4.670589e-02 |
| cross_attn_q | 0.000000e+00 | 0.000000e+00 |
| cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| adaln | 5.505781e-01 | 2.212058e-01 |
| ffn | 1.601508e+00 | 7.470711e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 1.682751e+00 | 1.138895e-01 |

### loss_stat_hh

| path | grad norm | grad/param |
|---|---:|---:|
| style_memory | 0.000000e+00 | 0.000000e+00 |
| style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| target_hf_fusion | 7.588117e-01 | 2.443860e-02 |
| target_hf_subband | 7.588117e-01 | 2.443860e-02 |
| input_time | 2.012041e+00 | 2.410529e-01 |
| self_attn | 7.092957e-01 | 3.801550e-02 |
| cross_attn_q | 0.000000e+00 | 0.000000e+00 |
| cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| adaln | 4.994798e-01 | 2.006760e-01 |
| ffn | 1.139761e+00 | 5.316754e-02 |
| head_ll | 0.000000e+00 | 0.000000e+00 |
| head_hf | 7.766466e-01 | 5.256383e-02 |

## Loss Gradient Cosines

| pair / norm | value |
|---|---:|
| cos_fm_hf_vs_stat | 5.969871e-01 |
| cos_lh_mse_vs_stat | 5.502067e-01 |
| cos_hl_mse_vs_stat | 6.810301e-01 |
| cos_hh_mse_vs_stat | 4.994306e-01 |
| grad_norm_loss_fm_hf_total | 1.500133e+00 |
| grad_norm_loss_stat | 7.309086e+00 |
| grad_norm_loss_fm_spectral_lh | 7.156811e-01 |
| grad_norm_loss_stat_lh | 2.741482e+00 |
| grad_norm_loss_fm_spectral_hl | 8.914367e-01 |
| grad_norm_loss_stat_hl | 3.181967e+00 |
| grad_norm_loss_fm_spectral_hh | 6.142188e-01 |
| grad_norm_loss_stat_hh | 2.695596e+00 |

## Activation Gradient Probes

| module | act rms | grad rms | grad/act |
|---|---:|---:|---:|
| style_conditioner.patch_proj | 5.277607e-01 | 0.000000e+00 | 0.000000e+00 |
| target_latent_hf_subband_encoder_lh | 1.855279e-01 | 5.837469e-04 | 3.146410e-03 |
| target_latent_hf_subband_proj_lh | 1.065237e+00 | 2.009945e-04 | 1.886853e-04 |
| target_latent_hf_subband_encoder_hl | 1.667283e-01 | 9.845904e-04 | 5.905358e-03 |
| target_latent_hf_subband_proj_hl | 1.104973e+00 | 2.660117e-04 | 2.407406e-04 |
| target_latent_hf_subband_encoder_hh | 2.321403e-01 | 1.314699e-03 | 5.663383e-03 |
| target_latent_hf_subband_proj_hh | 9.680135e-01 | 6.684561e-04 | 6.905442e-04 |
| time_proj | 7.447503e-01 | 1.722593e-02 | 2.312981e-02 |
| input_proj | 3.386531e-01 | 1.175086e-04 | 3.469882e-04 |
| block0.sa_qkv | 5.601763e-01 | 1.468623e-05 | 2.621716e-05 |
| block0.ffn | 2.194013e-01 | 8.791247e-05 | 4.006925e-04 |
| block0.residual | 3.473659e-01 | 8.791247e-05 | 2.530832e-04 |
| block1.sa_qkv | 6.153828e-01 | 1.028126e-05 | 1.670709e-05 |
| block1.ffn | 2.475662e-01 | 6.367957e-05 | 2.572224e-04 |
| block1.residual | 4.283486e-01 | 6.367957e-05 | 1.486630e-04 |
| block2.sa_qkv | 6.207510e-01 | 5.595448e-06 | 9.013998e-06 |
| block2.ffn | 3.148481e-01 | 4.940467e-05 | 1.569159e-04 |
| block2.residual | 5.427620e-01 | 4.940467e-05 | 9.102456e-05 |
| block3.sa_qkv | 6.801915e-01 | 3.384746e-06 | 4.976166e-06 |
| block3.ffn | 4.054182e-01 | 4.094102e-05 | 1.009847e-04 |
| block3.residual | 7.739959e-01 | 4.094102e-05 | 5.289566e-05 |
| head_ll | 2.208292e-01 | 1.107128e-05 | 5.013504e-05 |
| head_lh | 2.851104e-01 | 1.593304e-04 | 5.588376e-04 |
| head_hl | 3.000511e-01 | 1.721877e-04 | 5.738613e-04 |
| head_hh | 1.636812e-01 | 2.204708e-04 | 1.346952e-03 |

## Cross-Attention Debug

| block | style gate | delta abs | ca in std | ca out std |
|---:|---:|---:|---:|---:|
| 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Style Condition Sensitivity


### style_id_only_fixed_target_latent

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 0.000000e+00 | 8.571109e-02 | 0.000000e+00 |
| lh | 0.000000e+00 | 1.663585e-01 | 0.000000e+00 |
| hl | 0.000000e+00 | 1.872678e-01 | 0.000000e+00 |
| hh | 0.000000e+00 | 1.172750e-01 | 0.000000e+00 |

### target_style_latent_only_fixed_id

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 0.000000e+00 | 8.571109e-02 | 0.000000e+00 |
| lh | 1.241409e-02 | 1.663579e-01 | 7.462276e-02 |
| hl | 1.325479e-02 | 1.872678e-01 | 7.077985e-02 |
| hh | 1.009180e-02 | 1.172752e-01 | 8.605233e-02 |

### style_id_and_target_latent

| band | delta rms | base rms | delta/base |
|---|---:|---:|---:|
| ll | 0.000000e+00 | 8.571109e-02 | 0.000000e+00 |
| lh | 1.241409e-02 | 1.663579e-01 | 7.462276e-02 |
| hl | 1.325479e-02 | 1.872678e-01 | 7.077985e-02 |
| hh | 1.009180e-02 | 1.172752e-01 | 8.605233e-02 |
