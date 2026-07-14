# Route Competition Probe

Config: `configs\exp_probe_target_hf_subband_memdrop_ft6.json`
Checkpoint: `exp\model_probe\target_hf_subband_memdrop_ft6\epoch_0006.pt`
Device: `cuda`
Load info: `{'missing': 0, 'unexpected': 0}`

## Reading

Mean HF cos(memory, desired)=0.0463; cos(target-HF, desired)=0.1561; cos(target-HF | memory, desired)=0.1586; full MSE improvement=0.0361. Interpretation: target-HF remains at least as aligned after style memory is present.

## Route Transitions

| transition | band | delta/desired | cos(delta, desired) | projection | orthogonal fraction | MSE improvement |
|---|---|---:|---:|---:|---:|---:|
| backbone_to_style_memory | lh | 0.062684 | 0.047358 | 0.003133 | 0.998408 | 0.001966 |
| backbone_to_style_memory | hl | 0.062563 | 0.060828 | 0.004239 | 0.996705 | 0.004376 |
| backbone_to_style_memory | hh | 0.037770 | 0.030654 | 0.001338 | 0.997341 | 0.000972 |
| backbone_to_target_hf | lh | 0.108700 | 0.108375 | 0.012028 | 0.993712 | 0.011759 |
| backbone_to_target_hf | hl | 0.093579 | 0.088870 | 0.008599 | 0.995489 | 0.008078 |
| backbone_to_target_hf | hh | 0.286124 | 0.271154 | 0.085135 | 0.956953 | 0.078748 |
| style_memory_to_full_target_hf_marginal | lh | 0.110614 | 0.109573 | 0.012439 | 0.993584 | 0.012149 |
| style_memory_to_full_target_hf_marginal | hl | 0.095512 | 0.090577 | 0.008863 | 0.995439 | 0.008419 |
| style_memory_to_full_target_hf_marginal | hh | 0.290863 | 0.275613 | 0.087588 | 0.955941 | 0.080530 |
| target_hf_to_full_style_memory_marginal | lh | 0.062522 | 0.050739 | 0.003270 | 0.998330 | 0.002361 |
| target_hf_to_full_style_memory_marginal | hl | 0.062810 | 0.063734 | 0.004358 | 0.996576 | 0.004720 |
| target_hf_to_full_style_memory_marginal | hh | 0.058233 | 0.048382 | 0.003630 | 0.997235 | 0.002885 |
| backbone_to_full | lh | 0.125587 | 0.119031 | 0.015312 | 0.992542 | 0.014094 |
| backbone_to_full | hl | 0.112396 | 0.109968 | 0.012858 | 0.993018 | 0.012763 |
| backbone_to_full | hh | 0.292494 | 0.276536 | 0.088432 | 0.955463 | 0.081417 |

## Route Interaction

| name | band | interaction/desired | cos(interaction, desired) | projection |
|---|---|---:|---:|---:|
| route_interaction | lh | 0.024115 | 0.016671 | 0.000151 |
| route_interaction | hl | 0.018616 | 0.010024 | 0.000019 |
| route_interaction | hh | 0.045847 | 0.037175 | 0.001959 |

## Per-Block Cross-Attention Marginals

| block transition | band | delta/desired | cos(delta, desired) | MSE improvement |
|---|---|---:|---:|---:|
| block0_memory_marginal_full | lh | 0.038083 | 0.026872 | 0.000766 |
| block0_memory_marginal_full | hl | 0.038347 | 0.035897 | 0.001532 |
| block0_memory_marginal_full | hh | 0.036164 | 0.029591 | 0.001499 |
| block0_memory_marginal_no_target_hf | lh | 0.037350 | 0.023761 | 0.000569 |
| block0_memory_marginal_no_target_hf | hl | 0.037090 | 0.025257 | 0.001023 |
| block0_memory_marginal_no_target_hf | hh | 0.021743 | 0.006349 | -0.000283 |
| block1_memory_marginal_full | lh | 0.026407 | 0.017532 | 0.000274 |
| block1_memory_marginal_full | hl | 0.026034 | 0.025045 | 0.000872 |
| block1_memory_marginal_full | hh | 0.026766 | 0.021610 | 0.000671 |
| block1_memory_marginal_no_target_hf | lh | 0.027712 | -0.008074 | -0.001206 |
| block1_memory_marginal_no_target_hf | hl | 0.026086 | 0.044543 | 0.001796 |
| block1_memory_marginal_no_target_hf | hh | 0.015305 | 0.048120 | 0.001741 |
| block2_memory_marginal_full | lh | 0.031540 | 0.027034 | 0.000717 |
| block2_memory_marginal_full | hl | 0.035110 | 0.060897 | 0.002937 |
| block2_memory_marginal_full | hh | 0.026579 | 0.017108 | 0.000240 |
| block2_memory_marginal_no_target_hf | lh | 0.029834 | 0.034113 | 0.001381 |
| block2_memory_marginal_no_target_hf | hl | 0.033578 | 0.049259 | 0.002055 |
| block2_memory_marginal_no_target_hf | hh | 0.014315 | 0.036296 | 0.000977 |
| block3_memory_marginal_full | lh | 0.035076 | 0.032578 | 0.001356 |
| block3_memory_marginal_full | hl | 0.038022 | 0.025501 | 0.000826 |
| block3_memory_marginal_full | hh | 0.027942 | 0.028539 | 0.001112 |
| block3_memory_marginal_no_target_hf | lh | 0.037646 | 0.034522 | 0.002028 |
| block3_memory_marginal_no_target_hf | hl | 0.036971 | 0.024252 | 0.000830 |
| block3_memory_marginal_no_target_hf | hh | 0.018561 | -0.037223 | -0.001409 |

## Gradient Competition

| loss | variant | path | grad norm | grad/param |
|---|---|---|---:|---:|
| loss_fm_hf_total | full | style_memory | 2.360799e-02 | 1.606657e-03 |
| loss_fm_hf_total | full | style_patch_proj | 5.547265e-02 | 2.653929e-03 |
| loss_fm_hf_total | full | target_hf_subband | 9.017122e-02 | 2.903209e-03 |
| loss_fm_hf_total | full | cross_attn_kv | 6.461583e-02 | 4.850584e-03 |
| loss_fm_hf_total | full | cross_attn_out_gate | 7.857402e-02 | 8.434644e-03 |
| loss_fm_hf_total | full | head_hf | 6.250461e-01 | 4.228172e-02 |
| loss_fm_hf_total | full | input_time | 6.944570e-01 | 8.325415e-02 |
| loss_fm_hf_total | no_style_memory | style_memory | 0.000000e+00 | 0.000000e+00 |
| loss_fm_hf_total | no_style_memory | style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| loss_fm_hf_total | no_style_memory | target_hf_subband | 8.390656e-02 | 2.701508e-03 |
| loss_fm_hf_total | no_style_memory | cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| loss_fm_hf_total | no_style_memory | cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| loss_fm_hf_total | no_style_memory | head_hf | 5.875638e-01 | 3.974621e-02 |
| loss_fm_hf_total | no_style_memory | input_time | 1.030297e+00 | 1.235159e-01 |
| loss_fm_hf_total | no_target_hf | style_memory | 2.037876e-02 | 1.386890e-03 |
| loss_fm_hf_total | no_target_hf | style_patch_proj | 4.119411e-02 | 1.970813e-03 |
| loss_fm_hf_total | no_target_hf | target_hf_subband | 0.000000e+00 | 0.000000e+00 |
| loss_fm_hf_total | no_target_hf | cross_attn_kv | 5.355905e-02 | 4.020573e-03 |
| loss_fm_hf_total | no_target_hf | cross_attn_out_gate | 4.631414e-02 | 4.971660e-03 |
| loss_fm_hf_total | no_target_hf | head_hf | 3.464509e+00 | 2.343594e-01 |
| loss_fm_hf_total | no_target_hf | input_time | 5.437345e-01 | 6.518496e-02 |
| loss_fm_hf_total | backbone_only | style_memory | 0.000000e+00 | 0.000000e+00 |
| loss_fm_hf_total | backbone_only | style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| loss_fm_hf_total | backbone_only | target_hf_subband | 0.000000e+00 | 0.000000e+00 |
| loss_fm_hf_total | backbone_only | cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| loss_fm_hf_total | backbone_only | cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| loss_fm_hf_total | backbone_only | head_hf | 3.306431e+00 | 2.236661e-01 |
| loss_fm_hf_total | backbone_only | input_time | 6.506373e-01 | 7.800087e-02 |
| loss_fm_spectral_lh | full | style_memory | 3.921003e-03 | 2.668464e-04 |
| loss_fm_spectral_lh | full | style_patch_proj | 8.952149e-03 | 4.282897e-04 |
| loss_fm_spectral_lh | full | target_hf_subband | 3.497331e-02 | 1.126022e-03 |
| loss_fm_spectral_lh | full | cross_attn_kv | 1.084276e-02 | 8.139450e-04 |
| loss_fm_spectral_lh | full | cross_attn_out_gate | 1.262932e-02 | 1.355713e-03 |
| loss_fm_spectral_lh | full | head_hf | 2.562642e-01 | 1.733519e-02 |
| loss_fm_spectral_lh | full | input_time | 1.490042e-01 | 1.786319e-02 |
| loss_fm_spectral_lh | no_style_memory | style_memory | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_lh | no_style_memory | style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_lh | no_style_memory | target_hf_subband | 4.327858e-02 | 1.393424e-03 |
| loss_fm_spectral_lh | no_style_memory | cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_lh | no_style_memory | cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_lh | no_style_memory | head_hf | 3.166334e-01 | 2.141891e-02 |
| loss_fm_spectral_lh | no_style_memory | input_time | 1.450287e-01 | 1.738659e-02 |
| loss_fm_spectral_lh | no_target_hf | style_memory | 6.168236e-03 | 4.197834e-04 |
| loss_fm_spectral_lh | no_target_hf | style_patch_proj | 1.545417e-02 | 7.393602e-04 |
| loss_fm_spectral_lh | no_target_hf | target_hf_subband | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_lh | no_target_hf | cross_attn_kv | 2.170998e-02 | 1.629726e-03 |
| loss_fm_spectral_lh | no_target_hf | cross_attn_out_gate | 2.598454e-02 | 2.789349e-03 |
| loss_fm_spectral_lh | no_target_hf | head_hf | 1.059089e+00 | 7.164292e-02 |
| loss_fm_spectral_lh | no_target_hf | input_time | 1.874889e-01 | 2.247688e-02 |
| loss_fm_spectral_lh | backbone_only | style_memory | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_lh | backbone_only | style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_lh | backbone_only | target_hf_subband | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_lh | backbone_only | cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_lh | backbone_only | cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_lh | backbone_only | head_hf | 1.197575e+00 | 8.101091e-02 |
| loss_fm_spectral_lh | backbone_only | input_time | 1.962491e-01 | 2.352708e-02 |
| loss_fm_spectral_hl | full | style_memory | 1.210464e-02 | 8.237891e-04 |
| loss_fm_spectral_hl | full | style_patch_proj | 2.259223e-02 | 1.080860e-03 |
| loss_fm_spectral_hl | full | target_hf_subband | 3.103112e-02 | 9.990974e-04 |
| loss_fm_spectral_hl | full | cross_attn_kv | 2.647433e-02 | 1.987376e-03 |
| loss_fm_spectral_hl | full | cross_attn_out_gate | 5.533623e-02 | 5.940150e-03 |
| loss_fm_spectral_hl | full | head_hf | 2.965721e-01 | 2.006185e-02 |
| loss_fm_spectral_hl | full | input_time | 5.335254e-01 | 6.396105e-02 |
| loss_fm_spectral_hl | no_style_memory | style_memory | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hl | no_style_memory | style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hl | no_style_memory | target_hf_subband | 3.221486e-02 | 1.037210e-03 |
| loss_fm_spectral_hl | no_style_memory | cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hl | no_style_memory | cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hl | no_style_memory | head_hf | 3.033983e-01 | 2.052361e-02 |
| loss_fm_spectral_hl | no_style_memory | input_time | 5.323326e-01 | 6.381806e-02 |
| loss_fm_spectral_hl | no_target_hf | style_memory | 1.037396e-02 | 7.060067e-04 |
| loss_fm_spectral_hl | no_target_hf | style_patch_proj | 2.071797e-02 | 9.911915e-04 |
| loss_fm_spectral_hl | no_target_hf | target_hf_subband | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hl | no_target_hf | cross_attn_kv | 2.200804e-02 | 1.652101e-03 |
| loss_fm_spectral_hl | no_target_hf | cross_attn_out_gate | 6.160237e-02 | 6.612798e-03 |
| loss_fm_spectral_hl | no_target_hf | head_hf | 7.470308e-01 | 5.053347e-02 |
| loss_fm_spectral_hl | no_target_hf | input_time | 4.975467e-01 | 5.964779e-02 |
| loss_fm_spectral_hl | backbone_only | style_memory | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hl | backbone_only | style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hl | backbone_only | target_hf_subband | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hl | backbone_only | cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hl | backbone_only | cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hl | backbone_only | head_hf | 7.730650e-01 | 5.229458e-02 |
| loss_fm_spectral_hl | backbone_only | input_time | 4.905974e-01 | 5.881468e-02 |
| loss_fm_spectral_hh | full | style_memory | 1.344062e-02 | 9.147103e-04 |
| loss_fm_spectral_hh | full | style_patch_proj | 2.990594e-02 | 1.430764e-03 |
| loss_fm_spectral_hh | full | target_hf_subband | 6.810627e-02 | 2.192792e-03 |
| loss_fm_spectral_hh | full | cross_attn_kv | 3.192270e-02 | 2.396375e-03 |
| loss_fm_spectral_hh | full | cross_attn_out_gate | 8.618827e-02 | 9.252007e-03 |
| loss_fm_spectral_hh | full | head_hf | 4.074606e-01 | 2.756298e-02 |
| loss_fm_spectral_hh | full | input_time | 5.409292e-01 | 6.484865e-02 |
| loss_fm_spectral_hh | no_style_memory | style_memory | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hh | no_style_memory | style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hh | no_style_memory | target_hf_subband | 6.426101e-02 | 2.068988e-03 |
| loss_fm_spectral_hh | no_style_memory | cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hh | no_style_memory | cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hh | no_style_memory | head_hf | 3.910550e-01 | 2.645322e-02 |
| loss_fm_spectral_hh | no_style_memory | input_time | 4.923775e-01 | 5.902809e-02 |
| loss_fm_spectral_hh | no_target_hf | style_memory | 3.994162e-03 | 2.718253e-04 |
| loss_fm_spectral_hh | no_target_hf | style_patch_proj | 8.922410e-03 | 4.268670e-04 |
| loss_fm_spectral_hh | no_target_hf | target_hf_subband | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hh | no_target_hf | cross_attn_kv | 1.115285e-02 | 8.372230e-04 |
| loss_fm_spectral_hh | no_target_hf | cross_attn_out_gate | 1.593744e-02 | 1.710828e-03 |
| loss_fm_spectral_hh | no_target_hf | head_hf | 3.010515e+00 | 2.036486e-01 |
| loss_fm_spectral_hh | no_target_hf | input_time | 2.122679e-01 | 2.544748e-02 |
| loss_fm_spectral_hh | backbone_only | style_memory | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hh | backbone_only | style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hh | backbone_only | target_hf_subband | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hh | backbone_only | cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hh | backbone_only | cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hh | backbone_only | head_hf | 2.983399e+00 | 2.018143e-01 |
| loss_fm_spectral_hh | backbone_only | input_time | 2.164174e-01 | 2.594494e-02 |
