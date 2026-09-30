# Route Competition Probe

Config: `configs\exp_probe_target_hf_subband_ft6.json`
Checkpoint: `exp\model_probe\target_hf_subband_ft6\epoch_0006.pt`
Device: `cuda`
Load info: `{'missing': 0, 'unexpected': 0}`

## Reading

Mean HF cos(memory, desired)=0.1599; cos(target-HF, desired)=0.1498; cos(target-HF | memory, desired)=0.1555; full MSE improvement=0.0576. Interpretation: target-HF remains at least as aligned after style memory is present.

## Route Transitions

| transition | band | delta/desired | cos(delta, desired) | projection | orthogonal fraction | MSE improvement |
|---|---|---:|---:|---:|---:|---:|
| backbone_to_style_memory | lh | 0.170091 | 0.166527 | 0.032468 | 0.982729 | 0.030772 |
| backbone_to_style_memory | hl | 0.161520 | 0.156540 | 0.028310 | 0.984290 | 0.026151 |
| backbone_to_style_memory | hh | 0.099817 | 0.156580 | 0.016698 | 0.984849 | 0.021892 |
| backbone_to_target_hf | lh | 0.096810 | 0.109288 | 0.010650 | 0.993639 | 0.011757 |
| backbone_to_target_hf | hl | 0.080585 | 0.074424 | 0.006120 | 0.996414 | 0.005709 |
| backbone_to_target_hf | hh | 0.243400 | 0.265614 | 0.069114 | 0.959679 | 0.074404 |
| style_memory_to_full_target_hf_marginal | lh | 0.109306 | 0.110100 | 0.012362 | 0.993538 | 0.012230 |
| style_memory_to_full_target_hf_marginal | hl | 0.093328 | 0.088112 | 0.008456 | 0.995649 | 0.007905 |
| style_memory_to_full_target_hf_marginal | hh | 0.282244 | 0.268146 | 0.082821 | 0.958019 | 0.076507 |
| target_hf_to_full_style_memory_marginal | lh | 0.171579 | 0.167363 | 0.033094 | 0.982423 | 0.031230 |
| target_hf_to_full_style_memory_marginal | hl | 0.169232 | 0.165758 | 0.030914 | 0.983165 | 0.028313 |
| target_hf_to_full_style_memory_marginal | hh | 0.162027 | 0.149822 | 0.027844 | 0.985019 | 0.024150 |
| backbone_to_full | lh | 0.202528 | 0.202409 | 0.044317 | 0.976800 | 0.042630 |
| backbone_to_full | hl | 0.185316 | 0.179782 | 0.036653 | 0.980398 | 0.033843 |
| backbone_to_full | hh | 0.313741 | 0.299678 | 0.102369 | 0.947223 | 0.096409 |

## Route Interaction

| name | band | interaction/desired | cos(interaction, desired) | projection |
|---|---|---:|---:|---:|
| route_interaction | lh | 0.042728 | 0.031206 | 0.001199 |
| route_interaction | hl | 0.036453 | 0.057925 | 0.002223 |
| route_interaction | hh | 0.125026 | 0.112042 | 0.016558 |

## Per-Block Cross-Attention Marginals

| block transition | band | delta/desired | cos(delta, desired) | MSE improvement |
|---|---|---:|---:|---:|
| block0_memory_marginal_full | lh | 0.045495 | 0.040925 | 0.001839 |
| block0_memory_marginal_full | hl | 0.044446 | 0.049034 | 0.002246 |
| block0_memory_marginal_full | hh | 0.041038 | 0.032892 | 0.001393 |
| block0_memory_marginal_no_target_hf | lh | 0.045378 | 0.043891 | 0.002157 |
| block0_memory_marginal_no_target_hf | hl | 0.043392 | 0.044523 | 0.001964 |
| block0_memory_marginal_no_target_hf | hh | 0.026172 | 0.073773 | 0.002567 |
| block1_memory_marginal_full | lh | 0.070726 | 0.067941 | 0.005003 |
| block1_memory_marginal_full | hl | 0.063331 | 0.071793 | 0.005358 |
| block1_memory_marginal_full | hh | 0.072546 | 0.074082 | 0.005909 |
| block1_memory_marginal_no_target_hf | lh | 0.072331 | 0.039326 | 0.000825 |
| block1_memory_marginal_no_target_hf | hl | 0.063771 | 0.091936 | 0.007917 |
| block1_memory_marginal_no_target_hf | hh | 0.045293 | 0.130422 | 0.010628 |
| block2_memory_marginal_full | lh | 0.087173 | 0.084404 | 0.008028 |
| block2_memory_marginal_full | hl | 0.079794 | 0.089836 | 0.008551 |
| block2_memory_marginal_full | hh | 0.082390 | 0.074071 | 0.006864 |
| block2_memory_marginal_no_target_hf | lh | 0.081408 | 0.090368 | 0.009859 |
| block2_memory_marginal_no_target_hf | hl | 0.074675 | 0.077576 | 0.006966 |
| block2_memory_marginal_no_target_hf | hh | 0.044762 | 0.093173 | 0.007724 |
| block3_memory_marginal_full | lh | 0.105663 | 0.098527 | 0.016489 |
| block3_memory_marginal_full | hl | 0.095936 | 0.079417 | 0.007653 |
| block3_memory_marginal_full | hh | 0.078629 | 0.065450 | 0.007128 |
| block3_memory_marginal_no_target_hf | lh | 0.108838 | 0.100099 | 0.018727 |
| block3_memory_marginal_no_target_hf | hl | 0.089844 | 0.070256 | 0.006243 |
| block3_memory_marginal_no_target_hf | hh | 0.046694 | 0.011550 | -0.000085 |

## Gradient Competition

| loss | variant | path | grad norm | grad/param |
|---|---|---|---:|---:|
| loss_fm_hf_total | full | style_memory | 4.533568e-02 | 3.083356e-03 |
| loss_fm_hf_total | full | style_patch_proj | 7.904833e-02 | 3.778683e-03 |
| loss_fm_hf_total | full | target_hf_subband | 8.182503e-02 | 2.635290e-03 |
| loss_fm_hf_total | full | cross_attn_kv | 8.342451e-02 | 6.262466e-03 |
| loss_fm_hf_total | full | cross_attn_out_gate | 2.127772e-01 | 2.284518e-02 |
| loss_fm_hf_total | full | head_hf | 5.738021e-01 | 3.883521e-02 |
| loss_fm_hf_total | full | input_time | 5.907107e-01 | 7.077019e-02 |
| loss_fm_hf_total | no_style_memory | style_memory | 0.000000e+00 | 0.000000e+00 |
| loss_fm_hf_total | no_style_memory | style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| loss_fm_hf_total | no_style_memory | target_hf_subband | 1.382714e-01 | 4.453224e-03 |
| loss_fm_hf_total | no_style_memory | cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| loss_fm_hf_total | no_style_memory | cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| loss_fm_hf_total | no_style_memory | head_hf | 1.105826e+00 | 7.484287e-02 |
| loss_fm_hf_total | no_style_memory | input_time | 6.065731e-01 | 7.267058e-02 |
| loss_fm_hf_total | no_target_hf | style_memory | 4.119730e-02 | 2.801898e-03 |
| loss_fm_hf_total | no_target_hf | style_patch_proj | 9.353986e-02 | 4.471410e-03 |
| loss_fm_hf_total | no_target_hf | target_hf_subband | 0.000000e+00 | 0.000000e+00 |
| loss_fm_hf_total | no_target_hf | cross_attn_kv | 1.306521e-01 | 9.807718e-03 |
| loss_fm_hf_total | no_target_hf | cross_attn_out_gate | 2.205599e-01 | 2.368078e-02 |
| loss_fm_hf_total | no_target_hf | head_hf | 3.057474e+00 | 2.069314e-01 |
| loss_fm_hf_total | no_target_hf | input_time | 4.849133e-01 | 5.809511e-02 |
| loss_fm_hf_total | backbone_only | style_memory | 0.000000e+00 | 0.000000e+00 |
| loss_fm_hf_total | backbone_only | style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| loss_fm_hf_total | backbone_only | target_hf_subband | 0.000000e+00 | 0.000000e+00 |
| loss_fm_hf_total | backbone_only | cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| loss_fm_hf_total | backbone_only | cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| loss_fm_hf_total | backbone_only | head_hf | 3.540912e+00 | 2.396507e-01 |
| loss_fm_hf_total | backbone_only | input_time | 3.997825e-01 | 4.789601e-02 |
| loss_fm_spectral_lh | full | style_memory | 1.061164e-02 | 7.217152e-04 |
| loss_fm_spectral_lh | full | style_patch_proj | 1.827076e-02 | 8.733821e-04 |
| loss_fm_spectral_lh | full | target_hf_subband | 3.865545e-02 | 1.244953e-03 |
| loss_fm_spectral_lh | full | cross_attn_kv | 2.252429e-02 | 1.690841e-03 |
| loss_fm_spectral_lh | full | cross_attn_out_gate | 2.782495e-02 | 2.987472e-03 |
| loss_fm_spectral_lh | full | head_hf | 2.977280e-01 | 2.015038e-02 |
| loss_fm_spectral_lh | full | input_time | 1.690077e-01 | 2.024800e-02 |
| loss_fm_spectral_lh | no_style_memory | style_memory | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_lh | no_style_memory | style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_lh | no_style_memory | target_hf_subband | 8.620105e-02 | 2.776226e-03 |
| loss_fm_spectral_lh | no_style_memory | cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_lh | no_style_memory | cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_lh | no_style_memory | head_hf | 6.644517e-01 | 4.497043e-02 |
| loss_fm_spectral_lh | no_style_memory | input_time | 1.341386e-01 | 1.607050e-02 |
| loss_fm_spectral_lh | no_target_hf | style_memory | 1.130577e-02 | 7.689243e-04 |
| loss_fm_spectral_lh | no_target_hf | style_patch_proj | 3.796577e-02 | 1.814847e-03 |
| loss_fm_spectral_lh | no_target_hf | target_hf_subband | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_lh | no_target_hf | cross_attn_kv | 6.028229e-02 | 4.525238e-03 |
| loss_fm_spectral_lh | no_target_hf | cross_attn_out_gate | 6.429634e-02 | 6.903283e-03 |
| loss_fm_spectral_lh | no_target_hf | head_hf | 1.097807e+00 | 7.430015e-02 |
| loss_fm_spectral_lh | no_target_hf | input_time | 2.156618e-01 | 2.583739e-02 |
| loss_fm_spectral_lh | backbone_only | style_memory | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_lh | backbone_only | style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_lh | backbone_only | target_hf_subband | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_lh | backbone_only | cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_lh | backbone_only | cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_lh | backbone_only | head_hf | 1.381483e+00 | 9.349947e-02 |
| loss_fm_spectral_lh | backbone_only | input_time | 1.722173e-01 | 2.063253e-02 |
| loss_fm_spectral_hl | full | style_memory | 1.938412e-02 | 1.318346e-03 |
| loss_fm_spectral_hl | full | style_patch_proj | 3.560864e-02 | 1.702171e-03 |
| loss_fm_spectral_hl | full | target_hf_subband | 3.108727e-02 | 1.001209e-03 |
| loss_fm_spectral_hl | full | cross_attn_kv | 4.342935e-02 | 3.260131e-03 |
| loss_fm_spectral_hl | full | cross_attn_out_gate | 8.489278e-02 | 9.114654e-03 |
| loss_fm_spectral_hl | full | head_hf | 3.191353e-01 | 2.159924e-02 |
| loss_fm_spectral_hl | full | input_time | 3.924515e-01 | 4.701771e-02 |
| loss_fm_spectral_hl | no_style_memory | style_memory | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hl | no_style_memory | style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hl | no_style_memory | target_hf_subband | 8.296756e-02 | 2.672087e-03 |
| loss_fm_spectral_hl | no_style_memory | cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hl | no_style_memory | cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hl | no_style_memory | head_hf | 7.731930e-01 | 5.233009e-02 |
| loss_fm_spectral_hl | no_style_memory | input_time | 3.321591e-01 | 3.979438e-02 |
| loss_fm_spectral_hl | no_target_hf | style_memory | 1.965703e-02 | 1.336907e-03 |
| loss_fm_spectral_hl | no_target_hf | style_patch_proj | 3.600555e-02 | 1.721144e-03 |
| loss_fm_spectral_hl | no_target_hf | target_hf_subband | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hl | no_target_hf | cross_attn_kv | 4.027244e-02 | 3.023150e-03 |
| loss_fm_spectral_hl | no_target_hf | cross_attn_out_gate | 7.179771e-02 | 7.708681e-03 |
| loss_fm_spectral_hl | no_target_hf | head_hf | 7.742871e-01 | 5.240414e-02 |
| loss_fm_spectral_hl | no_target_hf | input_time | 3.742716e-01 | 4.483967e-02 |
| loss_fm_spectral_hl | backbone_only | style_memory | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hl | backbone_only | style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hl | backbone_only | target_hf_subband | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hl | backbone_only | cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hl | backbone_only | cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hl | backbone_only | head_hf | 5.129100e-01 | 3.471401e-02 |
| loss_fm_spectral_hl | backbone_only | input_time | 3.071350e-01 | 3.679636e-02 |
| loss_fm_spectral_hh | full | style_memory | 2.111294e-02 | 1.435926e-03 |
| loss_fm_spectral_hh | full | style_patch_proj | 4.406427e-02 | 2.106368e-03 |
| loss_fm_spectral_hh | full | target_hf_subband | 6.507437e-02 | 2.095812e-03 |
| loss_fm_spectral_hh | full | cross_attn_kv | 4.446274e-02 | 3.337705e-03 |
| loss_fm_spectral_hh | full | cross_attn_out_gate | 1.196447e-01 | 1.284586e-02 |
| loss_fm_spectral_hh | full | head_hf | 3.725044e-01 | 2.521129e-02 |
| loss_fm_spectral_hh | full | input_time | 2.608170e-01 | 3.124722e-02 |
| loss_fm_spectral_hh | no_style_memory | style_memory | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hh | no_style_memory | style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hh | no_style_memory | target_hf_subband | 6.931616e-02 | 2.232425e-03 |
| loss_fm_spectral_hh | no_style_memory | cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hh | no_style_memory | cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hh | no_style_memory | head_hf | 4.284016e-01 | 2.899444e-02 |
| loss_fm_spectral_hh | no_style_memory | input_time | 3.027348e-01 | 3.626919e-02 |
| loss_fm_spectral_hh | no_target_hf | style_memory | 2.107586e-02 | 1.433404e-03 |
| loss_fm_spectral_hh | no_target_hf | style_patch_proj | 6.269480e-02 | 2.996949e-03 |
| loss_fm_spectral_hh | no_target_hf | target_hf_subband | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hh | no_target_hf | cross_attn_kv | 8.540826e-02 | 6.411381e-03 |
| loss_fm_spectral_hh | no_target_hf | cross_attn_out_gate | 1.355488e-01 | 1.455342e-02 |
| loss_fm_spectral_hh | no_target_hf | head_hf | 2.746533e+00 | 1.858868e-01 |
| loss_fm_spectral_hh | no_target_hf | input_time | 1.843553e-01 | 2.208672e-02 |
| loss_fm_spectral_hh | backbone_only | style_memory | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hh | backbone_only | style_patch_proj | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hh | backbone_only | target_hf_subband | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hh | backbone_only | cross_attn_kv | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hh | backbone_only | cross_attn_out_gate | 0.000000e+00 | 0.000000e+00 |
| loss_fm_spectral_hh | backbone_only | head_hf | 3.219703e+00 | 2.179111e-01 |
| loss_fm_spectral_hh | backbone_only | input_time | 1.596101e-01 | 1.912211e-02 |
