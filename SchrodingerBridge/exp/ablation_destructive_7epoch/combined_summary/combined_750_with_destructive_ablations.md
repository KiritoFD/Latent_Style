# Combined 750 Summary With Destructive Ablations

Notes:

- `Ours` is replaced by `D0_full_correct_7ep` as requested.
- Artifact-pack columns are filled only where that expensive metric pack has actually been run.
- All ablation rows have strict 750 images plus base/guard/HF-KID/plain-KID coverage.

## Main + Ablation Table

| Group | Method | Run | Images | LPIPS↓ | CLIP-S↑ | CLIP-C↑ | SSIM-Y↑ | Edge-F1↑ | ExtraEdge↓ | Chroma-Z↓ | HF-KID↓ | KID↓ | Train sec |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ablation | Ours D0 full | `D0_full_correct_7ep` | 750 | 0.4593 | 0.7014 | 0.8022 | 0.4542 | 0.3090 | 0.0976 | -0.5035 | 4.184545 | 0.054214 | 290.6 |
| baseline | SaMST strict | `samst_strict` | 750 | 0.4664 | 0.7194 | 0.8193 | 0.6520 | 0.5162 | 0.1077 | 0.2636 | 6.759762 | 0.048909 |  |
| baseline | StyleID strict | `styleid_strict` | 750 | 0.7497 | 0.7597 | 0.5519 | 0.1466 | 0.1954 | 0.2886 | 0.7909 |  |  |  |
| baseline | S2WAT strict | `s2wat_strict` | 750 | 0.5263 | 0.7139 | 0.7465 | 0.5070 | 0.2574 | 0.0776 | -0.4523 | 12.662270 | 0.056735 |  |
| baseline | AdaIN v32k | `adain_v32k` | 750 | 0.6298 | 0.7130 | 0.6990 | 0.3246 | 0.0167 | 0.0071 | 1.1066 |  |  |  |
| baseline | AdaIN vgg19 | `adain_vgg19` | 750 | 0.6870 | 0.6930 | 0.5991 | 0.2897 | 0.0182 | 0.0197 | 0.3528 |  |  |  |
| baseline | AdaIN bad | `adain_bad` | 750 | 0.8490 | 0.6308 | 0.5297 | 0.2868 | 0.0000 | 0.0000 | 410.2368 |  |  |  |
| ablation | w/o terminal SWD | `D1_no_terminal_swd` | 750 | 0.3490 | 0.6708 | 0.8989 | 0.4729 | 0.1824 | 0.0000 | -0.4394 | 5.843525 | 0.041960 | 295.9 |
| ablation | w/o kinetic | `D2_no_kinetic` | 750 | 0.6375 | 0.7159 | 0.6624 | 0.2687 | 0.2492 | 0.1527 | -0.0002 | 4.776494 | 0.126270 | 303.3 |
| ablation | w/o SWD and kinetic | `D3_no_swd_no_kinetic` | 750 | 0.3938 | 0.6884 | 0.8527 | 0.4368 | 0.1694 | 0.0021 | -0.3180 | 6.745173 | 0.049534 | 306.6 |
| ablation | conv body, no global attention | `D4_conv_body_no_global_attn` | 750 | 0.4594 | 0.7022 | 0.8020 | 0.4543 | 0.3091 | 0.0978 | -0.5031 | 4.179079 | 0.054530 | 295.5 |
| ablation | disable routed skip path | `D5_disable_skip_routing` | 750 | 0.4727 | 0.6951 | 0.8057 | 0.4613 | 0.2975 | 0.0670 | -0.3204 | 4.985016 | 0.062353 | 294.6 |
| ablation | disable spatial style prior | `D6_disable_spatial_prior` | 750 | 0.4589 | 0.7022 | 0.8033 | 0.4548 | 0.3095 | 0.0971 | -0.5018 | 4.194194 | 0.054133 | 305.8 |
| ablation | no residual path | `D7_no_residual_path` | 750 | 0.4592 | 0.7013 | 0.8025 | 0.4547 | 0.3093 | 0.0972 | -0.5061 | 4.176114 | 0.054496 | 304.3 |
| ablation | strong color loss | `D8_strong_color_loss` | 750 | 0.5675 | 0.6923 | 0.6629 | 0.3401 | 0.2769 | 0.1966 | -0.7679 | 5.679315 | 0.148481 | 308.6 |
| ablation | L2 matching cost | `D9_l2_ot_cost` | 750 | 0.4589 | 0.7016 | 0.8021 | 0.4550 | 0.3090 | 0.0969 | -0.4986 | 4.191200 | 0.054227 | 311.1 |
| ablation | micro high-frequency SWD | `D10_micro_hf_swd_trap` | 750 | 0.4863 | 0.6989 | 0.7772 | 0.3883 | 0.1925 | 0.0452 | -0.3699 | 5.710126 | 0.089156 | 302.1 |
| ablation | single terminal step | `D11_single_terminal_step` | 750 | 0.4585 | 0.7012 | 0.8032 | 0.4550 | 0.3096 | 0.0965 | -0.5014 | 4.172345 | 0.054519 | 298.6 |

## Ablation Purposes

| Run | Label | Purpose |
| --- | --- | --- |
| `D0_full_correct_7ep` | Full control from corrected config | 7-epoch control using S-add__K-1_C-0_W-20_Col-0/config.json without model/loss changes. |
| `D1_no_terminal_swd` | w/o terminal SWD | Destructive removal of endpoint style-distribution matching. |
| `D2_no_kinetic` | w/o kinetic | Destructive removal of flow regularization/content-stability pressure. |
| `D3_no_swd_no_kinetic` | w/o SWD and kinetic | Strong negative control: remove both style distribution endpoint and trajectory regularization. |
| `D4_conv_body_no_global_attn` | conv body, no global attention | Destructive architecture ablation replacing the global-attention body with convolutional blocks. |
| `D5_disable_skip_routing` | disable routed skip path | Destructive removal of the routed skip pathway that carries clean structure. |
| `D6_disable_spatial_prior` | disable spatial style prior | Destructive removal of the spatial prior used by the style-conditioned model. |
| `D7_no_residual_path` | no residual path | Destructive model ablation disabling the learned residual update path. |
| `D8_strong_color_loss` | strong color loss | Strong negative control for naive color matching that previously harmed content. |
| `D9_l2_ot_cost` | L2 matching cost | Replace SWD-based matching cost with global latent L2 cost. |
| `D10_micro_hf_swd_trap` | micro high-frequency SWD | Stress test: force SWD toward tiny high-frequency patches to expose grain/noise tendencies. |
| `D11_single_terminal_step` | single terminal step | Collapse endpoint matching from four terminal steps to one to test endpoint optimization strength. |
