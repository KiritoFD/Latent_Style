# Destructive 7-Epoch Ablation Summary

| ID | Label | Status | Eval | LPIPS down | CLIP-style up | CLIP-content up | Train sec | Eval sec |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `D0_full_correct_7ep` | Full control from corrected config | ok | ok | 0.4528 | 0.7129 | 0.8065 | 290.6500 | 60.0340 |
| `D1_no_terminal_swd` | w/o terminal SWD | ok | ok | 0.2976 | 0.6654 | 0.9037 | 295.9350 | 63.6300 |
| `D2_no_kinetic` | w/o kinetic | ok | ok | 0.6325 | 0.7225 | 0.6608 | 303.3120 | 64.4230 |
| `D3_no_swd_no_kinetic` | w/o SWD and kinetic | ok | ok | 0.3594 | 0.6898 | 0.8582 | 306.5500 | 60.5670 |
| `D4_conv_body_no_global_attn` | conv body, no global attention | ok | ok | 0.4529 | 0.7128 | 0.8065 | 295.4980 | 78.7710 |
| `D5_disable_skip_routing` | disable routed skip path | ok | ok | 0.4513 | 0.7064 | 0.8147 | 294.6090 | 61.9950 |
| `D6_disable_spatial_prior` | disable spatial style prior | ok | ok | 0.4524 | 0.7130 | 0.8075 | 305.7830 | 61.8030 |
| `D7_no_residual_path` | no residual path | ok | ok | 0.4525 | 0.7130 | 0.8073 | 304.2510 | 61.8870 |
| `D8_strong_color_loss` | strong color loss | ok | ok | 0.5677 | 0.6963 | 0.6625 | 308.5880 | 60.9870 |
| `D9_l2_ot_cost` | L2 matching cost | ok | ok | 0.4523 | 0.7131 | 0.8066 | 311.0980 | 59.3230 |
| `D10_micro_hf_swd_trap` | micro high-frequency SWD | ok | ok | 0.4671 | 0.7024 | 0.7832 | 302.1190 | 59.3990 |
| `D11_single_terminal_step` | single terminal step | ok | ok | 0.4518 | 0.7129 | 0.8078 | 298.5650 | 59.9040 |
