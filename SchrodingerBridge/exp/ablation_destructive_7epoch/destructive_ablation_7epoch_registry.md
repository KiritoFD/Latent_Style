# Destructive 7-Epoch Ablation Registry

Base config: `G:\GitHub\Latent_Style\SchrodingerBridge\S-add__K-1_C-0_W-20_Col-0\config.json`

| ID | Label | Train | Train sec | Eval | Eval sec | Purpose |
| --- | --- | --- | ---: | --- | ---: | --- |
| `D0_full_correct_7ep` | Full control from corrected config | ok | 290.650 | ok | 60.034 | 7-epoch control using S-add__K-1_C-0_W-20_Col-0/config.json without model/loss changes. |
| `D1_no_terminal_swd` | w/o terminal SWD | ok | 295.935 | ok | 63.630 | Destructive removal of endpoint style-distribution matching. |
| `D2_no_kinetic` | w/o kinetic | ok | 303.312 | ok | 64.423 | Destructive removal of flow regularization/content-stability pressure. |
| `D3_no_swd_no_kinetic` | w/o SWD and kinetic | ok | 306.550 | ok | 60.567 | Strong negative control: remove both style distribution endpoint and trajectory regularization. |
| `D4_conv_body_no_global_attn` | conv body, no global attention | ok | 295.498 | ok | 78.771 | Destructive architecture ablation replacing the global-attention body with convolutional blocks. |
| `D5_disable_skip_routing` | disable routed skip path | ok | 294.609 | ok | 61.995 | Destructive removal of the routed skip pathway that carries clean structure. |
| `D6_disable_spatial_prior` | disable spatial style prior | ok | 305.783 | ok | 61.803 | Destructive removal of the spatial prior used by the style-conditioned model. |
| `D7_no_residual_path` | no residual path | ok | 304.251 | ok | 61.887 | Destructive model ablation disabling the learned residual update path. |
| `D8_strong_color_loss` | strong color loss | ok | 308.588 | ok | 60.987 | Strong negative control for naive color matching that previously harmed content. |
| `D9_l2_ot_cost` | L2 matching cost | ok | 311.098 | ok | 59.323 | Replace SWD-based matching cost with global latent L2 cost. |
| `D10_micro_hf_swd_trap` | micro high-frequency SWD | ok | 302.119 | ok | 59.399 | Stress test: force SWD toward tiny high-frequency patches to expose grain/noise tendencies. |
| `D11_single_terminal_step` | single terminal step | ok | 298.565 | ok | 59.904 | Collapse endpoint matching from four terminal steps to one to test endpoint optimization strength. |
