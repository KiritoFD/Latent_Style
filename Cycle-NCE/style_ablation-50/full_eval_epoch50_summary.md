# style_ablation-50 epoch_0050 full_eval summary

| exp_name | status | transfer_clip | transfer_cls | transfer_lpips | p2a_clip | p2a_cls |
|---|---:|---:|---:|---:|---:|---:|
| baseline_50e | ok | 0.5171 | 0.1600 | 0.5117 | 0.5511 | 0.2000 |
| dyn_single_step_only | ok | 0.5053 | 0.1200 | 0.4498 | 0.5397 | 0.1400 |
| dyn_style_strength_jitter | ok | 0.5249 | 0.1500 | 0.5462 | 0.5501 | 0.2000 |
| inj_no_decoder_adagn | ok | 0.5123 | 0.1100 | 0.4950 | 0.5432 | 0.1200 |
| inj_no_decoder_spatial | ok | 0.5172 | 0.1800 | 0.5150 | 0.5478 | 0.2600 |
| inj_no_delta_gate | ok | 0.5168 | 0.1700 | 0.5096 | 0.5512 | 0.2200 |
| inj_no_delta_highpass | ok | 0.5162 | 0.1500 | 0.4987 | 0.5497 | 0.2000 |
| inj_no_output_affine | ok | 0.5172 | 0.1600 | 0.5117 | 0.5510 | 0.2000 |
| inj_no_texture_gain | ok | 0.5169 | 0.1400 | 0.5128 | 0.5592 | 0.2000 |
| inj_spatial_blur_off | ok | 0.5295 | 0.2200 | 0.5574 | 0.5424 | 0.3400 |
| inj_spatial_highpass_on | ok | 0.5108 | 0.1200 | 0.5113 | 0.5462 | 0.1400 |
| inj_spatial_map_norm_off | ok | 0.5093 | 0.1500 | 0.4884 | 0.5542 | 0.2200 |
| loss_no_code | ok | 0.5174 | 0.1700 | 0.5116 | 0.5514 | 0.2200 |
| loss_no_color_moment | ok | 0.5139 | 0.1000 | 0.4401 | 0.5495 | 0.1200 |
| loss_no_cycle | ok | 0.5178 | 0.1600 | 0.5117 | 0.5517 | 0.2000 |
| loss_no_distill | ok | 0.5174 | 0.1600 | 0.5117 | 0.5510 | 0.2000 |
| loss_no_nce | ok | 0.5174 | 0.1600 | 0.5116 | 0.5516 | 0.2000 |
| loss_no_push | ok | 0.5175 | 0.1600 | 0.5116 | 0.5512 | 0.2000 |
| loss_no_semigroup | ok | 0.5172 | 0.1600 | 0.5116 | 0.5515 | 0.2000 |
| loss_no_stroke_gram | ok | 0.4585 | 0.0500 | 0.3294 | 0.4965 | 0.0400 |
| loss_no_struct_edge | ok | 0.5432 | 0.2500 | 0.6088 | 0.5571 | 0.4000 |
| loss_no_style_spatial_tv | ok | 0.5175 | 0.1600 | 0.5117 | 0.5514 | 0.2000 |
