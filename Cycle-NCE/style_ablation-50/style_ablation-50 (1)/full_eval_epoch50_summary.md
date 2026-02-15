# style_ablation-50 epoch_0050 full_eval summary

| exp_name | status | transfer_clip | transfer_cls | transfer_lpips | p2a_clip | p2a_cls |
|---|---:|---:|---:|---:|---:|---:|
| baseline_50e | ok | 0.4995 | 0.1200 | 0.4242 | 0.5443 | 0.1600 |
| bundle_all_style_paths_off | ok | 0.4414 | 0.0600 | 0.2439 | 0.4737 | 0.0400 |
| bundle_content_guard_relaxed | ok | 0.5057 | 0.1600 | 0.4460 | 0.5426 | 0.2200 |
| bundle_content_guard_strict | ok | 0.4927 | 0.1300 | 0.4033 | 0.5418 | 0.1600 |
| bundle_style_down | ok | 0.5080 | 0.1100 | 0.4715 | 0.5464 | 0.1200 |
| bundle_style_losses_off | ok | 0.4414 | 0.0600 | 0.2439 | 0.4737 | 0.0400 |
| bundle_style_up | ok | 0.5093 | 0.1600 | 0.4557 | 0.5436 | 0.2200 |
| dyn_flat_step_schedule | ok | 0.4991 | 0.1100 | 0.4244 | 0.5401 | 0.1200 |
| dyn_single_step_only | ok | 0.4878 | 0.0900 | 0.3537 | 0.5341 | 0.0800 |
| dyn_style_strength_jitter | ok | 0.5115 | 0.1600 | 0.4783 | 0.5442 | 0.2200 |
| inj_no_decoder_adagn | ok | 0.4941 | 0.1500 | 0.4029 | 0.5384 | 0.2000 |
| inj_no_decoder_spatial | ok | 0.4998 | 0.1200 | 0.4280 | 0.5406 | 0.1600 |
| inj_no_delta_gate | ok | 0.4987 | 0.1100 | 0.4209 | 0.5434 | 0.1400 |
| inj_no_delta_highpass | ok | 0.4991 | 0.1100 | 0.4111 | 0.5445 | 0.1400 |
| inj_no_output_affine | ok | 0.4998 | 0.1200 | 0.4241 | 0.5442 | 0.1600 |
| inj_no_texture_gain | ok | 0.5038 | 0.1300 | 0.4438 | 0.5529 | 0.1800 |
| inj_spatial_blur_off | ok | 0.5062 | 0.1400 | 0.4461 | 0.5423 | 0.1800 |
| inj_spatial_highpass_on | ok | 0.4950 | 0.1300 | 0.4212 | 0.5410 | 0.1800 |
| inj_spatial_map_norm_off | ok | 0.4930 | 0.1200 | 0.4162 | 0.5436 | 0.1600 |
| loss_no_code | ok | 0.4991 | 0.1200 | 0.4242 | 0.5436 | 0.1600 |
| loss_no_color_moment | ok | 0.4929 | 0.0800 | 0.3642 | 0.5373 | 0.0800 |
| loss_no_cycle | ok | 0.4990 | 0.1200 | 0.4241 | 0.5432 | 0.1600 |
| loss_no_distill | ok | 0.4996 | 0.1200 | 0.4242 | 0.5439 | 0.1600 |
| loss_no_nce | ok | 0.4994 | 0.1200 | 0.4242 | 0.5437 | 0.1600 |
| loss_no_push | ok | 0.4993 | 0.1200 | 0.4241 | 0.5438 | 0.1600 |
| loss_no_semigroup | ok | 0.4999 | 0.1200 | 0.4242 | 0.5443 | 0.1600 |
| loss_no_stroke_gram | ok | 0.4501 | 0.0500 | 0.2938 | 0.4877 | 0.0400 |
| loss_no_struct_edge | ok | 0.5207 | 0.2000 | 0.5049 | 0.5471 | 0.3000 |
| loss_no_style_spatial_tv | ok | 0.4994 | 0.1200 | 0.4241 | 0.5438 | 0.1600 |
