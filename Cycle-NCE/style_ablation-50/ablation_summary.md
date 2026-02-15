# Style Ablation Summary

<<<<<<< Updated upstream
- Generated: 2026-02-14 12:26:12
=======
- Generated: 2026-02-14 16:29:47
>>>>>>> Stashed changes
- Mode: `all`
- Target epochs per run: `50`

| Variant | Category | Status | Latest transfer | Best transfer | Best epoch | Latest p2a | Last train loss | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| baseline_50e | baseline | skipped_existing | - | - | - | - | - | 50-epoch short-run baseline (legacy warmup keys removed). |
| inj_no_decoder_spatial | inj_single | skipped_existing | - | - | - | - | 3.1844 | Disable decoder spatial inject path. |
| inj_no_texture_gain | inj_single | skipped_existing | - | - | - | - | 3.1914 | Disable texture residual head contribution. |
| inj_no_delta_highpass | inj_single | skipped_existing | - | - | - | - | 3.1913 | Disable delta high-frequency bias. |
| inj_no_delta_gate | inj_single | skipped_existing | - | - | - | - | 3.1863 | Disable style delta gate. |
| inj_no_decoder_adagn | inj_single | skipped_existing | - | - | - | - | 3.2003 | Disable decoder AdaGN, fallback to GroupNorm. |
| inj_no_output_affine | inj_single | skipped_existing | - | - | - | - | 3.1849 | Disable output style affine. |
| inj_spatial_highpass_on | inj_single | skipped_existing | - | - | - | - | 3.1882 | Enable high-pass style spatial maps. |
| inj_spatial_map_norm_off | inj_single | skipped_existing | - | - | - | - | 3.2526 | Disable style spatial map normalization. |
| inj_spatial_blur_off | inj_single | skipped_existing | - | - | - | - | 3.1701 | Disable style spatial blur before injection. |
| loss_no_distill | loss_single | skipped_existing | - | - | - | - | 3.1849 | Disable distillation term. |
| loss_no_code | loss_single | skipped_existing | - | - | - | - | 3.1849 | Disable code-closure term. |
| loss_no_stroke_gram | loss_single | skipped_existing | - | - | - | - | 1.2440 | Disable stroke gram style statistics. |
| loss_no_color_moment | loss_single | skipped_existing | - | - | - | - | 1.9785 | Disable color moment style statistics. |
| loss_no_style_spatial_tv | loss_single | skipped_existing | - | - | - | - | 3.1847 | Disable style spatial TV regularization. |
| loss_no_struct_edge | loss_single | skipped_existing | - | - | - | - | 2.6479 | Disable struct and edge guards. |
| loss_no_cycle | loss_single | skipped_existing | - | - | - | - | 3.1849 | Disable cycle consistency. |
| loss_no_nce | loss_single | skipped_existing | - | - | - | - | 3.1849 | Disable NCE content term. |
| loss_no_semigroup | loss_single | skipped_existing | - | - | - | - | 3.1849 | Disable semigroup consistency. |
| loss_no_push | loss_single | skipped_existing | - | - | - | - | 3.1849 | Disable style push-away term. |
| dyn_style_strength_jitter | dynamics | skipped_existing | - | - | - | - | 3.2006 | Widen train-time style strength range. |
<<<<<<< Updated upstream
| dyn_num_steps_1_to_3 | dynamics | failed | - | - | - | - | - | Use variable training steps in [1,3]. |
=======
>>>>>>> Stashed changes
| dyn_single_step_only | dynamics | skipped_existing | - | - | - | - | 3.2238 | Force single-step training and inference. |
| dyn_flat_step_schedule | dynamics | skipped_existing | - | - | - | - | 3.1750 | Use flat step schedule for train/infer. |
| bundle_style_up | bundle | skipped_existing | - | - | - | - | 3.8772 | Increase style injection + style losses for 50-epoch run. |
| bundle_style_down | bundle | skipped_existing | - | - | - | - | 2.3643 | Reduce style injection + style losses. |
| bundle_content_guard_strict | bundle | skipped_existing | - | - | - | - | 3.3164 | Increase structure-preservation losses. |
| bundle_content_guard_relaxed | bundle | skipped_existing | - | - | - | - | 3.0342 | Reduce structure-preservation losses for style-first bias. |
<<<<<<< Updated upstream
| bundle_style_losses_off | bundle | failed | - | - | - | - | 0.0004 | Turn off all style-specific loss terms. |
| bundle_all_style_paths_off | bundle | failed | - | - | - | - | - | Ablate main style-injection paths + style losses. |
| bundle_style_injection_heavy | bundle | failed | - | - | - | - | - | Strong style injection with extra smoothing regularization. |
=======
| bundle_style_losses_off | bundle | ok | - | - | - | - | 0.0000 | Turn off all style-specific loss terms. |
| bundle_all_style_paths_off | bundle | ok | - | - | - | - | 0.0000 | Ablate main style-injection paths + style losses. |
| bundle_style_injection_heavy | bundle | ok | - | - | - | - | 3.0193 | Strong style injection with extra smoothing regularization. |
>>>>>>> Stashed changes
