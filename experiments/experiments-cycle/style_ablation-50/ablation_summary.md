# Style Ablation Summary

- Generated: 2026-02-13 16:10:09
- Mode: `all`
- Target epochs per run: `50`

| Variant | Category | Status | Latest transfer | Best transfer | Best epoch | Latest p2a | Last train loss | Note |
|---|---|---|---:|---:|---:|---:|---:|---|
| baseline_50e | baseline | generated | - | - | - | - | 3.2715 | 50-epoch short-run baseline (legacy warmup keys removed). |
| inj_no_decoder_spatial | inj_single | generated | - | - | - | - | - | Disable decoder spatial inject path. |
| inj_no_texture_gain | inj_single | generated | - | - | - | - | - | Disable texture residual head contribution. |
| inj_no_delta_highpass | inj_single | generated | - | - | - | - | - | Disable delta high-frequency bias. |
| inj_no_delta_gate | inj_single | generated | - | - | - | - | - | Disable style delta gate. |
| inj_no_decoder_adagn | inj_single | generated | - | - | - | - | - | Disable decoder AdaGN, fallback to GroupNorm. |
| inj_no_output_affine | inj_single | generated | - | - | - | - | - | Disable output style affine. |
| inj_spatial_highpass_on | inj_single | generated | - | - | - | - | - | Enable high-pass style spatial maps. |
| inj_spatial_map_norm_off | inj_single | generated | - | - | - | - | - | Disable style spatial map normalization. |
| inj_spatial_blur_off | inj_single | generated | - | - | - | - | - | Disable style spatial blur before injection. |
| loss_no_distill | loss_single | generated | - | - | - | - | - | Disable distillation term. |
| loss_no_code | loss_single | generated | - | - | - | - | - | Disable code-closure term. |
| loss_no_stroke_gram | loss_single | generated | - | - | - | - | - | Disable stroke gram style statistics. |
| loss_no_color_moment | loss_single | generated | - | - | - | - | - | Disable color moment style statistics. |
| loss_no_style_spatial_tv | loss_single | generated | - | - | - | - | - | Disable style spatial TV regularization. |
| loss_no_struct_edge | loss_single | generated | - | - | - | - | - | Disable struct and edge guards. |
| loss_no_cycle | loss_single | generated | - | - | - | - | - | Disable cycle consistency. |
| loss_no_nce | loss_single | generated | - | - | - | - | - | Disable NCE content term. |
| loss_no_semigroup | loss_single | generated | - | - | - | - | - | Disable semigroup consistency. |
| loss_no_push | loss_single | generated | - | - | - | - | - | Disable style push-away term. |
| dyn_style_strength_jitter | dynamics | generated | - | - | - | - | - | Widen train-time style strength range. |
| dyn_num_steps_1_to_3 | dynamics | generated | - | - | - | - | - | Use variable training steps in [1,3]. |
| dyn_single_step_only | dynamics | generated | - | - | - | - | - | Force single-step training and inference. |
| dyn_flat_step_schedule | dynamics | generated | - | - | - | - | - | Use flat step schedule for train/infer. |
| bundle_style_up | bundle | generated | - | - | - | - | - | Increase style injection + style losses for 50-epoch run. |
| bundle_style_down | bundle | generated | - | - | - | - | - | Reduce style injection + style losses. |
| bundle_content_guard_strict | bundle | generated | - | - | - | - | - | Increase structure-preservation losses. |
| bundle_content_guard_relaxed | bundle | generated | - | - | - | - | - | Reduce structure-preservation losses for style-first bias. |
| bundle_style_losses_off | bundle | generated | - | - | - | - | - | Turn off all style-specific loss terms. |
| bundle_all_style_paths_off | bundle | generated | - | - | - | - | - | Ablate main style-injection paths + style losses. |
| bundle_style_injection_heavy | bundle | generated | - | - | - | - | - | Strong style injection with extra smoothing regularization. |
