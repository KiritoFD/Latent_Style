# Style Overdrive k070 e3 Eval-Only Closure

Date: 2026-06-15

## Scope

This is an eval-only extrapolation scan from `k070 epoch_0003`, not a clean trained model improvement. After the 2026-06-15 infra cleanup, style overdrive requires explicit `model.allow_style_overdrive=true`; these points should be read as diagnostic evidence about style capacity and extrapolation behavior.

## Curve

Source: `docs/experiments/phase2_fiber_bundle/curves/style_overdrive_all_k070_e3_eval_only_curve.csv`

| tag | style strength | latent affine | transfer CLIP-S | transfer LPIPS | all-pairs CLIP-S | all-pairs LPIPS | decision |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| s110 | 1.10 | none | 0.673598 | 0.304429 | 0.705568 | 0.302402 | pure_style_overdrive |
| s120 | 1.20 | none | 0.675085 | 0.296139 | 0.707714 | 0.294172 | lpips_target_positive |
| s135 | 1.35 | none | 0.678224 | 0.288947 | 0.711243 | 0.287131 | balanced_positive |
| s160 | 1.60 | none | 0.683721 | 0.295983 | 0.716000 | 0.294629 | style_overdrive_frontier |
| s135_lataff045 | 1.35 | 0.45 | 0.682952 | 0.311925 | 0.715829 | 0.306325 | combo_style_candidate |
| s160_lataff045 | 1.60 | 0.45 | 0.686336 | 0.315394 | 0.718753 | 0.309988 | style_ceiling_lpips_cost |

## Decision

Status: `diagnostic_only_not_promoted`.

Overdrive shows that the current k070 velocity field has hidden style headroom: pure `s1.35` improves transfer style while staying below the LPIPS target, and `s1.60` pushes style further. However, this is out-of-training-domain integration (`t>1`) and therefore cannot be treated as a clean Fiber Bundle mechanism result.

Latent affine raises style more, but it is a global calibration layer and now requires explicit `allow_metric_postprocess`; it is retained as a style-statistics diagnostic, not as model capacity.

## Next Action

Use these points to motivate an in-domain actuation fix: train a mechanism that increases style injection strength inside `t in [0, 1]`, rather than depending on overdrive or metric-affecting affine calibration.
