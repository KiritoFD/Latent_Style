# I2SB Slerp + Orthogonal Low/High Closure

Date: 2026-06-16

## Status

- Run:
  `aaai2027_phase2_i2sb_slerp_orthogonal_lowhigh_k070_e3_sigma0p02_b8a2_vlen010`.
- Parent:
  `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`.
- Scope:
  transfer-only `CLIP-S + LPIPS`, every retained checkpoint e1-e16.
- Decision:
  `closed_negative_style_suppressed_not_promoted`.

## Best Points

| point | transfer CLIP-S | LPIPS | style - IDT | read |
|---|---:|---:|---:|---|
| e1 style peak | 0.704828 | 0.446676 | +0.064907 | best style; below latent-slerp e2 |
| e3 early structure | 0.698710 | 0.392849 | +0.058790 | structure improves, style drops below 0.70 |
| e8 LPIPS knee | 0.685025 | 0.352807 | +0.045104 | low-style LPIPS-side Pareto |
| e15 LPIPS floor | 0.678109 | 0.350421 | +0.038188 | best LPIPS, not target-facing |
| e16 final | 0.678957 | 0.371222 | +0.039036 | no style rebound |

## Matched Delta

| comparison | candidate | control | delta CLIP-S | delta LPIPS | decision |
|---|---|---|---:|---:|---|
| e1 vs latent-slerp e2 | 0.704828 / 0.446676 | 0.712038 / 0.476511 | -0.007210 | -0.029835 | structure gain, style loss |
| e1 vs orthogonal-lowhigh e1 | 0.704828 / 0.446676 | 0.705847 / 0.451386 | -0.001019 | -0.004710 | effectively same mechanism |
| e15 vs latent-slerp e28 | 0.678109 / 0.350421 | 0.682638 / 0.352726 | -0.004529 | -0.002305 | LPIPS-only, not promoted |

Full machine-readable deltas are appended to
`docs/experiments/phase2_fiber_bundle/control_delta.csv`.

## Interpretation

This test cleanly falsifies the simple integration hypothesis. Latent-slerp
does create an early style shock when used alone, and orthogonal low/high
projection restrains structure when used alone. Combining them with a hard
low-frequency content anchor does not preserve both effects. The endpoint
projection dominates the path geometry and suppresses style actuation.

The useful lesson is narrow but important: structure is no longer the only
failure mode. We can drive LPIPS toward the acceptable `0.35` region, but the
current projection removes the low-frequency/color component of style along
with structural drift. A better correction must constrain structure without
globally replacing the endpoint lowpass.

## Decision

Do not promote and do not continue this run. The run was stopped after e16
because the best style point was e1 and 15 later retained checkpoints did not
recover style. The e15 LPIPS point is a diagnostic structure point, not progress
toward the style-first Seedream target.

## Next Action

The next mechanism should keep absolute endpoint style force and make the
structure correction local or semantic rather than a hard lowpass replacement.
Recommended next controlled probes:

- `endpoint_lowpass_anchor_blend`: use a partial lowpass anchor instead of
  full replacement, with only one anchor coefficient changed per run.
- `endpoint_chroma_preserve_lowpass`: anchor luminance/edge structure while
  preserving raw endpoint chroma or style low-frequency residual.
- `structure_loss_only_i2sb_abs`: keep absolute endpoint and add a measured
  structure constraint during training, rather than modifying endpoint geometry
  at inference/training target construction.

Artifacts:

- Curve CSV:
  `docs/experiments/phase2_fiber_bundle/curves/i2sb_slerp_orthogonal_lowhigh_k070_e3_fast10_curve.csv`.
- Eval mirror:
  `docs/experiments/phase2_fiber_bundle/eval/aaai2027_phase2_i2sb_slerp_orthogonal_lowhigh_k070_e3_sigma0p02_b8a2_vlen010/full_eval_fast10/`.
- Homepage plot CSV:
  `docs/experiments/phase2_fiber_bundle/plot_points.csv`.
- Paper-facing page-1 CSV:
  `aaai2027/page1_bundle/wikiart5_page1_clip_lpips_points.csv`.
