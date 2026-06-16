# I2SB Orthogonal Low-Anchor 0.50 Closure

Date: 2026-06-16

## Status

- Run:
  `aaai2027_phase2_i2sb_orthogonal_lowanchor050_k070_e3_sigma0p02_b8a2_vlen010`.
- Parent:
  `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`.
- Scope:
  transfer-only `CLIP-S + LPIPS`, every retained checkpoint e1-e15.
- Decision:
  `closed_partial_positive_promote_e9_as_candidate`.

## Best Points

| point | transfer CLIP-S | LPIPS | style - IDT | read |
|---|---:|---:|---:|---|
| e1 style peak | 0.711470 | 0.472991 | +0.071549 | high style, structure still damaged |
| e5 balanced early | 0.702532 | 0.393892 | +0.062611 | close but not in-band |
| e9 target-facing | 0.701429 | 0.372203 | +0.061508 | promoted candidate for this mechanism |
| e14 LPIPS floor | 0.686635 | 0.348625 | +0.046715 | LPIPS-only, not promoted |
| e15 final | 0.682993 | 0.355775 | +0.043072 | final tail point, no style recovery |

## Matched Delta

| comparison | candidate | control | delta CLIP-S | delta LPIPS | decision |
|---|---|---|---:|---:|---|
| e9 vs hard orthogonal e4 | 0.701429 / 0.372203 | 0.698245 / 0.390826 | +0.003184 | -0.018623 | positive |
| e9 vs slerp+hard e15 | 0.701429 / 0.372203 | 0.678109 / 0.350421 | +0.023320 | +0.021781 | style-first positive |
| e9 vs clean I2SB e2 | 0.701429 / 0.372203 | 0.709094 / 0.490233 | -0.007665 | -0.118030 | structure-positive, style cost |

Machine-readable matched deltas are in
`docs/experiments/phase2_fiber_bundle/control_delta.csv`.

## Interpretation

The low-anchor coefficient is the first endpoint-geometry control that gives a
target-facing in-band point. Hard low-frequency replacement suppressed style;
weakening the low anchor to `0.50` preserves enough raw endpoint low-frequency
style/color to keep CLIP-S above `0.700` while moving LPIPS below `0.38`.

The tail still shows the familiar cooling behavior. After e9, e10-e15 improve
LPIPS or add low-style Pareto points, but none recover target-facing style.
Therefore e9 is the mechanism candidate, not e14/e15.

## Decision

Promote e9 as the current low-anchor candidate, but do not treat the family as
solved. The objective remains `0.74 / 0.30` with style priority. This result
shows the correct control knob exists; the next step is to scan anchor strength
or add a local/semantic structure correction without removing style low
frequencies.

## Next Action

Run the next single-variable anchor probe from the same parent:

- `endpoint_orthogonal_low_anchor=0.65`, no latent-slerp, no DINO/VLM, same
  solver/objective/loss/eval contract.
- Positive target:
  keep style near or above e9 while reducing LPIPS further.
- Stop rule:
  do not promote LPIPS-only points below `0.700` style.

Artifacts:

- Curve CSV:
  `docs/experiments/phase2_fiber_bundle/curves/i2sb_orthogonal_lowanchor050_k070_e3_fast10_curve.csv`.
- Eval mirror:
  `docs/experiments/phase2_fiber_bundle/eval/aaai2027_phase2_i2sb_orthogonal_lowanchor050_k070_e3_sigma0p02_b8a2_vlen010/full_eval_fast10/`.
- Homepage plot CSV:
  `docs/experiments/phase2_fiber_bundle/plot_points.csv`.
