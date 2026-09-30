# I2SB Latent Slerp Closure

Date: 2026-06-16

## Status

- Run:
  `aaai2027_phase2_i2sb_latent_slerp_k070_e3_sigma0p02_b8a2_vlen010`.
- Continuation config:
  `configs/aaai2027/phase2_i2sb_latent_slerp_k070_e3_sigma0p02_b8a2_vlen010_e28.json`.
- Parent:
  `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`.
- Scope:
  transfer-only `CLIP-S + LPIPS`, every retained checkpoint e1-e28.
- Decision:
  `closed_partial_positive_not_promoted`.

## Best Points

| point | transfer CLIP-S | LPIPS | style - IDT | read |
|---|---:|---:|---:|---|
| e2 style peak | 0.712038 | 0.476511 | +0.072117 | best style; matched clean-I2SB gain |
| e10 structure-side knee | 0.701837 | 0.385366 | +0.061916 | first useful mid-curve structure point |
| e20 structure-side Pareto | 0.687134 | 0.360616 | +0.047213 | low-LPIPS tail, style too low |
| e28 LPIPS floor | 0.682638 | 0.352726 | +0.042718 | best LPIPS, style cooled further |

Matched e2 delta against clean absolute I2SB sigma0.02:

- Clean e2: `0.709094 / 0.490233`.
- Slerp e2: `0.712038 / 0.476511`.
- Delta: `+0.002944` transfer CLIP-S and `-0.013722` LPIPS.

## Convergence Read

The automatic joint Pareto tracker remains `converged=false` at e28 because
late checkpoints keep making tiny LPIPS-only improvements:

- e22: `0.683500 / 0.356225`.
- e26: `0.684500 / 0.355143`.
- e28: `0.682638 / 0.352726`.

This is not progress toward the active target. The target is style-first
`0.74 / 0.30`, and the style front has not improved after e2 across 26 later
retained checkpoints. The e21-e28 extension was run specifically to avoid
closing on the earlier e20 LPIPS-only Pareto. It confirms the late trend is
structure cooling with style decay, not a delayed style breakthrough.

## Decision

Latent slerp is a real path-geometry positive, but not a promotable standalone
model:

- Effective:
  it improves clean absolute I2SB at the matched e2 checkpoint on both transfer
  style and LPIPS.
- Not enough:
  it does not solve the style/structure coupling; high style remains high
  LPIPS, and low LPIPS comes with style collapse to the `0.68` band.
- Next use:
  keep latent-slerp as an integration ingredient only if the next mechanism
  explicitly constrains structure without scalar-shrinking the endpoint style
  force.

## Next Action

Do not spend another lane on more latent-slerp epochs or more isotropic noise.
The next controlled experiment should keep the e2 style shock while adding a
geometry-aware structure correction:

- `latent_slerp + orthogonal_lowhigh endpoint`, same parent and eval contract.
- Or an endpoint projection that preserves absolute high-frequency style while
  anchoring low-frequency/content structure.

Artifacts:

- Curve CSV:
  `docs/experiments/phase2_fiber_bundle/curves/i2sb_latent_slerp_k070_e3_fast10_curve.csv`.
- Eval mirror:
  `docs/experiments/phase2_fiber_bundle/eval/i2sb_latent_slerp_k070_e3_sigma0p02_b8a2_vlen010/full_eval_fast10/`.
- Homepage plot CSV:
  `docs/experiments/phase2_fiber_bundle/plot_points.csv`.
- Paper-facing page-1 CSV:
  `aaai2027/page1_bundle/wikiart5_page1_clip_lpips_points.csv`.
