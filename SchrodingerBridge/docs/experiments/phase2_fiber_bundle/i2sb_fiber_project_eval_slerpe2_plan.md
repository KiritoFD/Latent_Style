# Eval-Only Orthogonal Fiber-SDE Projection On Slerp e2

Date: 2026-06-16

## Goal

Check whether the negative low-anchor0.50 e9 hard-projection result is
checkpoint-specific. Use the stronger-style latent-slerp e2 checkpoint and run
only two diagnostic sigmas.

## Parent Checkpoint

- Checkpoint:
  `exp/aaai2027_phase2_i2sb_latent_slerp_k070_e3_sigma0p02_b8a2_vlen010/epoch_0002.pt`
- Baseline:
  `0.712038 / 0.476511`

## Scan

| config | sigma |
|---|---:|
| `phase2_eval_fiber_project_sigma0p0_slerpe2.json` | `0.0` |
| `phase2_eval_fiber_project_sigma0p5_slerpe2.json` | `0.5` |

## Decision

- If sigma0 lowers LPIPS while keeping style above `0.700`, hard endpoint
  projection may still be checkpoint-dependent.
- If sigma0.5 raises style without catastrophic LPIPS, highpass noise may still
  be useful on stronger-style parents.
- If both fail, close the raw latent avg-pool projector family more broadly.

## Artifact Targets

- Eval mirrors:
  `docs/experiments/phase2_fiber_bundle/eval/aaai2027_eval_fiber_project_slerpe2_sigma*/`
- Consolidated CSV:
  `docs/experiments/phase2_fiber_bundle/curves/i2sb_fiber_project_slerpe2_sigma_scan.csv`
