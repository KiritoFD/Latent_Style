# Phase 2 Plan Pivot

Date: 2026-06-13

## Trigger

- reference docs:
  - [612-lookback/action_plan.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/612-lookback/action_plan.md)
  - [612-lookback/analysis.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/612-lookback/analysis.md)
  - [612-phase2/README.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/612-phase2/README.md)
- governing interpretation:
  - `content_lpips >= 0.70` is a complete failure
  - `0.40 <= content_lpips < 0.70` is archival only, not a promotable compromise

## Immediate Decision

- stop the corrected `rtfix` I2SB lane after the first settled point
- retire endpoint / I2SB from the active Distinct5 remote training queue
- keep true I2SB in code as implementation capability and theory evidence only
- downgrade any older round2 wording like “frontier”, “compromise”, or “mainline” when it refers to `LPIPS >= 0.40`

## Why

- corrected true-I2SB runtime point:
  - `rtfix epoch_0001`
  - transfer `0.724444 / 0.712723`
  - all-pairs `0.724472 / 0.707551`
- interpretation:
  - style is strong
  - structure is fully outside the acceptable band
  - this is not a tradeoff worth extending under the current paper gate

## Operational Consequences

1. The single formal 3060 training lane moves to `vel_pattn_enhanced_tok`.
2. `eval_only_pc_solver` becomes the first reuse-style auxiliary probe, but it cannot preempt the main training lane.
3. Endpoint / I2SB docs stay as historical implementation logs, not as the live Distinct5 promotion plan.
4. A first settled checkpoint is now sufficient to kill a lane if LPIPS is already out of band.

## New Priority Order

1. `vel_pattn_enhanced_tok`
   - velocity
   - enhanced `PureLatentSpatialTokenizer`
   - `manifold_adaptive_split`
   - `crossattn_texture`
   - config anchors:
     - [inmortal_k_manifold_seed42_b16.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_k_manifold_seed42_b16.json)
     - [inmortal_xpred_kmanifold_pattn_seed42_b16.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_seed42_b16.json)
2. `eval_only_pc_solver`
   - reuse strong style ckpts
   - test whether `solver_pc` can recover structure at inference time
   - config anchor:
     - [aaai2027_round1_solver_pc_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/round1_full_sweep/aaai2027_round1_solver_pc_seed42_b8a2.json)
3. `vel_kman_pattn_kin_sweep`
   - scan `w_kinetic`
   - only after queue 1 proves the velocity mainline stays inside the structure-safe band

## Hard Gates

- `LPIPS >= 0.70`
  - immediate fail-stop
- `0.40 <= LPIPS < 0.70`
  - non-promotable, archival only
  - not allowed to continue occupying the only formal remote training lane
- only `LPIPS < 0.40` lines remain eligible for the remote main lane

## Status

- remote `rtfix` lane has been stopped and archived as a structural failure line
- current planning authority is the Phase 2 velocity queue, not the endpoint / I2SB queue
