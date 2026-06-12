# Phase 2 Plan Pivot

Date: 2026-06-13

## Trigger

- reference docs:
  - [612-lookback/action_plan.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/612-lookback/action_plan.md)
  - [612-lookback/analysis.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/612-lookback/analysis.md)
  - [612-phase2/README.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/612-phase2/README.md)
- new hard rule:
  - `LPIPS 0.7` is a complete failure

## Immediate Decision

- stop the corrected `rtfix` I2SB lane after the first settled point
- do not continue any Distinct5 endpoint / I2SB line whose `content_lpips >= 0.70`
- keep the true-I2SB implementation in code, but remove it from the active remote training lane

## Why

- corrected true-I2SB runtime point:
  - `rtfix epoch_0001`
  - transfer `0.724444 / 0.712723`
  - all-pairs `0.724472 / 0.707551`
- interpretation:
  - style is strong
  - structure is completely outside the acceptable band
  - this is not a “tradeoff” worth extending under the current paper gate

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

## Hard Gates

- `LPIPS >= 0.70`:
  - immediate fail-stop
- `0.40 <= LPIPS < 0.70`:
  - non-promotable, archival only
- only `LPIPS < 0.40` lines remain eligible for the remote main lane

## Status

- remote `rtfix` lane has been stopped
- next launch should come from the Phase 2 queue, not from the endpoint / I2SB queue
