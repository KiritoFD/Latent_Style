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

1. The single formal 3060 training lane moved to `vel_pattn_enhanced_tok`, and that first Phase 2 velocity packet is now closed at `epoch_0006`.
2. `eval_only_pc_solver` has now completed as a negative reuse-style auxiliary probe.
3. Endpoint / I2SB docs stay as historical implementation logs, not as the live Distinct5 promotion plan.
4. A first settled checkpoint is now sufficient to kill a lane if LPIPS is already out of band.
5. Even an in-band line loses the formal slot once it shows a flat style plateau with no new joint point.

## Current Phase Node

- refreshed remote read at `2026-06-13 04:00`:
  - best `epoch_0002`
    - transfer `0.673934 / 0.384340`
    - all-pairs `0.701666 / 0.381724`
  - latest `epoch_0006`
    - transfer `0.668831 / 0.370651`
    - all-pairs `0.698086 / 0.367844`
- interpretation:
  - this is not an LPIPS failure line
  - it stayed in-band, but never broke out from the `0.699 +/-` style shelf
  - `best_in_newest_2 = false` and the tail is flat enough to stop burning the only formal lane
- execution result:
  - `watch_phase2_velocity_handoff.py` now encodes both LPIPS hard gates and the plateau rule
  - the velocity PID was stopped after `epoch_0006`
  - `eval_only_pc_solver` was launched against `epoch_0011` of the style-strong `xpred + pattn` parent
  - that eval finished at `2026-06-13 04:12`
    - transfer `0.729014 / 0.621056`
    - all-pairs `0.735295 / 0.611310`
  - interpretation:
    - solver-only correction did not rescue structure
    - this probe becomes archival evidence, not a promotion path
- next formal candidate, if we keep pushing Phase 2, returns to training-side structure control rather than more solver-only recycling
  - the first concrete follow-up packet is now the topology-anchor velocity retry:
    - [2026-06-13-phase2-topology-anchor-followup.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-topology-anchor-followup.md)
  - first settled read on that packet:
    - `epoch_0001`
    - transfer `0.674077 / 0.393103`
    - all-pairs `0.700842 / 0.390843`
  - interpretation:
    - still in-band
    - not yet better than the previous velocity shelf
    - continue only as a short early check, not as an open-ended promotion

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
  - queue 2 has now answered that inference-time correction alone is not enough
  - this is therefore the next formal candidate if we stay on the velocity mainline

## Hard Gates

- `LPIPS >= 0.70`
  - immediate fail-stop
- `0.40 <= LPIPS < 0.70`
  - non-promotable, archival only
  - not allowed to continue occupying the only formal remote training lane
- only `LPIPS < 0.40` lines remain eligible for the remote main lane

## Status

- remote `rtfix` lane has been stopped and archived as a structural failure line
- the first Phase 2 velocity queue is now also closed after `epoch_0006`
- current planning authority is the Phase 2 structure-first queue:
  - `eval_only_pc_solver` has finished as a negative read
  - the next formal candidate is the training-side velocity follow-up, not another solver-only patch
