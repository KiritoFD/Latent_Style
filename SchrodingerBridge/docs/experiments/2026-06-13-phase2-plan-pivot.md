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
- stop the residual exact-Brownian retry after its first settled point
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
  - the first concrete follow-up packet was the topology-anchor velocity retry:
    - [2026-06-13-phase2-topology-anchor-followup.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-topology-anchor-followup.md)
  - first settled read on that packet:
    - `epoch_0001`
    - transfer `0.674077 / 0.393103`
    - all-pairs `0.700842 / 0.390843`
  - interpretation:
    - still in-band
    - not yet better than the previous velocity shelf
    - continue only as a short early check, not as an open-ended promotion
  - final closure read:
    - `epoch_0002`
    - transfer `0.680803 / 0.417910`
    - all-pairs `0.706132 / 0.413976`
  - interpretation:
    - style moved upward
    - but the line crossed into `archival only`
    - so the velocity topology-anchor retry is closed
  - the exact-I2SB fallback ladder was then executed and fully closed:
    - [2026-06-13-phase2-i2sb-topology-anchor-fallback.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-i2sb-topology-anchor-fallback.md)
    - `sigma=0.25, b30`
      - all-pairs `0.719743 / 0.725755`
      - immediate `fail_stop`
    - `sigma=0.10, warm_vel2, b30`
      - all-pairs `0.702178 / 0.711280`
      - immediate `fail_stop`
    - `sigma=0.10 + pattn + topology anchor, b22`
      - all-pairs `0.713362 / 0.684586`
      - immediate `fail_stop`
    - `sigma=0.02 + pattn + topology anchor, b22`
      - all-pairs `0.709801 / 0.675418`
      - archival only
    - `sigma=0.02 + residual endpoint, b22`
      - transfer `0.688376 / 0.571735`
      - all-pairs `0.697686 / 0.569086`
      - archival only
  - interpretation:
    - residual endpoint parameterization materially improved LPIPS relative to the absolute `sigma=0.02` lane
    - but it still failed the formal `< 0.40` structure gate by a wide margin
    - so the exact-I2SB queue is now closed for Distinct5 promotion work
  - execution result:
    - the residual lane was stopped immediately after the first settled point at `2026-06-13 07:30`
    - there is no active formal remote training lane right now

## New Priority Order

1. `vel_tok32_pos_refresh`
   - return to `velocity`
   - tokenizer-only strengthening from the 612 lookback:
     - deeper query extractor
     - 2D positional encoding
     - `num_clusters = 32`
     - tighter global-spatial coupling
   - start from the existing safe velocity parents:
     - [inmortal_k_manifold_seed42_b16.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_k_manifold_seed42_b16.json)
     - [inmortal_xpred_kmanifold_pattn_seed42_b16.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_seed42_b16.json)
   - prepared packet:
     - [phase2_vel_tok32_pos_refresh_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_pos_refresh_seed42_b20a1.json)
     - [2026-06-13-phase2-vel-tok32-pos-refresh.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-pos-refresh.md)
2. `vel_structure_control_reentry`
   - keep `velocity`, not endpoint
   - move structure control back into training:
     - lighter kinetic + topology anchor
     - latent lowpass / edge content correction
     - adaptive skip or PnP-style self-inject only as structure tools
   - queued concrete packet:
     - [phase2_vel_tok32_topo_anchor_k075_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_topo_anchor_k075_seed42_b20a1.json)
     - [2026-06-13-phase2-vel-tok32-topo-anchor-reentry.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-topo-anchor-reentry.md)
3. `vel_kinetic_anchor_rescan`
   - only if queue 1 yields a stronger in-band parent
   - retune the balance between style lift and structure guard on the safe velocity line
4. `i2sb_eval_only_diagnostics`
   - keep exact-I2SB for theory and implementation checks only
   - no formal training lane unless a cheap diagnostic first shows sub-`0.40` evidence

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
- the exact-I2SB fallback ladder is now also closed at the residual-endpoint `epoch_0001` archival readout
- current planning authority is the Phase 2 structure-first queue:
  - `eval_only_pc_solver` has finished as a negative read
  - the velocity topology-anchor retry has now also been closed as archival only
  - the active formal lane is now:
    - `aaai2027_phase2_vel_tok32_pos_refresh_seed42_b20a1`
    - launch time `2026-06-13 07:49`
    - 30s launch health `10073 MiB`
    - the packet reached `epoch_0001` and saved its first checkpoint
    - but the first settled eval is still pending because the launcher runtime guard killed the process during epoch-end eval offload:
      - `RUNTIME_UNDER_BAND_STOP used=2101MiB floor=9216MiB`
    - next action is not a model-side decision:
      - fix the launcher guard, relaunch, and recover from local `epoch_0001`
    - a local `watch_phase2_velocity_handoff.py --handoff-mode stop_only --wait --execute` watcher is now attached to enforce the same LPIPS / plateau close rule without auto-launching the old solver_pc follow-up
    - relaunch status:
      - the launcher fix is now in place
      - the same run resumed from local `epoch_0001`
      - the post-fix 30s health read is `10151 MiB`
      - first settled authority point is now:
        - `epoch_0002`
        - transfer `0.673024 / 0.390256`
        - all-pairs `0.700342 / 0.387609`
      - interpretation:
        - still in-band, so the lane stays alive
        - but it is not yet better than the previous safe parent shelf `0.701666 / 0.381724`
        - the stale `epoch_0001` half-eval is now classified as ops residue rather than a real pending authority point
        - the live read is back to `training_after_settled_eval` while `epoch_3` trains
      - second settled authority point:
        - `epoch_0003`
        - transfer `0.668702 / 0.364875`
        - all-pairs `0.698072 / 0.361798`
      - updated interpretation:
        - still in-band
        - no breakout on style
        - but the line is still producing new in-band Pareto points, so the formal lane remains justified for now
      - third settled authority point:
        - `epoch_0004`
        - transfer `0.673399 / 0.376463`
        - all-pairs `0.701161 / 0.374695`
      - updated interpretation:
        - still in-band
        - still below the old style shelf `0.701666`
        - but now better on LPIPS than the old safe parent
        - and strictly stronger than this packet's earlier `epoch_0003` point
        - so the line is still alive and should keep the formal slot for now
      - fourth settled authority point:
        - `epoch_0005`
        - transfer `0.670604 / 0.375912`
        - all-pairs `0.699187 / 0.373331`
      - updated interpretation:
        - still in-band
        - still not a style breakout
        - but it adds another Pareto-valid point rather than flattening
        - so the lane still does not satisfy the current closure rule
