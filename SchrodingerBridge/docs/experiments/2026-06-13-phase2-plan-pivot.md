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
6. A `0.60-0.70+` LPIPS line is no longer allowed to influence queue ordering just because it carries higher style.

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
    - at that decision instant there was no active formal remote training lane
    - the later `vel_tok32_pos_refresh` launch is recorded below in the Status section

## Tightened Priority Order

- governing change after the 612 reread:
  - `LPIPS >= 0.70` is not just fail-stop at the run level; it is evidence that the family is off the Distinct5 paper path
  - `0.40 <= LPIPS < 0.70` is archival evidence only and cannot be used to justify a more aggressive next packet
  - the queue must therefore optimize for in-band improvement first, not style-first rescue

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
   - stage target:
     - first beat `all-pairs 0.701666 / 0.381724`
     - then reach `style >= 0.705` with `LPIPS <= 0.380`
2. `vel_safe_family_rescan`
   - stay inside the same `velocity + tokenizer` family
   - scan only safe knobs first:
     - tokenizer temperature / structured temperature
     - global-spatial coupling
     - `w_kinetic` around the safe shelf
   - do this before reopening topology-anchor style structure patches
   - first concrete packet:
     - [phase2_vel_tok32_safe_rescan_r1_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_rescan_r1_seed42_b20a1.json)
     - [2026-06-13-phase2-vel-tok32-safe-rescan-r1.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-safe-rescan-r1.md)
3. `vel_structure_control_reentry`
   - keep `velocity`, not endpoint
   - only after queue 1 or 2 creates a stronger in-band parent
   - structure control is still training-side, but is now explicitly third priority because the first topology-anchor retry already crossed `0.40`
   - queued reference packet:
     - [phase2_vel_tok32_topo_anchor_k075_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_topo_anchor_k075_seed42_b20a1.json)
     - [2026-06-13-phase2-vel-tok32-topo-anchor-reentry.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-topo-anchor-reentry.md)
4. `i2sb_eval_only_diagnostics`
   - keep exact-I2SB for theory and implementation checks only
   - no formal training lane unless a cheap diagnostic first shows sub-`0.40` evidence

## Hard Gates

- `LPIPS >= 0.70`
  - immediate fail-stop
  - family leaves the Distinct5 formal promotion path until redesigned from a safe parent
- `0.40 <= LPIPS < 0.70`
  - non-promotable, archival only
  - not allowed to continue occupying the only formal remote training lane
  - not allowed to define the next formal queue item
- only `LPIPS < 0.40` lines remain eligible for the remote main lane

## In-Band Milestones

1. shelf break
   - exceed `all-pairs 0.701666 / 0.381724`
2. stage-B break
   - exceed `all-pairs style 0.705` while staying at `LPIPS <= 0.380`
3. stage-C break
   - exceed `all-pairs style 0.710` while staying at `LPIPS <= 0.370`
4. long-horizon paper target
   - exceed `style 0.72` while staying at `LPIPS <= 0.35`

## Status

- remote `rtfix` lane has been stopped and archived as a structural failure line
- the first Phase 2 velocity queue is now also closed after `epoch_0006`
- the exact-I2SB fallback ladder is now also closed at the residual-endpoint `epoch_0001` archival readout
- current planning authority is the Phase 2 safe-band queue:
  - `eval_only_pc_solver` has finished as a negative read
  - the velocity topology-anchor retry has now also been closed as archival only
  - `vel_tok32_pos_refresh` is now closed:
    - best `epoch_0004`
      - transfer `0.673399 / 0.376463`
      - all-pairs `0.701161 / 0.374695`
    - closure `epoch_0006`
      - transfer `0.671522 / 0.385051`
      - all-pairs `0.699725 / 0.381878`
    - interpretation:
      - the line stayed fully in-band
      - but it never beat the old safe shelf `0.701666 / 0.381724`
      - the newest two settled points did not recover a style breakout
      - so the formal slot should move to queue-2 safe-family rescan
  - the active formal lane is now:
    - [phase2_vel_tok32_safe_rescan_r1_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_rescan_r1_seed42_b20a1.json)
    - [2026-06-13-phase2-vel-tok32-safe-rescan-r1.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-safe-rescan-r1.md)
    - initial launch time `2026-06-13 10:24`
    - early diagnostic relaunch `2026-06-13 10:35`
    - current 30s launch health `10140 MiB`
    - current live read:
      - `live_state = training_after_settled_eval`
      - `remote_gpu ~= 9792 MiB`
      - `latest_checkpoint_epoch = epoch_0001`
      - `latest_settled_epoch = epoch_0001`
    - first settled authority point:
      - transfer `0.672934 / 0.384740`
      - all-pairs `0.700686 / 0.383351`
    - interpretation:
      - still in-band
      - not better than the old safe shelf
      - not better than `tok32_pos_refresh epoch_0004`
      - but also not bad enough to give up the slot after one point
      - the packet is therefore switched into a short-screen audit:
        - if no shelf break appears by `epoch_0003`, close early
    - watcher:
      - `watch_phase2_velocity_handoff.py --run-name aaai2027_phase2_vel_tok32_safe_rescan_r1_seed42_b20a1 --wait --execute --handoff-mode stop_only --min-settled-epoch 3`
