# Phase 2 Plan Pivot

Date: 2026-06-13

## Authority Update

- This note now serves as the dated pivot record only.
- The live execution authority is [612-phase2/README.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/612-phase2/README.md).
- The reread conclusion is strict:
  - `content_lpips >= 0.70` is complete failure.
  - `0.40 <= content_lpips < 0.70` is archival only.
  - no `LPIPS >= 0.40` line may define the next formal Distinct5 packet.

## Trigger

- reference docs:
  - [612-lookback/action_plan.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/612-lookback/action_plan.md)
  - [612-lookback/analysis.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/612-lookback/analysis.md)
  - [612-phase2/README.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/612-phase2/README.md)
- governing interpretation:
  - `content_lpips >= 0.70` is a complete failure
  - `0.40 <= content_lpips < 0.70` is archival only, not a promotable compromise

## Immediate Decision

- keep the live formal lane on `vel_tok32_safe_rescan_r2` because its first settled point is still in-band
- stop any current or future packet immediately once a settled authority point enters `0.40+`
- treat `content_lpips >= 0.70` as family-level failure evidence, not just a packet-level stop
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

1. The only live formal 3060 lane remains `vel_tok32_safe_rescan_r2` until it either breaks the in-band shelf or proves the safe-family sweep exhausted.
2. `eval_only_pc_solver` is already a negative auxiliary probe and cannot reenter the main queue.
3. Endpoint / I2SB docs stay as historical implementation logs, not as the live Distinct5 promotion plan.
4. A first settled checkpoint is now sufficient to kill a lane if LPIPS is already out of band.
5. Even an in-band line loses the formal slot once it shows a flat style plateau with no new joint point.
6. A `0.60-0.70+` LPIPS line is no longer allowed to influence queue ordering just because it carries higher style.
7. After the 612 reread, the queue is explicitly `safe tokenizer rescan -> structure-side reentry -> diagnostic-only I2SB`, not `style-first rescue`.

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

1. `vel_safe_family_rescan`
   - stay inside the same `velocity + tokenizer` family
   - scan only safe knobs first:
      - tokenizer temperature / structured temperature
      - global-spatial coupling
      - `w_kinetic` around the safe shelf
   - do this before reopening topology-anchor style structure patches
   - current surviving packet:
      - [phase2_vel_tok32_safe_rescan_r2_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_rescan_r2_seed42_b20a1.json)
      - [2026-06-13-phase2-vel-tok32-safe-rescan-r2.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-safe-rescan-r2.md)
   - stricter interpretation:
     - if `r2` crosses `0.40`, the safe-family sweep is exhausted rather than “encouraging but promotable later”
2. `vel_structure_control_reentry`
   - keep `velocity`, not endpoint
   - only after queue 1 creates a stronger in-band parent, or queue 1 is conclusively exhausted
   - structure control is still training-side, but is now explicitly third priority because the first topology-anchor retry already crossed `0.40`
   - queued reference packet:
      - [phase2_vel_tok32_topo_anchor_k075_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_topo_anchor_k075_seed42_b20a1.json)
      - [2026-06-13-phase2-vel-tok32-topo-anchor-reentry.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-topo-anchor-reentry.md)
3. `i2sb_eval_only_diagnostics`
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
  - `vel_tok32_safe_rescan_r1` is now closed:
    - `epoch_0001`
      - transfer `0.672934 / 0.384740`
      - all-pairs `0.700686 / 0.383351`
      - in-band but jointly dominated by both the old shelf and the direct parent
    - `epoch_0002`
      - transfer `0.676378 / 0.400694`
      - all-pairs `0.702543 / 0.397891`
      - style lift is real
      - but worst authority LPIPS crosses `0.40`
    - watcher reason:
      - `lpips_archival_stop`
    - interpretation:
      - the first safe-family rollback almost worked
      - but it is still archival-stop, not promotable
  - the live formal remote training lane is now:
    - [phase2_vel_tok32_safe_rescan_r2_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_rescan_r2_seed42_b20a1.json)
    - [2026-06-13-phase2-vel-tok32-safe-rescan-r2.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-safe-rescan-r2.md)
  - current live read:
    - remote process active on the 3060 lane
    - latest settled point is now `epoch_0002`
      - transfer `0.675645 / 0.395898`
      - all-pairs `0.702225 / 0.393204`
    - GPU stays around `9.9 GiB / 12.29 GiB`
  - execution interpretation:
    - `r2` remains a strict short-screen, not a style-first rescue line
    - `epoch_0002` is a new in-family Pareto point, but it still misses the safe-shelf recovery gate because LPIPS rose above `0.381724 / 0.384340`
    - if any settled point crosses `0.40`, queue ownership moves away from safe-family tokenizer rescan
    - if any settled point crosses `0.70`, the corresponding family is considered off the Distinct5 paper path
