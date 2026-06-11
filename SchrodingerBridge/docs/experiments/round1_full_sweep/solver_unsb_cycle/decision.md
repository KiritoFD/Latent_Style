# solver_unsb_cycle Decision

- Decision date:
  - `2026-06-11`
- Current status:
  - `recalibration_needed`
- Current read:
  - the first under-band `batch=8` opening has been superseded
  - the current canonical launch is now `batch=17`
  - the family has resumed successfully through bounded continuation, but it is not left as a continuously live lane between segments

## Launch Decision

- Promote to `running`.
- Rationale:
  - `batch=15`
  - `accumulation_steps=2`
  - 30-second health sample:
    - `9677 MiB / 12288 MiB`
  - this sits cleanly inside the formal `9.0-10.8 GiB` band

## First Formal Read

- First settled authority points:
  - `epoch_0001`
    - transfer `0.7057 / 0.5669`
    - full `0.7150 / 0.5608`
  - `epoch_0002`
    - transfer `0.6975 / 0.5372`
    - full `0.7101 / 0.5312`
  - `epoch_0003`
    - transfer `0.7027 / 0.5117`
    - full `0.7195 / 0.5024`
  - `epoch_0004`
    - transfer `0.7001 / 0.5181`
    - full `0.7164 / 0.5097`
  - `epoch_0005`
    - transfer `0.6951 / 0.5144`
    - full `0.7119 / 0.5054`
- Read:
  - the family is no longer in calibration mode
  - `epoch_0003` is the first clear solver-style best point so far:
    - LPIPS improved materially from the opening
    - all-pairs style also improved
  - `epoch_0004-0005` then gave two consecutive mild rollbacks from that point
  - all future keep/reject decisions should now be made from the all-ckpt remote fast curve

## Current Direction Read

- Latest settled authority point:
  - `epoch_0014`
  - transfer `0.6929 / 0.5097`
  - full `0.7097 / 0.5009`
- Current interpretation:
  - `epoch_0006-0007` had made the line look like a continuing rollback from `epoch_0003`
  - `epoch_0008` first recovered materially on both style and LPIPS relative to `epoch_0007`
  - `epoch_0009` then converts that rebound into a true new Pareto point
  - it now owns the best all-pairs `CLIP-S` and the best LPIPS point in this family
  - `epoch_0010` softens from that new frontier, but does not erase it
  - `epoch_0011` softens again, so the post-`epoch_0009` tail currently looks like follow-up drift rather than stable frontier occupancy
  - `epoch_0011` also carries a much slower eval wall time than the surrounding checkpoints
  - `epoch_0012` rebounds from `epoch_0011`, but it still does not reclaim the `epoch_0009` frontier
  - `epoch_0013-0014` remain in the same partial-recovery band as `epoch_0012`, rather than extending into a real frontier recapture
- Operational consequence:
  - fully reset the solver patience read
  - do not treat this family as approaching closure any more
  - keep the formal lane open until the post-`epoch_0009` tail is actually observed
  - the immediate decision hinge is whether `epoch_0015+` continue climbing back toward `epoch_0009` or whether `epoch_0012-0014` were only a weak recovery shelf

## Interruption Audit

- Audit timestamp:
  - `2026-06-11`
- Current machine read:
  - no remote train process remains for `aaai2027_round1_solver_unsb_cycle_seed42_b8a2`
  - fast-eval watcher still exists
  - retained checkpoints stop at `epoch_0014`
  - latest train log exit line is:
    - `2026-06-11 13:33:06 +08:00`
    - `=== END 2026-06-11T13:33:06+08:00 rc=143 ===`
- Interpretation:
  - the family did not close by convergence
  - it also did not produce any retained checkpoint after `epoch_0014`
  - therefore the previous `running` state had become stale and had to be cleared
- Decision:
  - downgrade the family from `running` to `recalibration_needed`
  - do not treat the post-UNSB queue handoff plan as active closure logic yet
  - the next valid step for this family is resume, not replacement-by-default

## Resume Recalibration Audit

- first segmented continuation attempt from `epoch_0014`:
  - launcher path succeeded
  - checkpoint resume succeeded
  - but formal health gate failed before the lane could continue
- observed read:
  - `batch=15`
  - health memory `7734 MiB`
  - below the formal floor
- decision:
  - keep the family at `recalibration_needed`
  - first raise the canonical UNSB batch from `15` to `19`
  - then retry the segmented continuation

## Second Recalibration Read

- the `batch=19` retry started cleanly and resumed from `epoch_0014`
- but runtime guard then recorded:
  - `used=11811 MiB`
  - `cap=11571 MiB`
  - process end `rc=143`
- decision:
  - `batch=19` is above the hard paper-facing cap
  - move the next formal UNSB retry to the midpoint:
    - `batch=17`

## Third Recalibration Read

- the `batch=17` retry passed the formal launch gate cleanly
- two bounded continuation segments then completed:
  - `epoch_0015 -> epoch_0016`
  - `epoch_0017 -> epoch_0018`
- all retained checkpoints through `epoch_0018` now have remote `CLIP-S + LPIPS`
- curve read:
  - `epoch_0015`
    - transfer `0.6901 / 0.4824`
    - full `0.7113 / 0.4718`
  - `epoch_0016`
    - transfer `0.6954 / 0.5000`
    - full `0.7156 / 0.4876`
  - `epoch_0017`
    - transfer `0.6964 / 0.5277`
    - full `0.7139 / 0.5156`
  - `epoch_0018`
    - transfer `0.7012 / 0.5041`
    - full `0.7208 / 0.4901`
- decision:
  - `batch=17` is the current paper-safe UNSB batch
  - `epoch_0018` is a real new Pareto point inside this family
  - solver patience is reset again at `epoch_0018`
  - this family should continue as the next bounded continuation candidate, not hand off to the DINO tail or the next non-DINO family yet

## Post-Reset Followup Read

- bounded continuation from `epoch_0018` through `epoch_0020` is now settled
- new points:
  - `epoch_0019`
    - transfer `0.6916 / 0.5202`
    - full `0.7097 / 0.5083`
  - `epoch_0020`
    - transfer `0.6894 / 0.4574`
    - full `0.7133 / 0.4465`
- interpretation:
  - neither `epoch_0019` nor `epoch_0020` beats the `epoch_0018` Pareto point
  - `epoch_0020` does recover LPIPS relative to `epoch_0019`
  - the tail is therefore still live and non-flat rather than closure-grade
- current decision:
  - keep `solver_unsb_cycle` as the active logical family
  - continue bounded segmented continuation from `epoch_0020`
  - do not hand off the formal program to another family yet

## Second Post-Reset Followup Read

- bounded continuation from `epoch_0020` through `epoch_0022` is now settled
- new points:
  - `epoch_0021`
    - transfer `0.6937 / 0.5124`
    - full `0.7129 / 0.4997`
  - `epoch_0022`
    - transfer `0.6923 / 0.5101`
    - full `0.7105 / 0.4992`
- interpretation:
  - neither `epoch_0021` nor `epoch_0022` beats the `epoch_0018` Pareto point
  - the post-`epoch_0018` tail now has two further non-winning confirmations
  - but the tail is still not flat enough for closure and the solver patience has not been exhausted
- current decision:
  - keep `solver_unsb_cycle` as the active logical family
  - continue bounded segmented continuation from `epoch_0022`
  - do not hand off to tokenizer `DINO` or the next non-DINO family yet

## Third Post-Reset Followup Read

- bounded continuation from `epoch_0022` through `epoch_0024` is now settled
- new points:
  - `epoch_0023`
    - transfer `0.6903 / 0.4866`
    - full `0.7108 / 0.4749`
  - `epoch_0024`
    - transfer `0.6927 / 0.5197`
    - full `0.7102 / 0.5083`
- interpretation:
  - `epoch_0023-0024` still do not beat the `epoch_0018` Pareto point
  - `since_last_pareto` has now reached `6`
  - but the tail is still not flat enough for closure because the post-`epoch_0018` band remains visibly non-monotone
- current decision:
  - keep `solver_unsb_cycle` open
  - the next bounded continuation should test whether the tail finally flattens after `epoch_0024`
  - do not hand off to a different family yet

## Fourth Post-Reset Followup Read

- bounded continuation from `epoch_0024` through `epoch_0026` is now settled
- new points:
  - `epoch_0025`
    - transfer `0.6866 / 0.4675`
    - full `0.7095 / 0.4578`
  - `epoch_0026`
    - transfer `0.6915 / 0.5290`
    - full `0.7078 / 0.5188`
- interpretation:
  - `epoch_0025-0026` still do not beat the `epoch_0018` Pareto point
  - `since_last_pareto` is now `8`, which is beyond the solver patience target
  - but the tail is still not flat enough to justify closure
- current decision:
  - keep `solver_unsb_cycle` open for at least one more bounded continuation check
  - the next segment should decide whether the tail finally flattens or recovers

## Promotion Rule

- No early promotion from the first in-band health sample alone.
- Require the same family closure package as every other round-1 family before any keep/reject claim.
- If this family is resumed, prefer bounded segmented continuation from `epoch_0014` rather than a fresh from-parent restart.
