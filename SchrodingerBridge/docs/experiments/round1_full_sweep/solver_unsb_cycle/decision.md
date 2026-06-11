# solver_unsb_cycle Decision

- Decision date:
  - `2026-06-11`
- Current status:
  - `running`
- Current read:
  - the first under-band `batch=8` opening has been superseded
  - the current canonical launch is `batch=15`
  - this family now holds the remote formal lane

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
  - `epoch_0013`
  - transfer `0.6963 / 0.5094`
  - full `0.7158 / 0.4977`
- Current interpretation:
  - `epoch_0006-0007` had made the line look like a continuing rollback from `epoch_0003`
  - `epoch_0008` first recovered materially on both style and LPIPS relative to `epoch_0007`
  - `epoch_0009` then converts that rebound into a true new Pareto point
  - it now owns the best all-pairs `CLIP-S` and the best LPIPS point in this family
  - `epoch_0010` softens from that new frontier, but does not erase it
  - `epoch_0011` softens again, so the post-`epoch_0009` tail currently looks like follow-up drift rather than stable frontier occupancy
  - `epoch_0011` also carries a much slower eval wall time than the surrounding checkpoints
  - `epoch_0012` rebounds from `epoch_0011`, but it still does not reclaim the `epoch_0009` frontier
  - `epoch_0013` remains in the same partial-recovery band as `epoch_0012`, rather than extending into a real frontier recapture
- Operational consequence:
  - fully reset the solver patience read
  - do not treat this family as approaching closure any more
  - keep the formal lane open until the post-`epoch_0009` tail is actually observed
  - the immediate decision hinge is whether `epoch_0014+` continue climbing back toward `epoch_0009` or whether `epoch_0012-0013` were only a weak recovery shelf

## Promotion Rule

- No early promotion from the first in-band health sample alone.
- Require the same family closure package as every other round-1 family before any keep/reject claim.
