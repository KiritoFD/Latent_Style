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
  - `epoch_0008`
  - transfer `0.6955 / 0.5184`
  - full `0.7121 / 0.5088`
- Current interpretation:
  - `epoch_0006-0007` had made the line look like a continuing rollback from `epoch_0003`
  - `epoch_0008` then recovered materially on both style and LPIPS relative to `epoch_0007`
  - this is still not enough to create a new Pareto point over `epoch_0003`
  - but it is enough to keep the family open under the solver `6`-checkpoint patience rule
- Operational consequence:
  - do not close the family yet
  - wait for at least the next settled checkpoint before making a stronger long-tail judgement

## Promotion Rule

- No early promotion from the first in-band health sample alone.
- Require the same family closure package as every other round-1 family before any keep/reject claim.
