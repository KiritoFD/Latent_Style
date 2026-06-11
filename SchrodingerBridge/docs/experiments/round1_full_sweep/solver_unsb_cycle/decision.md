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
- Read:
  - the family is no longer in calibration mode
  - `epoch_0003` is the first clear solver-style best point so far:
    - LPIPS improved materially from the opening
    - all-pairs style also improved
  - `epoch_0004` then gave a mild rollback from that point
  - all future keep/reject decisions should now be made from the all-ckpt remote fast curve

## Promotion Rule

- No early promotion from the first in-band health sample alone.
- Require the same family closure package as every other round-1 family before any keep/reject claim.
