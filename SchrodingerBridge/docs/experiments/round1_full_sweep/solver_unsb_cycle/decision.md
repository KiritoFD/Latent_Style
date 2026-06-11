# solver_unsb_cycle Decision

- Decision date:
  - `2026-06-11`
- Current status:
  - `recalibration_needed`
- Current read:
  - the first formal launch attempt was made
  - it opened far below the required formal VRAM band
  - so the next decision boundary is calibration, not keep/reject

## Launch Decision

- Keep as the next solver-family handoff candidate.
- Rationale:
  - `solver_pc` training phase is already closed
  - `solver_unsb_cycle` already has:
    - canonical config
    - queue slot
    - smoke artifact
    - local/remote doc roots
  - but the first `batch=8` opening only used about `5223 MiB`
  - so it cannot yet count as a formal paper-facing lane

## Expected First Judgment

- The first real decision is now the next calibration target above `batch=8`.
- First measured read:
  - `5223 MiB / 12288 MiB`
  - `epoch 1/48`
  - read: strongly under-band
- Current rule:
  - increase the effective batch until the lane enters the formal `9.0-10.8 GiB` band
- If in-band:
  - promote to `running`
  - let the all-ckpt remote fast curve become the authority

## Promotion Rule

- No early promotion from smoke or first-point optimism.
- Require the same family closure package as every other round-1 family before any keep/reject claim.
