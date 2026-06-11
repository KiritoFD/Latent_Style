# solver_tangent_rk Decision

- Decision date:
  - `2026-06-11`
- Current status:
  - `reviewing`
- Current gate:
  - remote training is closed
  - next required work is shortlisted local deep review plus frozen `VLM`

## Formal Setting

- Canonical family:
  - `solver_tangent_rk`
- Final training policy:
  - `batch=17`
  - `virtual_length_multiplier=0.5`
  - short continuation segments after the first long tail was observed
- Final settled authority span:
  - `epoch_0001` through `epoch_0032`

## Curve Summary

- Best transfer `CLIP-S`:
  - `epoch_0001`
  - transfer `0.6999 / 0.5295`
- Best transfer `LPIPS`:
  - `epoch_0019`
  - transfer `0.6909 / 0.4498`
- Strongest all-pairs late frontier:
  - `epoch_0007`
  - full `0.7159 / 0.4675`
- Final settled point:
  - `epoch_0032`
  - transfer `0.6893 / 0.4807`
  - read: clearly below the late best frontier

## Closure Read

- The line produced a meaningful late solver frontier at `epoch_0019`.
- Every short continuation after that failed to create a new Pareto point.
- The tail through `epoch_0025-0032` behaved like long oscillatory drift rather than renewed frontier search.
- Training was therefore closed on trajectory evidence, not because the family became externally promotable.

## Decision

- Keep as `reviewing`, not `running`.
- Do not reopen training unless deep review exposes a contradiction large enough to justify another bounded continuation.
- Do not promote on fast-curve evidence alone:
  - this family still needs `IntroStyle`, `DINO`, and frozen `VLM` review before any keep/reject conclusion
  - even its best internal point is not enough by itself to claim it beats the current external board

## Next Action

- Use the existing bestfew handoff for local deep review.
- Freeze the stage-close board after `IntroStyle + DINO + VLM` are written.
- Keep the remote formal lane on `solver_pc` while this family remains in review only.
