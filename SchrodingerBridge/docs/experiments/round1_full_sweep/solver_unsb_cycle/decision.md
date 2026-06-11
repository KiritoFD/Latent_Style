# solver_unsb_cycle Decision

- Decision date:
  - `2026-06-11`
- Current status:
  - `planned`
- Current read:
  - this family has switch-smoke evidence only
  - it has not yet consumed the formal remote lane
  - the next decision boundary is launch readiness, not keep/reject

## Launch Decision

- Keep as the next solver-family handoff candidate.
- Rationale:
  - `solver_pc` is still the active formal lane
  - `solver_unsb_cycle` already has:
    - canonical config
    - queue slot
    - smoke artifact
    - local/remote doc roots
  - so there is no reason to block on more setup before handoff

## Expected First Judgment

- The first real decision is whether `batch=8` lands in the formal VRAM band.
- If under-band:
  - treat it as calibration only
  - keep the family `planned` or move to `recalibration_needed` depending on the launch outcome
- If in-band:
  - promote to `running`
  - let the all-ckpt remote fast curve become the authority

## Promotion Rule

- No early promotion from smoke or first-point optimism.
- Require the same family closure package as every other round-1 family before any keep/reject claim.
