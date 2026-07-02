# tok_b_cross_image Decision

- Decision date:
  - `2026-06-12`
- Current status:
  - `recalibration_needed`
- Decision:
  - do not keep `tok_b_cross_image` in the active formal slot for now
  - the family has one real strict formal read:
    - `batch=8` enters the band early, then drifts under-band late
  - later retries at `batch=9/10` are currently polluted by run-root `OSError: [Errno 5] Input/output error`
- Why:
  - this means the next useful move is no longer "just keep increasing batch"
  - we first need either:
    - a cleaner run-root / launcher path for this family, or
    - a different preliminary stage such as warmstart / reconstruction-pretrain
- Next action:
  - keep `tok_b_cross_image` in `recalibration_needed`
  - prioritize the tokenizer line that already holds a clean formal lane
  - revisit `tok_b` after the current tokenizer lead line closes or after the I/O issue is explicitly neutralized
