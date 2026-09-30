# tok_c_residual_adapter Decision

- Decision date:
  - `2026-06-12`
- Current status:
  - `recalibration_needed`
- Decision:
  - do not keep `tok_c_residual_adapter` in the active formal slot for now
  - its first strict new-data bracket is already informative:
    - `batch=8` under-fills late
    - `batch=9` overshoots late
- Why:
  - unlike the earliest tokenizer-family launches, this family has already proved it can truly enter a formal lane
  - but it still lacks a stable batch that stays inside the requested contract for a full retained checkpoint
- Next action:
  - keep it in `recalibration_needed`
  - continue the tokenizer tail with the next untried family first
  - revisit `tok_c_residual_adapter` later only if a new memory-saving change or a different cadence changes the `8/9` bracket
