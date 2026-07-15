# tok_d_vlm_prompt Decision

- Decision date:
  - `2026-06-12`
- Current status:
  - `recalibration_needed`
- Decision:
  - do not keep `tok_d_vlm_prompt` in the active formal slot
  - its first strict new-data bracket is already informative:
    - `batch=8` overshoots
    - `batch=7` under-fills
- Why:
  - unlike the earliest tokenizer-family attempts, the current failure is not caused by missing DINO cache or the old patch-grid reshape bug
  - the line now fails for the same fundamental reason as `tok_a_dino_dict`:
    - no clean in-band batch has been found yet
- Next action:
  - keep the family in `recalibration_needed`
  - prioritize the tokenizer family that already made it into a formal lane
  - only reopen `tok_d_vlm_prompt` after the active tokenizer lead line closes or after a concrete memory-saving change alters the `7/8` bracket
