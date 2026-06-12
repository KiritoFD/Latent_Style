# tok_baseline_global Plan

- Wave: `wave1_tokenizer`
- Axis: `tokenizer`
- Notes: Wave-1 baseline with the legacy global code only. No DINO sidecar and no stochastic bridge.
- DINO policy: archived-only unless overwhelming gain appears.
- Validation:
  - local trainer init passes under `freeze_mode=style_branch`
