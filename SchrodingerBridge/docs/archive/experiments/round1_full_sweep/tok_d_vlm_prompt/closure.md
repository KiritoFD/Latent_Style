# tok_d_vlm_prompt Closure

- Status: `recalibration_needed`
- Current read:
  - this family now shares the same fixed new-data DINO cache path as the other tokenizer families
  - it no longer fails on the earlier patch-grid reshape bug
  - its first real strict-band bracket is:
    - `batch=8`
      - entered training
      - later hit runtime guard at about `11896MiB`
      - above the `11.3GiB` hard cap
    - `batch=7`
      - entered training
      - but 180-second health read stayed about `8532MiB`
      - below the formal floor
- Closure consequence:
  - keep `tok_d_vlm_prompt` in `recalibration_needed`
  - this is now a real VRAM bracket problem, not a missing-data or tokenizer-shape bug
  - do not treat the line as formally exercised beyond this strict `7/8` calibration read
