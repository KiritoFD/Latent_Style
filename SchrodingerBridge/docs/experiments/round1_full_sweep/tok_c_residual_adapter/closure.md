# tok_c_residual_adapter Closure

- Status: `recalibration_needed`
- Current read:
  - this family is the first tokenizer-tail line that actually entered a clean formal lane on the new-data DINO path
  - but its late-train memory drift still produced a strict bracket instead of a stable converged lane:
    - `batch=8`
      - entered formal in-band early
      - later fell to about `8313MiB`
      - hit late-train `under_band` stop before the first retained checkpoint landed
    - `batch=9`
      - entered training again
      - later rose to about `11898MiB`
      - hit the hard cap before the first retained checkpoint landed
- Closure consequence:
  - `tok_c_residual_adapter` is now another real strict-band bracketed tokenizer family
  - keep it in `recalibration_needed`
  - do not treat this line as formally exercised beyond the `8/9` bracket until a new memory tradeoff is chosen
