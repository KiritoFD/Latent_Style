# attn_gw_ot Decision

- Decision date:
  - `2026-06-10`
- Status:
  - `recalibration_needed`
- This is not a reject:
  - current checkpoints are still worth local fast-eval for directional read
  - but they are not acceptable as formal round-1 closure evidence
- Why the current run is non-formal:
  - recent runtime samples remained `under_band`
  - runtime summary now records consecutive `under_band` samples and `nonformal_under_band`
  - a stale unrelated remote training/eval lane was found concurrently on the same `3060`
  - that breaks the single-lane paper-facing execution rule
- Required before any real family decision:
  - complete the deferred local fast-eval on the existing checkpoints after `SaMAM` converges
  - if the curve is weak, close as non-promoted exploratory evidence
  - if the curve is strong, relaunch `attn_gw_ot` as a fresh in-band single-lane formal run, then judge that rerun instead
