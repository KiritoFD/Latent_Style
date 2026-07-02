# Timing Artifact Prune

Date: 2026-06-03

Purpose:

- reduce local disk waste in timing-only experiment surfaces;
- preserve timing summaries, logs, and provenance;
- avoid breaking current paper-facing citations.

## Rule used

This prune followed the policy in:

- `docs/experiments/2026-06-03-exp-surface-classification.md`

Operational rule:

1. keep any timing directory whose exact subdirectory name is cited in current
   docs or paper sources;
2. for zero-hit timing subdirectories, delete only generated image payloads;
3. keep:
   - `summary.json`
   - `metrics.csv`
   - wall-time text files
   - train/eval logs
   - config snapshots

## Pruned surfaces

### `exp/timing_20260602`

Deleted only `images/` directories from zero-hit timing subdirectories:

- `live_windows_png750_b12_nogrid`
- `live_wsl_png750_b12_nogrid`
- `run_eval_opt150_b8_v2_w8_png`
- `run_eval_opt150_b8_v2_w8_png_nogrid`
- `run_eval_opt150_b8_v4_w8_png`
- `run_eval_opt150_b8_v2_w4_png`
- `run_eval_opt150_b12_v2_w8_png`
- `run_eval_opt150_b12_v2_w8_png_nogrid`
- `run_eval_opt150_b15_v2_w8_png_nogrid`
- `run_eval_opt150_b12_v4_w8_png_nogrid`
- `run_eval_opt150_b8_v2_w8_tvpng`
- `run_eval_src_uint8_ema_generate750_b8_t5_vaebs2`
- `run_eval_src_uint8_smoke25`
- `run_eval_opt150_b8_v2_w8_piljpg`
- `run_eval_opt150_b8_v2_w4_piljpg`
- `run_eval_src_uint8_ema_smoke25`

Retained because they are still cited by current notes:

- `run_eval_png750_b12_v2_w8_grid`
- `run_eval_png750_b12_v2_w8_nogrid`
- `lancet_from_scratch_e8_full_eval_direct750`
- `lancet_from_scratch_e8_generate750`

Size change:

- before:
  - `8437` image files
  - `1089.96 MB`
- after:
  - `3762` image files
  - `732.68 MB`
- reclaimed:
  - about `357.28 MB`

### `exp/timing_20260601`

Deleted zero-hit generated image files from:

- `lancet_generate750_b2_tchunk5_vaebs2`
- `lancet_generate150_b2_tchunk5`
- `lancet_generate150_b2_tchunk5_vaebs2`
- `lancet_generate150_b2`

Retained because the directory name is still cited:

- `lancet_fulleval750_b2_tchunk5_vaebs2`

Size change:

- before:
  - `1951` image files
  - `936.39 MB`
- after:
  - `751` image files
  - `363.20 MB`
- reclaimed:
  - about `573.19 MB`

## Net effect

Approximate reclaimed local disk:

- `930.47 MB`

## Git/worktree effect

This prune did not dirty the tracked git worktree because the removed timing
image payloads were not tracked research-source artifacts.

That is the desired pattern for future cleanup:

- shrink runtime bulk first;
- preserve summaries and logs;
- record the prune in `docs/experiments/`.
