# Selected Style Metrics Helper Smoke

Date: 2026-06-03

Purpose:

- verify that the current `tools/eval_selected_style_metrics.py` still runs on a
  real legacy bundle after the local CLIP-path and ArtFID helper changes;
- define the contract boundary so this helper is not confused with the formal
  `full_eval` protocol.

## Files under test

- `src/utils/artfid_metric.py`
- `tools/eval_selected_style_metrics.py`

## Smoke command

The helper was exercised by importing the tool module and evaluating one real
archive bundle:

```text
bundle:
archives/old_experiment_dirs/grid_search_3epoch/
  S-none_K-1_C-0_W-20_Col-0/full_eval/epoch_0008/images
summary:
archives/old_experiment_dirs/grid_search_3epoch/
  S-none_K-1_C-0_W-20_Col-0/full_eval/epoch_0008/summary.json
device:
cuda
batch_size:
16
enable_artfid:
true
```

## Observed result

Returned row:

```json
{
  "method": "OursArchive",
  "run": "legacy256_e8_archive",
  "images": 750,
  "clip_style_up": 0.7036010053555171,
  "lpips_down": 0.44816957577333333,
  "fid_down": 237.41816847581012,
  "clip_fid_down": 0.5317947815685663,
  "artfid_down": 290.3882900938975,
  "artfid_fid_down": 206.09614389256652,
  "artfid_content_lpips_down": 0.4021907150745392,
  "artfid_error": ""
}
```

Archive target-wise ArtFID reference for the same bundle:

```text
aggregate_art_fid_fid         = 208.24250822979752
aggregate_art_fid_content_lpips = 0.40219072699546815
aggregate_art_fid             = 293.018483138329
```

## Interpretation

### What is verified

1. The helper runs end to end on a real 750-image bundle.
2. The resized image loading path in `src/utils/artfid_metric.py` is behaving
   sensibly.
3. The content-distance branch is aligned closely enough to the archive
   reference:
   - helper: `0.4021907150745392`
   - archive: `0.40219072699546815`
4. The local CLIP-source resolution fallback does not break evaluation.

### What is not claimed

This helper is **not** an exact replacement for the formal `full_eval`
ArtFID pipeline.

Observed mismatch:

- helper `artfid_fid_down`: `206.0961`
- archive `aggregate_art_fid_fid`: `208.2425`
- helper `artfid_down`: `290.3883`
- archive `aggregate_art_fid`: `293.0185`

Likely reason:

- `tools/eval_selected_style_metrics.py` is a compact comparison helper with its
  own reference-image discovery and summary logic;
- the formal archive numbers come from the dedicated full-eval pipeline and its
  saved target-wise aggregation;
- therefore the helper should be treated as a **selected-style comparison
  utility**, not as the authoritative formal-eval reporter.

## Promotion rule

It is safe to keep and commit these helper changes only under this contract:

1. `tools/eval_selected_style_metrics.py` may be used for quick comparison
   tables and helper analyses;
2. any paper-facing formal ArtFID claim must still cite the dedicated
   `aggregate_targetwise_artfid.json` or the main `full_eval` outputs;
3. do not present this helper as a protocol-identical recreation of the formal
   evaluator.
