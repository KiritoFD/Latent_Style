# Local Generated Media Owner Review Pass 5 - 2026-06-05

This pass re-opened the local generated-media items that were still pending
after pass4. The candidate list was not treated as proof. Each exact directory
was opened again, with git tracking/ignore state, nearby evidence, and
replacement packets checked before any cleanup decision.

No cleanup was performed in this pass.

## Reviewed Paths

### Seedream Gap Diagnostic Inputs

- `SchrodingerBridge\exp\diagnostics\seedream_gap`

Manual open result:

- The directory contains `inputs\` with 7 child image sets.
- Each child set contains exactly 750 JPG files and no
  `json/csv/md/txt/log/yaml` evidence files.
- Total payload is 5,250 JPG files, about `79.523 MB`.
- `git ls-files` returns no tracked files for this target.
- `git check-ignore` matches `.gitignore:77` via `SchrodingerBridge/exp/**`.
- Parent diagnostics evidence exists:
  `wikiart512_5style_source_clip_geometry.md/json` and
  `wikiart_candidate_style_clip_separability.md/json`.
- A representative JPG was visually opened:
  `ema_sconv_support_w40_style_e6\cezanne_00057_to_cezanne.jpg`.

Decision: `retain_pending_owner`.

Reason: these are diagnostic image inputs for the Seedream gap investigation.
The parent directory has CLIP geometry/separability evidence, but the target
itself has no summary, metrics, log, config, or duplicate proof. Deleting it
would discard potentially useful diagnostic lineage without an owner decision.

### Inference Parameter Sweeps

- `SchrodingerBridge\exp\inference\inference_param_sweep_t01e8_quick`
- `SchrodingerBridge\exp\inference\inference_param_sweep_t01e8_fine`

Manual open result:

- `quick` contains 14 parameter subdirs, each with `images\` and 250 JPG files.
- `quick` total is 3,500 JPG files, about `40.843 MB`.
- `fine` contains 8 parameter subdirs, each with `images\` and 250 JPG files.
- `fine` total is 2,000 JPG files, about `22.361 MB`.
- Neither directory contains `json/csv/md/txt/log/yaml` evidence files.
- `rg` found only archaeology references outside these targets, not paper or
  current docs usage.
- `git ls-files` returns no tracked files for both targets.
- `git check-ignore` matches `.gitignore:77` via `SchrodingerBridge/exp/**`.
- A representative fine-sweep image was visually opened:
  `endpoint_s0p60\images\cezanne_00123_to_vangogh.jpg`.

Decision: `retain_pending_owner`.

Reason: these are small parameter-behavior preview sweeps. They are not closed
metric/timing packets, but there is also no duplicate proof and no owner
authorization to remove them. If they are deleted later, the parameter list and
representative grids should be preserved first.

### CUT Native Raw Web Outputs

- `Related_Works\runs\cut_5x5\raw_results`
- `Related_Works\runs\cut_5x5\raw_results_val`

Manual open result:

- `raw_results` contains 5 target dirs: `cut_to_cezanne`, `cut_to_Hayao`,
  `cut_to_monet`, `cut_to_photo`, and `cut_to_vangogh`.
- Each `raw_results` target dir contains a native CUT `test_latest\index.html`
  plus 750 JPG web images referenced as `real_A`, `fake_B`, and `real_B`.
- `raw_results` totals 3,750 media files, about `109.181 MB`.
- `git ls-files` shows 5 tracked `index.html` files under `raw_results`.
- The `raw_results` images are ignored by `.gitignore:71`
  `Related_Works/runs/**`.
- `raw_results_val` contains the same 5 target dirs under `val_latest`.
- Each `raw_results_val` target dir contains 750 PNG files plus one HTML file.
- `raw_results_val` totals 3,750 media files, about `453.017 MB`.
- `git ls-files` shows 3,755 tracked files under `raw_results_val`.

Replacement evidence was opened:

- `Related_Works\runs\cut_5x5\infer_5x5`
- `Related_Works\runs\cut_5x5\infer_val_clean_5x5`

Both curated dirs have `summary.json`, `metrics.csv`, `summary_grid.png`,
`meta.json`, logs, images, and a tiny `fake_eval_checkpoint.pt`. The summaries
record the CUT metric packets dated `2026-03-08 02:28:14` and
`2026-03-08 02:34:43`.

Decision: `retain_tracked_boundary`.

Reason: the curated packets preserve metrics, but they do not authorize raw
deletion. `raw_results` mixes tracked HTML with ignored images, and
`raw_results_val` is tracked repository media/html. Deleting either raw target
would modify tracked repository content and violates the current cleanup
boundary.

## Delete Whitelist

None.

## Cleanup Boundary

This pass did not delete files, did not touch paper TeX/PDF, did not modify
source code, and did not modify `Related_Works` or experiment output
directories. It only writes this manual pass document and the row-level CSV.

## Remaining Owner Decisions

- Decide whether `seedream_gap` should be retained as diagnostic lineage or
  migrated into a compact manifest plus representative samples.
- Decide whether `inference_param_sweep_t01e8_quick` and
  `inference_param_sweep_t01e8_fine` still matter for inference behavior
  archaeology; if not, preserve parameter metadata before deletion.
- Decide whether CUT raw web outputs should be migrated/untracked/deleted.
  That requires an explicit tracked-file policy, not a generated-media cleanup
  pass.
