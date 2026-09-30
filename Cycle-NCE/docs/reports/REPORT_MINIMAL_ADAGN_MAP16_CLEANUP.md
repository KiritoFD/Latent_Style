# Minimal AdaGN + Map16 Cleanup

## Scope
This cleanup prunes the training/inference stack to a minimal effective path:

- Style injection: `AdaGN` (global style code) + single `16x16` pre-spatial inject.
- Objective: `struct + edge + stroke_gram + color_moment + delta_tv`.
- Remove non-contributing branches to improve stability and attribution.

## What Was Kept

1. Main style path in `src/model.py`:
- Style code from `style_id`/`style_ref` to `AdaGN` in residual blocks.
- `16x16` style map pre-add before body blocks.

2. Effective losses in `src/losses.py`:
- `w_struct`, `w_edge`, `w_stroke_gram`, `w_color_moment`, `w_delta_tv`.

## What Was Removed

1. Model-side dead/non-effective paths (pruned from active graph):
- Decoder spatial inject path.
- Output affine path.
- Delta gate path.
- Delta highpass bias path.
- Texture head path.
- `map_32` style spatial branch.
- `step_schedule` interface removed end-to-end (model/inference/eval/config).
- NCE projector (`projector_dim` / `project_tokens`) removed.
- Legacy inference no-op knobs removed (`temperature_*`, `cfg_*`, `source_repulsion`, ternary-guidance placeholder).
- Unused sampler compatibility wrapper removed from `src/utils/inference.py`.

2. Loss-side removed branches:
- Distill/code/cycle/push/nce/semigroup/style_spatial_tv.

3. Logging/CSV noise:
- Removed legacy metric columns from training logs in `src/trainer.py`.
- Simplified epoch log line in `src/run.py`.

## Mandatory Fixes Implemented

1. Compile config now truly wired from config in `src/trainer.py`:
- `compile_backend`
- `compile_mode`
- `compile_fullgraph`
- `compile_disable_cudagraphs`

2. `map_32` compute waste removed:
- Style spatial maps now only carry `map_16`.
- No decoder-side spatial map forwarding.

3. Transfer-mask semantics fixed for style stats:
- `stroke_gram` and `color_moment` are now transfer-masked in `src/losses.py`.
- Prevents same-domain samples from forcing target-style statistics.

## Config Direction

`src/config.json` is aligned to minimal training:

- Model: core architecture fields + `style_spatial_pre_gain_16`.
- Loss: only the five effective loss weights and related core options.
- Training: compile options are explicit and effective.

## Notes

- This change prioritizes clean attribution and stable optimization over backward compatibility with all historical ablation keys.
- Old checkpoints can still be loaded by key-compatible paths; removed branches are no longer part of active computation.
- Multi-step integration is now explicit uniform averaging (`1 / num_steps`) with no hidden schedule override path.
