# Model Quality Diagnosis, Research, and Fixes

## Goal
Reduce checker/grid artifacts **without** trading off into blurry outputs.

## Code-Level Diagnosis
The following issues were found in the current training/inference path and are likely to hurt visual quality.

1. Style losses were not applied to deployment path output
- In `src/losses.py`, style losses (`stroke_gram`, `color_moment`) were computed against `pred_teacher` (reference-conditioned), while deployment uses `pred_student` (style-id only).
- Relevant locations:
  - `src/losses.py`: `pred_teacher` / `pred_student` construction.
  - `src/losses.py`: style loss block previously used `pred_teacher.float()`.
- Risk:
  - Student can under-learn high-frequency style quality and appear softer/noisier than teacher.

2. NCE token resize used fixed `area` downsampling
- In `src/losses.py`, `calc_nce_loss()` resized both tensors with `mode="area"`.
- Relevant location:
  - `src/losses.py`: `x_in = F.interpolate(..., mode="area")`, same for `x_out`.
- Risk:
  - For small token grids, this can bias toward blocky/averaged local signals.

3. Spatial style template jitter used circular wrap-around
- In `src/model.py`, style-id spatial map jitter used `torch.roll`.
- Relevant location:
  - `src/model.py`: `encode_style_spatial_id()`.
- Risk:
  - Wrap-around introduces periodic boundary continuity that is unnatural for non-periodic image structure and can reinforce tiled texture behavior.

## External Research
1. Checkerboard artifacts from uneven sampling/upsampling overlap:
- Distill: *Deconvolution and Checkerboard Artifacts*  
  https://distill.pub/2016/deconv-checkerboard

2. Anti-aliasing improves shift stability and quality robustness:
- Zhang 2019 (ICML): *Making Convolutional Networks Shift-Invariant Again*  
  https://proceedings.mlr.press/v97/zhang19a.html

3. Perceptual/style losses are key to visual quality beyond pixel matching:
- Johnson et al. 2016: *Perceptual Losses for Real-Time Style Transfer and Super-Resolution*  
  https://arxiv.org/abs/1603.08155

## Implemented Fixes
1. Style loss source made explicit and switched to student by default
- Added config key: `loss.style_loss_source` in `{"student", "teacher"}`.
- Default now targets `student`.
- Code changes:
  - `src/losses.py`: selects `style_pred` based on `style_loss_source`.
  - Style loss branches now consume `style_pred`.

2. NCE resize mode made configurable
- Added config key: `loss.nce_resize_mode`.
- Supports `area`, `bilinear`, `bicubic`, `nearest`, `nearest-exact`.
- Default set to `bilinear`.
- Code changes:
  - `src/losses.py`: `calc_nce_loss(..., resize_mode=...)`.

3. Spatial jitter changed from wrap-around to reflect-pad crop
- Kept anti-grid jitter behavior, removed periodic roll behavior.
- Code changes:
  - `src/model.py`: `encode_style_spatial_id()` jitter now uses reflection padding + random crop.

4. Config defaults aligned with quality goal
- Updated `src/config.json` with:
  - `"style_loss_source": "student"`
  - `"nce_resize_mode": "bilinear"`

## Why these fixes are quality-oriented (not just “artifact suppression”)
1. Student-targeted style supervision trains the actual deployed output path.
2. Bilinear NCE tokenization is less blocky than area pooling at small spatial sizes.
3. Reflective jitter regularizes fixed spatial templates without injecting periodic wrap artifacts.

## Validation Performed
1. `python -m compileall src` passed.
2. Forward smoke test (`LatentAdaCUT` + `AdaCUTObjective.compute`) passed after fixes.

## Recommended Next Checks (during training)
1. Track `stroke_gram`, `edge`, `delta_tv`, `style_spatial_tv` curves together.
2. Compare early visual snapshots at epochs 5/10/20 for:
- local grid frequency
- edge crispness
- texture naturalness
3. If still slightly blocky:
- lower `style_spatial_dec_gain_32/out` a bit further
- keep `w_edge` stable while reducing `w_delta_tv` before reducing style terms
