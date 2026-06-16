# I2SB Latent Slerp Path Probe

Date: 2026-06-16

## Goal

Test the `fiber.md` diagnosis that straight-line latent interpolation sends
training states through off-manifold regions, encouraging endpoint averaging
and weak style actuation. This is a path-geometry probe only; it does not add
new tokenizer capacity, output residual heads, DINO, VLM, PC solver, or
postprocess.

## Controlled Delta

- Parent:
  `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`.
- Candidate config:
  `configs/aaai2027/phase2_i2sb_latent_slerp_k070_e3_sigma0p02_b8a2_vlen010.json`.
- Matched control:
  `configs/aaai2027/phase2_i2sb_clean_k070_e3_sigma0p02_b8a2_vlen010.json`.
- Held fixed:
  pure latent spatial tokenizer, TopoGate k070, `solver_i2sb`, endpoint
  absolute prediction, `bridge_sigma=0.02`, exact Brownian bridge schedule,
  terminal SWD, no style delta adapter, no proximal path, b8 accumulation-2,
  and fast10 transfer-only in-loop eval.
- Only bridge/path delta:
  `bridge.bridge_path_mode=latent_slerp`.
- Implementation guard:
  nonlinear path mode is allowed only for endpoint transport. Velocity mode
  raises instead of silently using an incorrect path derivative.

## Decision Rule

- Primary metric: transfer CLIP-S, style-first toward `0.74`.
- LPIPS budget: up to `0.35` is acceptable only if style clearly rises; above
  that is diagnostic but not promotable.
- Positive mechanism evidence requires a matched gain over clean I2SB:
  higher transfer style at comparable LPIPS, or materially lower LPIPS at
  comparable style.
- Negative evidence:
  same style/LPIPS curve as clean I2SB means path geometry is not the active
  bottleneck under the current endpoint objective;
  lower style with no LPIPS gain closes the lane early.
- Formal convergence:
  keep all retained checkpoints evaluated with CLIP-S + LPIPS;
  do not close while best style is in the newest two checkpoints;
  stop after four later retained checkpoints fail to create a new transfer
  Pareto point and the tail is near-flat.

## Runtime Observability

Required in training CSV:

- `bridge_path_slerp_active=1`
- `bridge_sigma=0.02`
- `bridge_noise_schedule_exact=1`
- endpoint/base/final endpoint magnitudes
- fast10 eval wall time

## Launch Log

- 2026-06-16 08:48 first remote launch was stopped immediately because the
  launcher resolved the inherited `../exp/...` resume path incorrectly and the
  trainer reported `No checkpoint found, start from scratch`. This invalid
  scratch directory/log was removed and is not used as evidence.
- 2026-06-16 08:52 remote WSL formal run restarted after explicitly setting
  `training.resume_checkpoint=./exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`.
- Remote PID: `21176` wrapper, `21183` python process at launch.
- Parent load confirmed:
  `loaded=272 skipped=0 missing=0 unexpected=0`.
- Health check:
  GPU memory `3085 MiB`, below the old preferred band but accepted because
  this b8a2 endpoint lane is compute-active and low memory is not a stop
  condition; high-memory guard remains active at `11300 MiB`.
- First train read:
  loss decreases from early `9.6472` to about `10.2274` around step 60/236,
  with nonzero endpoint flow and terminal SWD. Await e1 in-loop eval before
  any mechanism decision.

## Artifacts

- Local curve target:
  `docs/experiments/phase2_fiber_bundle/curves/i2sb_latent_slerp_k070_e3_fast10_curve.csv`.
- Local eval mirror target:
  `docs/experiments/phase2_fiber_bundle/eval/i2sb_latent_slerp_k070_e3_sigma0p02_b8a2_vlen010/`.

## Running Eval Curve

Local mirror:
`docs/experiments/phase2_fiber_bundle/eval/i2sb_latent_slerp_k070_e3_sigma0p02_b8a2_vlen010/full_eval_fast10/clip_lpips_curve.csv`

| epoch | transfer CLIP-S | transfer LPIPS | eval wall | read |
|---|---:|---:|---:|---|
| e1 | 0.709182 | 0.545727 | 44.06s | style high, structure damaged |
| e2 | 0.712038 | 0.476511 | 25.30s | current best style and matched-control Pareto gain |
| e3 | 0.704485 | 0.447166 | 24.99s | LPIPS improves, style retreats |
| e4 | 0.695003 | 0.453878 | 24.94s | style continues to retreat |
| e5 | 0.697559 | 0.425856 | 24.93s | lower-LPIPS tradeoff point, still style-retreated |
| e6 | 0.698255 | 0.474109 | 24.98s | no recovery |
| e7 | 0.694678 | 0.391787 | 25.02s | new lower-LPIPS Pareto point, style too low |
| e8 | 0.698919 | 0.406078 | 24.90s | partial style recovery, not Pareto |
| e9 | 0.691482 | 0.394639 | 24.93s | no improvement |
| e10 | 0.701837 | 0.385366 | 24.86s | new lower-LPIPS Pareto, best structure-side point |
| e11 | 0.686676 | 0.396227 | 24.89s | no improvement |
| e12 | 0.689118 | 0.384867 | 24.90s | tiny LPIPS-only Pareto |
| e13 | 0.690317 | 0.369346 | 25.62s | new structure-side Pareto, style still low |

Matched read against clean absolute I2SB sigma0.02:

- Clean e2: `0.709094 / 0.490233`.
- Slerp e2: `0.712038 / 0.476511`.
- Delta at e2: `+0.002944` style and `-0.013722` LPIPS.

Interim decision:

- `early_positive_not_closed`.
- e2 is the first path-geometry point that improves both transfer style and
  LPIPS against its clean I2SB matched control.
- Do not promote yet: LPIPS remains far above the desired `0.30-0.35` band,
  and e3/e4 show style retreat. e5 creates a lower-LPIPS tradeoff point but
  remains style-retreated. e7 pushes LPIPS to `0.391787` but style falls to
  `0.694678`; e10 improves the structure-side Pareto to
  `0.701837 / 0.385366`; e13 pushes LPIPS further to `0.369346` but only with
  `0.690317` style. The curve now cleanly separates a style peak (e2) from a
  structure peak (e13). Continue because the latest Pareto point is still in
  the newest checkpoints.
