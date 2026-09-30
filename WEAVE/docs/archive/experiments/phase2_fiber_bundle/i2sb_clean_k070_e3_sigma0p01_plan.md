# Clean I2SB Endpoint Path: k070 e3 sigma0p01

## Purpose

Test whether the absolute-endpoint I2SB style actuation from `sigma=0.02`
can be kept while reducing latent drift and LPIPS. This is a single-variable
sigma scan, not a tokenizer/backbone/loss change.

## Controlled Change

- Matched control:
  `configs/aaai2027/phase2_i2sb_clean_k070_e3_sigma0p02_b8a2_vlen010.json`.
- Candidate:
  `configs/aaai2027/phase2_i2sb_clean_k070_e3_sigma0p01_b8a2_vlen010.json`.
- Changed:
  `bridge.bridge_sigma=0.02 -> 0.01`.
- Unchanged:
  parent checkpoint, absolute endpoint parameterization, I2SB objective,
  exact Brownian schedule, endpoint time floors, pure latent spatial tokenizer,
  TopoGate k070, appearance alignment, semantic cross-attention, terminal SWD,
  b8a2 schedule, vlen `0.10`, and fast10 transfer eval.

## Prior Evidence

| lane | best/last read | transfer CLIP-S | transfer LPIPS | decision |
| --- | --- | ---: | ---: | --- |
| absolute sigma0p02 | e2 peak | 0.709094 | 0.490233 | style-positive, LPIPS too high |
| absolute sigma0p02 | e5 stop | 0.704671 | 0.408530 | style reversal, still out of band |
| residual sigma0p02 | e2 | 0.673869 | 0.308784 | structure-only, style negative |

## Eval Contract

- Training-time eval subdir: `full_eval_fast10`.
- Transfer-only, `10` source samples per style.
- `CLIP-S + LPIPS` every retained checkpoint.
- Generated-delta observability enabled.
- Training-time eval must remain subprocess-isolated:
  `full_eval_in_process=false`, `full_eval_runtime_model_cache=false`.
- Offline all-ckpt sweeps may use:
  `run_evaluation.py <run_dir> --batch_in_process --runtime_model_cache`.

## Decision Rule

- Continue while transfer CLIP-S is near or above the absolute sigma0p02
  style band and LPIPS trends down.
- Promote only if style remains materially above the k070/predec frontier
  while LPIPS approaches the accepted Seedream-like tolerance band.
- Stop if style falls below `0.700` for two consecutive retained checkpoints,
  because residual already proved low LPIPS without style is not useful.
- Stop if LPIPS remains above `0.42` after the style peak reverses, since that
  repeats the sigma0p02 failure at lower noise.

## Launch Notes

- Remote WSL repo:
  `/mnt/i/Github/Latent_Style/SchrodingerBridge`.
- Expected memory should be similar to the clean sigma0p02 lane; low VRAM is
  acceptable if throughput is healthy.
- First health check: process alive, log progressing, no eval cache flags in
  training config.

## Live Run Log

- 2026-06-16 04:19 remote WSL launch:
  `python src/run.py --config configs/aaai2027/phase2_i2sb_clean_k070_e3_sigma0p01_b8a2_vlen010.json`.
- PID: `15178`.
- Train log:
  `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_phase2_i2sb_clean_k070_e3_sigma0p01_b8a2_vlen010_train.log`.
- Health check: epoch 1 active, GPU around `3.25-3.57 GB`, no in-process eval
  cache, training peak logged as `2.50/2.87GB`.

## Live Fast10 Transfer Curve

Curve CSV:
`docs/experiments/phase2_fiber_bundle/curves/i2sb_clean_k070_e3_sigma0p01_fast10_curve.csv`.

| epoch | transfer CLIP-S | transfer LPIPS | eval wall | read |
|---:|---:|---:|---:|---|
| 1 | `0.713162` | `0.590598` | `26.78s` | new style-high, LPIPS worse |
| 2 | `0.709784` | `0.524506` | `29.16s` | LPIPS falling, style already reversing |
| 3 | `0.701776` | `0.482099` | `26.23s` | continued style reversal, LPIPS still out of band |

## Interim Read

`sigma=0.01` did not reduce latent drift; it amplified the style/destruction
tradeoff. The e1 style point is the strongest fast10 style read so far, but
style then reverses for e2/e3 while LPIPS remains far above the accepted band.

## Closure

- Remote process stopped after e3 on 2026-06-16.
- Stop reason:
  style peak reversed for two later checkpoints and LPIPS remained above
  `0.42`, matching the documented negative rule.
- Decision:
  `closed_negative_lower_sigma_high_lpips_style_reversal`.
- Interpretation:
  decreasing I2SB sigma from `0.02` to `0.01` does not fix the absolute
  endpoint coordinate drift. It can push CLIP-S higher (`0.713162`), but the
  price is too destructive (`0.590598` LPIPS at peak) and the advantage decays.
- Next decision:
  do not continue pure absolute sigma scans downward as the immediate path.
  The clean next mechanism should preserve absolute endpoint actuation while
  adding a content anchor, such as an explicit absolute/residual endpoint blend
  switch or a bounded endpoint delta, tested as one new controlled mechanism.
