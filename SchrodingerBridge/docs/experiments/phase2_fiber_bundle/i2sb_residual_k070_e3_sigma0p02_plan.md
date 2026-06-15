# Residual I2SB Endpoint Path: k070 e3 sigma0p02

## Purpose

Follow the clean absolute-endpoint I2SB lane with one controlled change:
predict the endpoint as a residual delta instead of absolute latent
coordinates. The previous lane proved that endpoint I2SB can raise transfer
style above `0.70`, but it paid too much LPIPS and reversed after e2.

## Controlled Change

- Matched control:
  `configs/aaai2027/phase2_i2sb_clean_k070_e3_sigma0p02_b8a2_vlen010.json`.
- Candidate:
  `configs/aaai2027/phase2_i2sb_residual_k070_e3_sigma0p02_b8a2_vlen010.json`.
- Changed:
  `model.endpoint_parameterization=absolute -> residual`.
- Unchanged:
  parent checkpoint, `solver_family=solver_i2sb`,
  `transport_prediction_mode=endpoint`, `bridge_sigma=0.02`,
  `bridge_noise_schedule=exact_brownian`, endpoint time floors, tokenizer,
  TopoGate, semantic cross-attention, terminal SWD, training schedule, and
  fast10 transfer eval.

## Rationale

The absolute endpoint lane asks the network to emit full target latent
coordinates, then I2SB turns that into a transport velocity. Residual endpoint
keeps the same mathematical bridge but makes the learned section local:
`z_base = x_t + delta_theta(x_t, s)`. This should reduce coordinate drift and
LPIPS while retaining the style actuation seen in the absolute endpoint lane.

## Eval Contract

- Training-time eval subdir: `full_eval_fast10`.
- Transfer-only, `10` source samples per style.
- Training-time eval uses short-lived subprocess isolation:
  `full_eval_in_process=false`, `full_eval_runtime_model_cache=false`.
  Runtime model cache is reserved for offline all-ckpt sweeps via
  `run_evaluation.py <ckpt_dir> --batch_in_process --runtime_model_cache`,
  so CLIP/LPIPS/ORT reuse cannot pollute the training process.
- ONNX VAE decoder: `../eval_cache/vae_onnx/ema_b16_32/decoder.onnx`.
- Save and eval every retained checkpoint with `CLIP-S + LPIPS`.
- Append transfer points to the AAAI page-1 plot CSV at closure.

## Runtime Notes

- Initial in-process e1 eval exposed a dtype mismatch in output appearance
  alignment under bf16 inference. Fixed by casting the adjusted appearance
  tensor back to `pred.dtype` before `lerp`.
- In-process runtime cache was beneficial for standalone repeated eval but
  unsafe inside training: after e2 it retained eval models in the training
  process, pushed VRAM to roughly `12.1 GB`, and made the eval wall time worse.
  Formal training configs now disable it; use the wrapper batch mode for
  non-blocking all-ckpt eval instead.
- Batch wrapper cache bug fixed on 2026-06-16: the first implementation used
  `runpy(..., run_name="__main__")`, which recreated the evaluator module for
  every checkpoint and silently defeated the runtime cache. The wrapper now
  calls `utils.run_evaluation.main(argv)` directly. Remote fast10 probe on the
  two residual checkpoints dropped from `305.95s` to `63.42s`; e2 reused ONNX
  decoder, LPIPS, and CLIP runtime objects, and GPU memory returned to
  `128 MB` after process exit.

## Current Fast10 Transfer Curve

| ckpt | CLIP-S | LPIPS | read |
| --- | ---: | ---: | --- |
| e1 | 0.676019 | 0.312541 | structure recovered, style below target |
| e2 | 0.673869 | 0.308784 | LPIPS improves slightly, style retreats |

## Interim Decision

Residual endpoint solves the absolute endpoint lane's LPIPS damage, but it
removes the style actuation that made I2SB interesting. Unless a later control
needs it as a structure anchor, this is a structure-only negative result rather
than a promotion candidate. Next clean direction should preserve absolute
endpoint actuation while reducing drift, for example a lower-sigma absolute
I2SB scan or an explicit bounded blend as a separate switch.

## Decision Rule

- Promote only if transfer style stays near or above the absolute endpoint e2
  peak (`0.709094`) while LPIPS moves materially down toward the accepted
  `~0.35` band.
- Continue if style is improving and LPIPS is falling, even if the first
  checkpoint remains above `0.38`.
- Stop early if style falls below `0.700` for two consecutive retained
  checkpoints while LPIPS remains above `0.38`.
- If residual endpoint improves LPIPS but collapses style back to the
  k070/predec frontier, archive as a structure-only fix and move to a stronger
  actuation route.

## Expected Read

- Positive: e1/e2 style `>=0.705` with LPIPS clearly below absolute e2
  (`0.490233`) and trending toward `0.35`.
- Negative: style regresses to `~0.68` or LPIPS remains `>0.42` after the first
  few fast10 checkpoints.
