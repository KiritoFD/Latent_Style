# Infra Cleanup: Training and Inference Guardrails

Date: 2026-06-15

## Scope

This cleanup addresses the infra risks listed in `docs/612-phase2/fiber.md` before launching the next Fiber Bundle experiment. The goal is to keep future mechanism deltas interpretable: no hidden CPU OT bottleneck, no unsafe compile/layout combo, no accidental postprocess metric inflation, and no solver corrector silently rewriting content/style.

## Changes

### 3.1 OT Coupling CPU Offload

- `BridgeConfig.coupling_solver` is confirmed default `sinkhorn`.
- `bridge.coupling_solver="hungarian"` now raises unless `bridge.allow_cpu_hungarian=true`.
- `scipy.optimize.linear_sum_assignment` is now lazily imported only inside the explicitly allowed Hungarian path.
- Decision: remote formal training should use GPU Sinkhorn. Hungarian is diagnostic-only.

### 3.2 SWD Projection Cache Shape Safety

- SWD projection cache keys now include spatial size `(H, W)`.
- This prevents projection bank reuse across latent resolutions or future multi-resolution schedules.

### 3.3 `torch.compile` x `channels_last`

- `training.channels_last=true` and `training.torch_compile=true` are now mutually exclusive.
- The trainer raises before model construction if both are requested.
- Decision: use one memory/perf path at a time; do not silently run compiled kernels over mixed stride assumptions.

### 3.4 Gradient Checkpointing x no-grad skip

- The high-resolution content body is no longer executed twice.
- `content_feat_16` keeps the gradient path; `skip_32` reuses the same result via `.detach()`.
- This preserves the old no-grad skip semantics while removing duplicated compute.

### 4.1 Style Overdrive

- Added `model.allow_style_overdrive=false` default.
- If disabled, integration horizon is clamped to `<=1.0` even when `style_strength_max` or requested strength is larger.
- Runtime debug records whether overdrive was clamped.
- Decision: overdrive is now an explicit extrapolation experiment, not a default inference trick.

### 4.2 RGB/Latent Affine Calibration

- RGB/latent style affine postprocess remains available but metric-affecting use now requires `--allow_metric_postprocess` or `full_eval.allow_metric_postprocess=true` / `training.full_eval_allow_metric_postprocess=true`.
- Summary JSON records `allow_metric_postprocess`.
- Decision: affine calibration is diagnostic-only unless explicitly labeled.

### 4.3 PC Lowpass Corrector

- `model.solver_corrector_mode` default is now `none`.
- The lowpass corrector must be explicitly enabled, e.g. `solver_corrector_mode="latent_lowpass"`.
- Decision: PC lowpass is structure repair, not a default style solver.

### 4.4 Endpoint Velocity Division

- Added `model.endpoint_velocity_time_floor=0.05` default.
- Endpoint velocity conversion in `forward()` now clamps `(1-t)` by this floor instead of `1e-3`.
- Decision: endpoint parameterization can no longer amplify late-time velocity by 1000x by default.

## Verification

- AST parse passed for:
  - `src/config_schema.py`
  - `src/losses.py`
  - `src/trainer.py`
  - `src/lancet_runtime.py`
  - `src/model.py`
  - `src/run.py`
  - `src/utils/run_evaluation.py`
  - `src/ot_cost.py`
- `git diff --check` passed.
- Guard smoke:
  - Sinkhorn objective initializes.
  - Unauthorized Hungarian raises.
  - Safe defaults load: `allow_style_overdrive=false`, `solver_corrector_mode=none`, `endpoint_velocity_time_floor=0.05`.
  - Active Phase2 k070 config loads under the new schema.

## Experiment Rule After Cleanup

The next remote experiment should start only after this cleanup is committed. Any config using:

- `coupling_solver="hungarian"`
- `channels_last=true` with `torch_compile=true`
- `style_strength_max>1` without `allow_style_overdrive=true`
- `solver_corrector_mode!="none"`
- metric-affecting RGB/latent affine postprocess

must be labeled as an explicit diagnostic run, not a clean mechanism result.
