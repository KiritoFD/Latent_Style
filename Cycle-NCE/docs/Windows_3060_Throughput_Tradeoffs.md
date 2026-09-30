# Windows + RTX 3060 (Torch 2.10 / CU128) Infra Trade-offs

## Goals

- Keep long-run training stable.
- Maximize effective throughput on modest SM compute.
- Use large VRAM and memory bandwidth to hide overhead.

## Implemented infra optimizations

1. **Model step-context reuse**
- `model.integrate()` now computes style context once and reuses it across all integration steps.
- This removes repeated style encode/spatial map prep in multi-step forward paths.

2. **Safe GPU dataset preload**
- `dataset.py` now supports `data.preload_to_gpu=true` with guardrails:
  - `data.preload_max_vram_gb`
  - `data.preload_reserve_ratio`
- If budget check fails or preload OOMs, it auto-falls back to CPU tensors.

3. **High-throughput optimizer/compiler path restored**
- `trainer.py` now respects:
  - `training.use_compile`
  - `training.fused_adamw`
  - `training.use_gradient_checkpointing`
- Compile still has safe fallback to eager if runtime compile fails.

4. **Semigroup compute budget controls**
- New loss config:
  - `loss.semigroup_max_samples`
  - `loss.semigroup_every_n_steps`
- Lets you scale global batch up without semigroup cost exploding linearly.

## Recommended profiles

### Profile A: Stable-Throughput (default recommendation)
- `use_amp=true`, `amp_dtype=bf16`, `use_grad_scaler=false`
- `channels_last=true`
- `use_compile=true` (allow fallback)
- `fused_adamw=true`
- `preload_to_gpu=true` with budget limits
- `semigroup_every_n_steps=1`, `semigroup_max_samples=32`

### Profile B: Max Throughput (accept more risk)
- Same as A, plus:
  - larger `batch_size`
  - optionally `semigroup_every_n_steps=2`
- This usually gives better `it/s`, with some objective signal thinning.

### Profile C: Maximum Stability
- `use_compile=false`
- `preload_to_gpu=false`
- keep AMP bf16 on, but avoid aggressive knobs.

## Practical tuning order

1. Fix a baseline with Profile A and monitor `data_time_sec` vs `compute_time_sec`.
2. Increase `batch_size` until first instability/throughput plateau.
3. If compute dominates, set `semigroup_every_n_steps=2`.
4. If dataloader dominates, keep preload enabled and reduce worker overhead.
