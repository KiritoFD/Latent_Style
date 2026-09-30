# DynamicCast `fetch_and_cast` Assertion Investigation

Date: 2026-02-15
Project: `Cycle-NCE`

## Symptom
Training intermittently crashes with CUDA device-side assert messages like:

- `/pytorch/c10/core/DynamicCast.h:78: fetch_and_cast ... Assertion 'false' failed`
- Followed by Python-side error at a later op (often tensor creation or backward), e.g. `cudaErrorAssert`/`cudaErrorUnknown`.

## Key Finding: First failing Python line is usually not root cause
CUDA kernels are async. Once a device-side assert happens, CUDA context becomes poisoned for this process. Later CUDA calls fail and can misleadingly point to unrelated lines.

Implication:
- This is consistent with "async dirty context" behavior.
- After first assert, process restart is required (cannot reliably recover in-process).

## New Evidence from repository logs (2026-02-15)
From `Cycle-NCE/train_logs/train_watchdog_20260215_090400.log`:
- `DataLoader | batch=160 ... preload_to_gpu=True`
- `Infra | channels_last=True ... compile=True`
- `Precision | amp=True amp_dtype=bf16`
- auto-resume from `../no-edge/epoch_0300.pt`

This runtime profile does **not** match the intended stability profile (small batch, no AMP, no channels_last, no compile).
It indicates a **config/entry drift** problem in addition to async CUDA error behavior.

Practical root cause chain:
1. Training launched with high-risk infra settings (large batch + preload_to_gpu + compile + channels_last).
2. A CUDA kernel asserts (`DynamicCast fetch_and_cast`) under pressure / unsupported cast path.
3. CUDA context becomes dirty; subsequent lines report misleading failures.

## Codebase Risk Audit (stability-first)
Searched high-risk paths:
- mixed precision + scaler path in `trainer.py`
- channels_last forced path in `trainer.py`
- index/gather style-id paths in `model.py`
- semigroup multi-forward branch in `losses.py`

### Confirmed instability amplifiers found earlier
1. `channels_last` was previously forced ON regardless config.
2. AMP default path could still activate scaler under certain configs.
3. Large batch (`160`) with semigroup branch increases kernel/VRAM pressure and error probability.
4. Loss stage traceback often points to post-failure line due async reporting.

## Hardening Changes Applied
1. Respect config for memory format (no forced channels_last)
   - File: `Cycle-NCE/src/trainer.py`
2. AMP default set to OFF (config-controlled)
   - File: `Cycle-NCE/src/trainer.py`
3. Added explicit `training.use_grad_scaler` gate
   - File: `Cycle-NCE/src/trainer.py`
4. Added strict batch sanity checks (finite tensors + style_id range)
   - File: `Cycle-NCE/src/trainer.py`
5. Added optional CUDA sync debug checkpoints (`training.cuda_sync_debug`)
   - File: `Cycle-NCE/src/trainer.py`
6. Added loss-stage tagging in exception to identify failing stage
   - File: `Cycle-NCE/src/losses.py`
7. Reduced main training batch to stable baseline
   - File: `Cycle-NCE/src/config.json` (`batch_size: 64`)
8. Watchdog launch hardened to prevent config drift:
   - absolute `CONFIG_PATH`
   - config hash and effective snippet logging
   - File: `Cycle-NCE/src/watchdog.sh`

## Current stability baseline config
Recommended for root-cause localization:
- `training.use_amp = false`
- `training.use_grad_scaler = false`
- `training.channels_last = false`
- `training.strict_batch_sanity = true`
- `training.cuda_sync_debug = true` (diagnosis mode only)
- moderate batch (`64` or lower)

## Repro / Debug procedure
1. Start fresh process (important after any assert).
2. Run with blocking + sync debug:

```bash
CUDA_LAUNCH_BLOCKING=1 python run.py --config /mnt/i/Github/Latent_Style/Cycle-NCE/src/config.json
```

3. If crash happens, use new message:
- `AdaCUTObjective.compute failed at stage='...'`
- plus trainer stage sync boundary logs to locate earliest failing region.

4. Verify runtime actually uses expected config:
- in watchdog log, confirm `config=...` and `config_sha256=...`
- check printed effective keys (`batch_size`, `use_amp`, `channels_last`, `use_compile`, etc.)

## Root-cause hypothesis (most likely)
Primary cause is kernel instability under aggressive runtime settings (large batch + heavy semigroup workload + previously forced channels_last / mixed-precision path), with async error reporting obscuring origin.

Not enough evidence yet for a single deterministic logic bug (e.g., fixed out-of-range index) after current guards, but new instrumentation is in place to capture a deterministic failing stage next run.

## Next step if still failing
- Temporarily set `loss.w_semigroup = 0.0` for isolation run.
- If stable, semigroup branch is the hot path to optimize/split.
- If still unstable, capture first failing stage from new tagged exception and profile that stage only.
