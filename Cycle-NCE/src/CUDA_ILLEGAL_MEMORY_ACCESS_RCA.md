# CUDA Illegal Memory Access RCA

## Symptom

Training occasionally crashes with:

`torch.AcceleratorError: CUDA error: an illegal memory access was encountered`

Stack trace often points to a harmless line (for example `torch.tensor(0.0, device=...)` in `losses.py`) because CUDA errors are asynchronous.

## What is actually happening

Primary trigger is in PatchNCE:

- `losses.py` computes a large similarity matrix per step: shape `[B, P, P]` where `P = H*W = 32*32 = 1024`.
- With large `B` (for example 160/256) and AMP FP16, this batched GEMM is both memory-heavy and kernel-fragile on some driver/runtime combinations.
- When that kernel fails asynchronously, the next CUDA API call reports `illegal memory access`, so traceback location is misleading.

Secondary signal:

- `__comment_gates__` in `config.json` is only a config comment key and produced warning noise, but it is **not** the crash root cause.

## Code changes made

1. Stabilized PatchNCE path in `losses.py`
- Switched NCE matrix math to FP32.
- Added batch-chunked computation (`batch_chunk_size=16` default) to reduce peak GPU pressure.
- Result: smaller peak allocation and fewer sporadic kernel faults.

2. Reduced warning noise in `model.py`
- Ignored model config keys starting with `__comment` when checking unknown keys.

3. Improved crash diagnostics in `trainer.py`
- Added explicit log when catching `illegal memory access` so epoch/step are visible and debugging action is clear.

## Operational recommendations

1. If crash reappears, run once with:
- `CUDA_LAUNCH_BLOCKING=1 uv run run.py --config config.json`

2. Keep batch size away from VRAM edge:
- If unstable at `batch=160`, try `128` or `96`.

3. After any `illegal memory access`, restart the training process:
- CUDA context may be corrupted after such errors.

4. Keep `data.preload_to_gpu=true` only when VRAM headroom is enough:
- Preload itself is valid, but it reduces free headroom for large GEMM workloads.

## Notes on the warning you asked about

`UserWarning: Unknown model config key(s): __comment_gates__` was harmless warning noise from config metadata and has now been filtered out.
