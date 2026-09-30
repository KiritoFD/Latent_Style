# Windows/WSL Infra Stability Notes

## Why epoch time barely changed across batch sizes

Two settings were masking real throughput differences:

1. `training.cuda_sync_debug=true`
- This inserts `torch.cuda.synchronize()` after move/loss/backward/step in every iteration.
- It serializes the pipeline and removes overlap, so larger batch does not show expected scaling.

2. DataLoader configured as single-threaded (`num_workers=0`) with `pin_memory=false`
- CPU input pipeline becomes bottleneck.
- GPU waits for host-side batch preparation/transfers.

## Stability-first defaults applied

- `num_workers=-1` (auto)
  - Windows auto policy uses conservative worker count.
- `pin_memory=true`
- `persistent_workers=true` (when workers > 0)
- `prefetch_factor=2`
- `cuda_sync_debug=false` (keep only for debugging)
- `strict_batch_sanity=true` with `strict_batch_sanity_interval` to reduce overhead
- `preload_to_gpu=false` (avoid allocator fragmentation/OOM risk in long runs)

## Code-level hardening

- Batch sanity checks now run on CPU (`raw_batch`) before device transfer.
- Loss/backward path no longer pays extra device sync from sanity checks.
- `run.py` warns when `cuda_sync_debug=true`.
- `watchdog_windows.ps1` now passes explicit `CONFIG_PATH` to avoid config drift.

## Recommended debug workflow

Only enable these for short repro runs:

- `training.cuda_sync_debug=true`
- `CUDA_LAUNCH_BLOCKING=1`

After root cause is identified, set both back to normal for throughput runs.
