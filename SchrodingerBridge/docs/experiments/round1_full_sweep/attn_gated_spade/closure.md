# attn_gated_spade Closure

- Status: `recalibration_needed`
- Closure read:
  - `batch_size = 13` was correctly rejected at health check as `under_band`
  - `batch_size = 19` reached retained checkpoints through `epoch_0022`
  - fast `CLIP-S / LPIPS` coverage exists for every retained checkpoint through `epoch_0022`
  - the lane then lost its active train pid during `epoch 23`, and no `epoch_0023.pt` landed
  - the remaining remote `fast_eval` watcher was manually stopped after confirming no train pid remained
- Why this is not a formal closure:
  - process-local memory evidence never satisfied the requested `9.0-10.8 GiB` band
  - host-side runtime samples also degraded to `nonformal_under_band`
  - the run stopped before the planned `24` epochs and before formal convergence closure
- What is still useful:
  - all settled `epoch_0001 -> epoch_0022` fast-eval points
  - the family-specific curve shape for style vs LPIPS tradeoff
  - a concrete calibration signal that `batch=19` is still too light for this cost class
- Reopen condition:
  - relaunch only after a new batch calibration is chosen explicitly for this family
