# Phase 2: vel_tok32_safe_rescan_r2

Date: 2026-06-13

## Role

- second safe-family follow-up after `safe_rescan_r1` produced a near-miss
- stays inside the same `velocity + pure_latent_spatial + crossattn_texture + manifold_adaptive_split` family
- tries to keep the style lift of `r1 epoch_0002` while pulling LPIPS back under the formal `< 0.40` gate

## Why This Exists

- `safe_rescan_r1` answered the main scientific question:
  - the safe-family tokenizer sweep can lift style above the old shelf
  - but it also proved the current `r1` setting overshoots the LPIPS gate
- specifically, `r1 epoch_0002` reached:
  - transfer `0.676378 / 0.400694`
  - all-pairs `0.702543 / 0.397891`
- interpretation:
  - style lift is real
  - but the packet is still archival-stop because the worst authority LPIPS crossed `0.40`
- that makes `r2` justified:
  - not a blind extra sweep
  - a narrow rollback from a very close near-miss

## Config

- config:
  - [phase2_vel_tok32_safe_rescan_r2_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_rescan_r2_seed42_b20a1.json)
- parent packet:
  - [phase2_vel_tok32_safe_rescan_r1_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_rescan_r1_seed42_b20a1.json)
- warm-start checkpoint:
  - `/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_tok32_safe_rescan_r1_seed42_b20a1/epoch_0001.pt`

## Deltas

- relative to `safe_rescan_r1`:
  - `tokenizer_structured_temperature: 0.07 -> 0.075`
  - `tokenizer_global_gate_scale: 1.2 -> 1.15`
  - `w_kinetic: 0.90 -> 0.95`
- rationale:
  - reduce style push slightly
  - restore a bit more structural restraint
  - preserve the same family and same implementation story

## Promotion Contract

- formal continuation gate:
  - all settled points must stay in `content_lpips < 0.40`
- shelf-break target:
  - exceed `all-pairs 0.701666 / 0.381724`
- short-screen target:
  - if the packet does not produce a non-dominated in-band point by `epoch_0003`, close early
- archival gate:
  - any settled point with worst authority LPIPS `>= 0.40` is immediate closure
- complete-failure gate:
  - any settled point with LPIPS `>= 0.70` is not only an immediate stop for this packet, but direct evidence that this family is off the Distinct5 promotion path until redesigned from a safe parent

## Smoke

- local synthetic smoke:
  - [phase2_vel_tok32_safe_rescan_r2_seed42_b20a1_smoke.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/phase2_vel_tok32_safe_rescan_r2_seed42_b20a1_smoke.json)
  - status `ok`
  - `objective_mode = bridge_velocity`
  - `tokenizer_family = pure_latent_spatial`
  - `solver_family = euler_legacy`
  - `transport_prediction_mode = velocity`
  - no DINO runtime required
  - tensor shapes:
    - forward `[1, 4, 32, 32]`
    - endpoint `[1, 4, 32, 32]`
    - integrated `[1, 4, 32, 32]`
  - loss read:
    - `loss = 2.300523`
    - `flow = 2.067606`
    - `terminal_swd = 0.009935`
    - `t_mean = 0.456406`
  - first grad:
    - `structured_style_tokenizer.universal_keys`
    - abs mean `0.005138`

## Run Log

- remote formal launch:
  - attempted after `safe_rescan_r1` closed
  - host-owned launcher path failed on the remote Windows side with:
    - `HCS_E_SERVICE_NOT_AVAILABLE`
  - diagnosis:
    - `Microsoft-Windows-Subsystem-Linux` was enabled
    - `VirtualMachinePlatform` was disabled and has now been enabled
    - `hypervisorlaunchtype` is now explicitly set to `Auto`
  - current blocker:
    - the remote Windows host still needs a reboot before WSL2 can actually start again
  - current status:
    - packet is prepared
    - smoke is done
    - no active remote process owns this packet yet
- health recheck at `2026-06-13 12:40:35 +08:00`:
  - `ssh_ok = true`
  - `wsl_exec_ok = false`
  - `remote_wsl_hcs_failure = true`
  - `reboot_required_for_wsl2 = true`
  - remote GPU idle-ish:
    - `552-556 MiB / 12288 MiB`
  - `live_state = remote_wsl_unavailable`
  - watcher status:
    - local recovery watcher PID `313620` is still alive
    - log path:
      - [phase2_vel_tok32_safe_rescan_r2_seed42_b20a1_recover_watcher.out.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/phase2_vel_tok32_safe_rescan_r2_seed42_b20a1_recover_watcher.out.log)
- watcher refresh at `2026-06-13 13:55:29 +08:00`:
  - the older explicit-args recovery watcher was replaced locally
  - current watcher now resolves the formal lane from the validated phase2 manifest
  - new local watcher PID:
    - `322404`
  - new logs:
    - [phase2_formal_lane_recover_from_manifest.out.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/phase2_formal_lane_recover_from_manifest.out.log)
    - [phase2_formal_lane_recover_from_manifest.err.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/phase2_formal_lane_recover_from_manifest.err.log)
  - first resolved packet read:
    - `packet_id = vel_tok32_safe_rescan_r2`
    - `watch_min_settled_epoch = 3`
    - `watch_min_allpairs_style_recovery = 0.701666`
    - `watch_max_allpairs_lpips_for_recovery = 0.381724`
    - `watch_min_transfer_style_recovery = 0.673934`
    - `watch_max_transfer_lpips_for_recovery = 0.384340`
    - `watch_handoff_mode = stop_only`
- reboot + relaunch at `2026-06-13 14:06:00 +08:00`:
  - remote Windows host reboot command was issued locally
  - post-reboot health now reads:
    - `wsl_exec_ok = true`
    - `remote_wsl_hcs_failure = false`
    - `reboot_required_for_wsl2 = false`
  - the older recovery watcher died during reboot and was replaced again
  - current manifest-driven watcher PID:
    - `138004`
  - formal lane relaunch:
    - remote launch start:
      - `2026-06-13T14:16:08+08:00`
    - remote process:
      - `458 /home/xy/venvs/samam312/bin/python SchrodingerBridge/src/run.py --config /mnt/i/Github/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_rescan_r2_seed42_b20a1.json`
    - current live state:
      - `training_before_first_settled_eval`
  - first post-recovery health read:
    - remote GPU `10380 MiB / 12288 MiB`
  - later live read:
    - remote GPU `9910 MiB / 12288 MiB`
  - interpretation:
    - the lane is back in the preferred formal band
    - training has resumed before the first settled checkpoint
- authority-progress read at `2026-06-13 14:40 +08:00`:
  - latest checkpoint:
    - `epoch_0001`
  - current live state:
    - `eval_in_progress_or_pending`
  - pending checkpoint epochs:
    - `epoch_0001`
  - current GPU read during eval/offload window:
    - `2066 MiB / 12288 MiB`
  - interpretation:
    - the first retained checkpoint now exists
    - the packet has entered the first checkpoint-level authority window
    - final keep/stop judgment still waits on the first settled summary
- first settled authority point at `2026-06-13 14:39:53 +08:00`:
  - `epoch_0001`
  - transfer `0.672065 / 0.379086`
  - all-pairs `0.700117 / 0.377982`
  - identity `0.812324 / 0.373565`
  - eval wall `218.42s`
  - generation `117.35s`
  - VAE decode `56.62s`
  - convergence read:
    - `row_count = 1`
    - `best_in_newest_2 = true`
    - `since_last_pareto = 0`
    - `converged = false`
  - interpretation:
    - the lane is still in-band because both transfer and all-pairs LPIPS remain `< 0.40`
    - but it does not yet beat the old safe shelf `0.701666 / 0.381724`
    - so this is a keep-running point, not a promotion point
    - the current short-screen remains:
      - do not close before `epoch_0003`
      - but expect closure if the next authority points still fail to form a new in-band non-dominated point
- second settled authority point at `2026-06-13 15:03:52 +08:00`:
  - `epoch_0002`
  - transfer `0.675645 / 0.395898`
  - all-pairs `0.702225 / 0.393204`
  - identity `0.808543 / 0.382426`
  - eval wall `217.94s`
  - generation `116.71s`
  - VAE decode `56.62s`
  - convergence read:
    - `row_count = 2`
    - `best_in_newest_2 = true`
    - `since_last_pareto = 0`
    - `tail_flat = true`
    - `converged = false`
  - interpretation:
    - the lane is still formally in-band because both transfer and all-pairs LPIPS remain `< 0.40`
    - this is a real in-family Pareto point over `epoch_0001`
    - style now exceeds the old shelf, but LPIPS also rose above the old shelf recovery ceilings:
      - all-pairs safe-shelf gate was `0.701666 / 0.381724`
      - transfer safe-shelf gate was `0.673934 / 0.384340`
    - so the packet is still not a promotable shelf-break
    - the short-screen therefore stays alive through `epoch_0003`
    - the next settled point now carries the main decision weight:
      - another LPIPS rise into `0.40+` means archival stop
      - a style plateau without LPIPS recovery likely means safe-family exhaustion
- third settled authority point at `2026-06-13 15:27:49 +08:00`:
  - `epoch_0003`
  - transfer `0.675325 / 0.398119`
  - all-pairs `0.701712 / 0.395315`
  - identity `0.807258 / 0.384102`
  - eval wall `218.02s`
  - generation `116.76s`
  - VAE decode `56.47s`
  - convergence read:
    - `row_count = 3`
    - `best_epoch = epoch_0002`
    - `best_in_newest_2 = true`
    - `since_last_pareto = 1`
    - `tail_flat = false`
    - `converged = false`
  - interpretation:
    - the packet did survive the short-screen because it already created a real in-family Pareto point at `epoch_0002`
    - `epoch_0003` regressed slightly on both style and LPIPS relative to `epoch_0002`
    - best settled point therefore remains:
      - `epoch_0002`
      - transfer `0.675645 / 0.395898`
      - all-pairs `0.702225 / 0.393204`
    - the line still fails the promotable safe-shelf break because LPIPS remains above:
      - all-pairs `0.381724`
      - transfer `0.384340`
    - however, the lane remains formally alive because:
      - all settled points are still `< 0.40`
      - the best point remains within the newest-2 window
    - next decision rule from here:
      - first `0.40+` settled point => immediate archival stop
      - if later retained checkpoints fail to improve beyond `epoch_0002`, safe-family sweep should be declared exhausted

## Ops Note

- this is not a model-side rejection
- it is a remote host / WSL2 availability issue
- once the remote Windows host is rebooted and WSL2 becomes runnable again, `r2` should be the next formal lane to relaunch
- preferred local recovery hook:
  - [watch_phase2_wsl_recover_and_launch.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_phase2_wsl_recover_and_launch.py)
  - this watcher can sit locally, wait for `wsl_exec_ok=true`, then auto-launch `r2` and hand off to the phase-2 close-rule watcher

## Intended Read

- success:
  - keep the style lift from `r1`
  - bring worst authority LPIPS back below `0.40`
  - ideally exceed the old shelf in a non-dominated way
- failure:
  - either lose too much style and collapse back under the shelf
  - or keep crossing into archival-stop
- next action if this fails:
  - safe-family sweep is exhausted
  - next candidate returns to queued training-side structure control, not another tokenizer-only retry
