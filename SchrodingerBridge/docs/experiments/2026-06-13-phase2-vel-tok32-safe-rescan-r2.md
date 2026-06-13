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
