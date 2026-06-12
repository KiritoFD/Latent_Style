# Phase 2: vel_pattn_enhanced_tok

Date: 2026-06-13

## Goal

- follow the `612-phase2` structure-first pivot
- this is the only formal Distinct5 remote training lane after the endpoint / I2SB fail-stop
- keep the pure-latent tokenizer path
- return to `velocity` as the transport target
- combine:
  - enhanced `PureLatentSpatialTokenizer`
  - `manifold_adaptive_split`
  - `crossattn_texture`
- target board:
  - style `>= 0.72`
  - LPIPS `<= 0.30`

## Why This Packet

- `rtfix epoch_0001` proved the corrected true-I2SB runtime can raise style, but it failed the new structure gate:
  - transfer `0.724444 / 0.712723`
  - all-pairs `0.724472 / 0.707551`
- `612-lookback` says the main bottleneck is still:
  - endpoint predicts `x_1` and behaves like repainting
  - velocity predicts delta and is the cleaner edit surface
- this packet therefore keeps:
  - pure latent tokenizer
  - proximal texture refinement
  - manifold-aware motion regularization
- but drops:
  - endpoint mainline training
  - SDE / I2SB as the active Distinct5 training lane

## Active Configs

- calibration configs:
  - [phase2_vel_pattn_enhanced_tok_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_pattn_enhanced_tok_seed42_b8a2.json)
- formal in-band config:
  - [phase2_vel_pattn_enhanced_tok_seed42_b22a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_pattn_enhanced_tok_seed42_b22a1.json)
- key settings:
  - `tokenizer_family = pure_latent_spatial`
  - `tokenizer_num_clusters = 32`
  - `transport_prediction_mode = velocity`
  - `solver_family = euler_legacy`
  - `proximal_mode = crossattn_texture`
  - `kinetic_penalty_mode = manifold_adaptive_split`
  - `w_kinetic = 1.0`
  - `structure_penalty_mode = off`

## Smoke

- local synthetic smoke:
  - status `ok`
  - tokenizer family `pure_latent_spatial`
  - transport mode `velocity`
  - solver `euler_legacy`
  - no DINO runtime required
  - corrected objective:
    - `objective_mode = bridge_velocity`
    - `t_mean = 0.6106`
    - no longer the old fixed-`t=1` OMF path
- important compatibility check:
  - pure-latent tokenizer now supports `crossattn_texture` proximal refinement without falling back to legacy `style_spatial_id_16`

## Promotion Contract

- paper-facing success target:
  - style `>= 0.72`
  - `content_lpips <= 0.30`
- continue-to-train gate:
  - settled checkpoints must remain in `content_lpips < 0.40`
- archival gate:
  - `0.40 <= content_lpips < 0.70`
- fail-stop gate:
  - `content_lpips >= 0.70`
- rule:
  - if the first settled point already lands in archival or fail-stop territory, the lane does not keep the formal 3060 slot

## Run Log

- remote status:
  - initial `b8/a2` calibration launched
  - 20s health check:
    - `5917 MiB`
    - too far under-band
  - after another 150s:
    - still `5917 MiB`
    - no late warmup into the formal band
  - training itself did run and stayed numerically stable, but the packet is a calibration miss
  - relaunch decision:
    - move to `batch=16`
    - move to `accumulation_steps=1`
    - keep effective batch at `16`
    - target the formal `9.x-10.x GiB` band directly
  - corrected `b16/a1` relaunch:
    - first attempt exposed a theory mismatch:
      - `objective_mode = omf` was still forcing the old fixed-`t=1` endpoint-style path
      - this contradicted the Phase 2 velocity plan
    - config has now been corrected to:
      - `objective_mode = bridge_velocity`
    - valid relaunch result:
      - 20s health check `7497 MiB`
      - after another 150s only `7947 MiB`
      - the theory was fixed, but VRAM was still under-band
  - next calibration:
    - raise batch to `22`
    - keep accumulation at `1`
    - target the formal `9.x-10.x GiB` band directly
  - formal `b22/a1` launch:
    - config:
      - [phase2_vel_pattn_enhanced_tok_seed42_b22a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_pattn_enhanced_tok_seed42_b22a1.json)
    - 20s health check:
      - `9423 MiB`
    - current status:
      - accepted as formal in-band lane
      - remote training is live
      - first settled checkpoint will decide whether this packet remains the formal lane or gets archived immediately

## Settled Curve

- live remote owner:
  - PID `45964`
  - command:
    - `/home/xy/venvs/samam312/bin/python SchrodingerBridge/src/run.py --config /mnt/i/Github/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_pattn_enhanced_tok_seed42_b22a1.json`
- current phase:
  - after `epoch_0005` full eval, training resumed into `Epoch 6/24`
- settled checkpoints so far:
  - `epoch_0001` at `2026-06-13 02:13:00`
    - transfer `0.670199 / 0.368480`
    - all-pairs `0.698346 / 0.367345`
    - eval wall `207.72s`
    - generation `111.69s`
    - VAE decode `54.50s`
  - `epoch_0002` at `2026-06-13 02:34:26`
    - transfer `0.673934 / 0.384340`
    - all-pairs `0.701666 / 0.381724`
    - eval wall `209.12s`
    - generation `113.48s`
    - VAE decode `54.52s`
  - `epoch_0003` at `2026-06-13 02:55:59`
    - transfer `0.670266 / 0.381765`
    - all-pairs `0.698319 / 0.379036`
    - eval wall `214.00s`
    - generation `115.27s`
    - VAE decode `54.52s`
  - `epoch_0004` at `2026-06-13 03:17:29`
    - transfer `0.670364 / 0.373813`
    - all-pairs `0.699071 / 0.370858`
    - eval wall `211.87s`
    - generation `113.73s`
    - VAE decode `54.52s`
  - `epoch_0005` at `2026-06-13 03:38:55`
    - transfer `0.671283 / 0.377020`
    - all-pairs `0.699481 / 0.375100`
    - eval wall `211.00s`
    - generation `115.36s`
    - VAE decode `54.54s`
- short-horizon trend:
  - `epoch_0001 -> epoch_0002`
    - transfer style `+0.003735`
    - transfer LPIPS `+0.015860`
    - all-pairs style `+0.003320`
    - all-pairs LPIPS `+0.014380`
  - `epoch_0002 -> epoch_0003`
    - transfer style `-0.003668`
    - transfer LPIPS `-0.002575`
    - all-pairs style `-0.003347`
    - all-pairs LPIPS `-0.002688`
  - `epoch_0003 -> epoch_0004`
    - transfer style `+0.000098`
    - transfer LPIPS `-0.007952`
    - all-pairs style `+0.000752`
    - all-pairs LPIPS `-0.008179`
  - `epoch_0004 -> epoch_0005`
    - transfer style `+0.000918`
    - transfer LPIPS `+0.003206`
    - all-pairs style `+0.000410`
    - all-pairs LPIPS `+0.004242`

## Read

- this line is still eligible:
  - all five settled points remain inside the Phase 2 continuation band `LPIPS < 0.40`
- but it is not yet promotable:
  - best current point is still only `all-pairs 0.701666 / 0.381724`
  - the paper target `0.72 / 0.30` remains far away
- current shape:
  - `epoch_0002` was the local style peak so far
  - `epoch_0003` gave back that style gain while recovering a small amount of LPIPS
  - `epoch_0004` improves LPIPS again while recovering only a negligible amount of style
  - `epoch_0005` drifts back upward on LPIPS while style still fails to reclaim the `epoch_0002` peak
  - this is a bounded `0.699 +/-` / `0.37x-0.38x` oscillation, not a breakout
  - the line is not yet showing a clean path toward `0.72 / 0.30`
- convergence authority:
  - `curve_summary.json` currently has `row_count = 5`
  - `pareto_epochs = [epoch_0001, epoch_0002, epoch_0003, epoch_0004, epoch_0005]`
  - `best_in_newest_2 = false`
  - `tail_flat = true`
  - `converged = false`
- live runtime read after `epoch_0005` settle:
  - remote training resumed into `Epoch 6/24`
  - `latest_checkpoint_epoch = epoch_0005`
  - `latest_settled_epoch = epoch_0005`
  - `pending_checkpoint_epochs = []`
- current decision:
  - keep the formal remote lane alive
  - do not promote any checkpoint yet
  - `epoch_0004` remains the strongest LPIPS-side compromise point while `epoch_0002` remains the style peak
  - `epoch_0005` does not change the frontier story materially enough to justify a longer open-ended continuation
  - if `epoch_0006` does not produce a clear style-side recovery or a materially better joint point, this lane should be closed and the queue should hand off to `eval_only_pc_solver`
