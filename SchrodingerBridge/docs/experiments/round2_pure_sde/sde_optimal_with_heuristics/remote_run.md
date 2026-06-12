# sde_optimal_with_heuristics Remote Run Log

- Family config:
  - `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\round2_pure_sde\aaai2027_round2_sde_optimal_with_heuristics_seed42_b8a2.json`
- Latest follow-on config:
  - `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\round2_pure_sde\followon\tok_pure_latent_spatial\aaai2027_round2_sde_optimal_with_heuristics_seed42_b8a2_from_tok_pure_latent_spatial_epoch_0002_h18fix.launch.json`
- Latest follow-on run:
  - `aaai2027_round2_sde_optimal_with_heuristics_seed42_b8a2_from_tok_pure_latent_spatial_epoch_0002_h18fix`
- Latest follow-on run dir:
  - `./exp/inmortal-exp/aaai2027_round2_sde_optimal_with_heuristics_seed42_b8a2_from_tok_pure_latent_spatial_epoch_0002_h18fix`
- Warm-start parent:
  - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c11/epoch_0002.pt`
- Contract:
  - `style_tokenizer = null`
  - `tokenizer_family = pure_latent_spatial`
  - `transport_prediction_mode = endpoint`
  - `solver_family = solver_i2sb`
  - `bridge.objective_mode = i2sb_endpoint`
  - DINO retired from the active lane
  - role:
    - heuristic ablation only

## Calibration: `h20`

- launch time:
  - `2026-06-12 17:27:50 +08:00`
- batch:
  - `20`
- first health check:
  - `10677 MiB`
- later sampled GPU:
  - `9631 MiB`
- difference vs clean line:
  - `use_diffeomorphic_stroke = true`
  - `style_injection_mode = body_decoder`
- lane status:
  - no retained checkpoint
  - failed in `epoch_1`
  - `RUNTIME_GUARD used=11107 MiB cap=11000 MiB`

## Active Launch: `h18`

- launch time:
  - `2026-06-12 17:36:55 +08:00`
- batch:
  - `18`
- first health check:
  - `9101 MiB`
- later sampled GPU:
  - `9631 MiB`
- difference vs clean line:
  - `use_diffeomorphic_stroke = true`
  - `style_injection_mode = body_decoder`
- first settled eval:
  - `epoch_0001`
  - completed at `2026-06-12 18:03:01 +08:00`
  - transfer:
    - `clip_style = 0.707659`
    - `content_lpips = 0.665060`
  - all-pairs:
    - `clip_style = 0.713311`
    - `content_lpips = 0.664216`
  - eval timing:
    - `wall_total = 224.61s`
    - `eval_total = 33.57s`
    - `generation = 120.38s`
    - `vae_decode = 58.31s`
- current read:
  - style is still below `clean30 epoch_0001` and legacy `b24c3 epoch_0001`
  - LPIPS is the strongest result so far among the audited true-I2SB / true-tokenizer branches
  - lane has already resumed into `Epoch 2`
  - latest sampled GPU is about `10590 MiB`, still under the hard cap but with little headroom
  - during the latest watch window, the lane stayed alive in `Epoch 2` for multiple minutes at about `10590 MiB`
  - no `epoch_0002.pt` is present yet

## Calibration Refresh: `h18fix`

- launch time:
  - `2026-06-12 18:35:00 +08:00`
- batch:
  - `18`
- first settled eval:
  - `epoch_0001`
  - completed at `2026-06-12 18:59:27 +08:00`
  - transfer:
    - `clip_style = 0.698094`
    - `content_lpips = 0.703000`
  - all-pairs:
    - `clip_style = 0.701402`
    - `content_lpips = 0.702265`
  - eval timing:
    - `wall_total = 210.96s`
    - `eval_total = 31.73s`
    - `generation = 112.83s`
    - `vae_decode = 54.58s`
- training read before stop:
  - resumed into `Epoch 2`
  - training loss remained dominated by the curvature branch
  - the lane was consuming about `10.6 GiB`
- comparison to prior heuristic point:
  - weaker than `h18 epoch_0001` on both `transfer` and `all_pairs`
  - still weaker in style than `clean30 epoch_0001`

## Decision

- `h18fix` is frozen as an ablation-only reference at `epoch_0001`
- stop time:
  - `2026-06-12 19:04:00 +08:00`
- stop reason:
  - the single remote lane was reassigned to the clean mainline after the user explicitly prioritized the true tokenizer + true I2SB path
  - `h18fix epoch_0001` did not justify continued occupancy because it regressed relative to the original `h18 epoch_0001`
- next role:
  - retain the heuristic branch only as a structure-favoring ablation baseline
