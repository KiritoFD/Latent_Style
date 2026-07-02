# sde_i2sb_sigma_0p5 Remote Run Log

- Active config:
  - `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\round2_pure_sde\followon\tok_pure_latent_spatial\aaai2027_round2_sde_i2sb_sigma_0p5_seed42_b8a2_from_tok_pure_latent_spatial_epoch_0002_b28c4.launch.json`
- Active run:
  - `aaai2027_round2_sde_i2sb_sigma_0p5_seed42_b8a2_from_tok_pure_latent_spatial_epoch_0002_b28c4`
- Active run dir:
  - `./exp/inmortal-exp/aaai2027_round2_sde_i2sb_sigma_0p5_seed42_b8a2_from_tok_pure_latent_spatial_epoch_0002_b28c4`
- Warm-start parent:
  - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c11/epoch_0002.pt`
- Contract:
  - `tokenizer_family = pure_latent_spatial`
  - `transport_prediction_mode = endpoint`
  - `solver_family = solver_i2sb`
  - `bridge.objective_mode = i2sb_endpoint`
  - `semantic_supervision_family = legacy_terminal_swd`
  - DINO retired from the active lane
  - preferred band `9.0-10.8 GiB`
  - hard stop `11.0 GiB`

## Calibration History

- `b34c1`
  - first health: `9288 MiB`
  - failed above cap at `11340 MiB`
- `b32c2`
  - first health: `8276 MiB`
  - failed above cap at `11444 MiB`
- `b30c3`
  - first health: `8518 MiB`
  - first settled eval:
    - transfer `0.696403 / 0.700698`
    - all-pairs `0.699444 / 0.693821`
  - failed above cap at `11768 MiB`
- `b28c4`
  - first health: `7644 MiB`
  - first settled eval completed and training resumed

## Current Lane: `b28c4`

- first settled checkpoint:
  - `epoch_0001.pt`
- first settled eval:
  - transfer:
    - `clip_style = 0.689155`
    - `content_lpips = 0.751470`
  - all-pairs:
    - `clip_style = 0.691801`
    - `content_lpips = 0.744453`
- eval timing:
  - `wall_total = 219.98s`
  - `eval_total = 33.39s`
  - `generation = 117.00s`
  - `vae_decode = 56.78s`
- resume evidence:
  - remote full eval finished at `2026-06-12 14:09:31 +08:00`
  - training restored to CUDA and resumed into `Epoch 2/24`
- latest checked GPU sample:
  - `10818 MiB`
  - still below the hard cap

## Decision

- `b28c4` is no longer the active remote lane
- keep this run as the first recorded `sigma=0.5` solver reference point
- compare future `sigma=0.25` and `sigma=1.0` lanes against this point instead of pretending this family is still actively training
