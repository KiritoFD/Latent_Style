# sde_optimal_clean Remote Run Log

- Run dir: `./exp/inmortal-exp/aaai2027_round2_sde_optimal_clean_seed42_b8a2`

## Active Launch: `c26main`

- active config:
  - `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\round2_pure_sde\followon\tok_pure_latent_spatial\aaai2027_round2_sde_optimal_clean_seed42_b8a2_from_tok_pure_latent_spatial_epoch_0002_c26main.launch.json`
- active run:
  - `aaai2027_round2_sde_optimal_clean_seed42_b8a2_from_tok_pure_latent_spatial_epoch_0002_c26main`
- active run dir:
  - `./exp/inmortal-exp/aaai2027_round2_sde_optimal_clean_seed42_b8a2_from_tok_pure_latent_spatial_epoch_0002_c26main`
- launch time:
  - `2026-06-12 19:09:14 +08:00`
- batch:
  - `26`
- warm-start parent:
  - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c11/epoch_0002.pt`
- clean contract:
  - `style_tokenizer = null`
  - `tokenizer_family = pure_latent_spatial`
  - `solver_family = solver_i2sb`
  - `bridge.objective_mode = i2sb_endpoint`
  - `bridge.loss_type = mse`
  - `use_diffeomorphic_stroke = false`
  - `style_injection_mode = none`
- health:
  - prelaunch GPU = `530 MiB`
  - `30s` check = `7912 MiB`
  - follow-up sample = `7686 MiB`
- training read:
  - `Epoch 1/24` is active
  - first logged step:
    - `flow = 0.1443`
    - `loss = 0.4109`
    - `terminal_swd = 0.0148`
- decision:
  - keep `c26main` as the only active remote lane for the true clean mainline
  - accept the current under-band launch as the safer calibration point because earlier clean `28-30` runs exceeded the hard cap after eval+resume
  - next gate is the first settled `epoch_0001` full eval

## `c26main` Result

- `epoch_0001`:
  - completed at `2026-06-12 19:27:51 +08:00`
  - transfer:
    - `clip_style = 0.681721`
    - `content_lpips = 0.741877`
  - all-pairs:
    - `clip_style = 0.685036`
    - `content_lpips = 0.735789`
  - eval timing:
    - `wall_total = 198.86s`
    - `eval_total = 31.91s`
    - `generation = 101.09s`
    - `vae_decode = 54.47s`
- epoch-1 train summary:
  - `loss = 0.4212`
  - `flow = 0.1085`
  - `terminal_swd = 0.0174`
  - `peak = 5.67 / 7.59 GiB`
- post-eval resume:
  - resumed into `Epoch 2`
  - later hit:
    - `RUNTIME_GUARD used=11048 MiB cap=11000 MiB`
  - exit time:
    - `2026-06-12 19:30:29 +08:00`
- read:
  - the line is genuinely clean:
    - `curv = 0`
    - no heuristic structure penalties are active
  - despite that, the first retained point is much weaker than the audited clean `sigma_0p25` points

## Decision

- `c26main` is recorded as negative evidence for the clean `sigma=0.5` mainline
- do not promote it
- move the remote lane back to the stronger `sigma_0p25` clean family for further survival calibration
- lane reassigned at:
  - `2026-06-12 19:33:45 +08:00`
