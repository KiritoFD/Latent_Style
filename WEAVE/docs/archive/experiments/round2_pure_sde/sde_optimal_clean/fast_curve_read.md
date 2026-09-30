# sde_optimal_clean Fast Curve Read

- Curve CSV: `clip_lpips_curve.csv`
- Active lane curve:
  - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round2_sde_optimal_clean_seed42_b8a2_from_tok_pure_latent_spatial_epoch_0002_c26main/full_eval/clip_lpips_curve.csv`

## Current State

- `c26main epoch_0001`:
  - transfer `0.681721 / 0.741877`
  - all-pairs `0.685036 / 0.735789`
  - eval timing:
    - `wall_total = 198.86s`
    - `eval_total = 31.91s`
    - `generation = 101.09s`
    - `vae_decode = 54.47s`
- immediate comparison:
  - much weaker than `sigma_0p25 clean30 epoch_0001`
  - much weaker than heuristic `h18 epoch_0001`
  - weaker than heuristic `h18fix epoch_0001`
- runtime outcome:
  - survived `epoch_1`
  - still failed after eval+resume at `used=11048 MiB`
- decision:
  - clean `sigma=0.5` is currently not the paper-facing frontier
  - next remote attempt should return to the stronger clean `sigma=0.25` family rather than keep burning time on this branch
