# sde_optimal_with_heuristics Fast Curve Read

- Active lane curve:
  - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round2_sde_optimal_with_heuristics_seed42_b8a2_from_tok_pure_latent_spatial_epoch_0002_h18/full_eval/clip_lpips_curve.csv`

## Current State

- `h18 epoch_0001`:
  - transfer `0.707659 / 0.665060`
  - all-pairs `0.713311 / 0.664216`
  - eval timing:
    - `wall_total = 224.61s`
    - `eval_total = 33.57s`
    - `generation = 120.38s`
    - `vae_decode = 58.31s`

## Immediate Read

- compared against the best audited true-clean point:
  - `h18 epoch_0001` has weaker style than `clean30 epoch_0001`
  - but better LPIPS than `clean30 epoch_0001`
- compared against the legacy-snapshot frontier:
  - `h18 epoch_0001` is still weaker than `b24c3 epoch_0001` on both `transfer` and `all_pairs`
- current runtime:
  - `h18` has already resumed into `Epoch 2`
  - latest sampled GPU is about `10590 MiB`
  - this is still under the `11.0 GiB` hard cap, but there is little headroom left

## Next Gate

- wait for the next retained checkpoint from `h18`
- decide whether the heuristic branch can keep its LPIPS advantage without losing too much style or crossing the hard cap
