# sde_i2sb_sigma_0p5 Fast Curve Read

- Curve CSV:
  - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round2_sde_i2sb_sigma_0p5_seed42_b8a2_from_tok_pure_latent_spatial_epoch_0002_b28c4/full_eval/clip_lpips_curve.csv`
- Curve summary:
  - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round2_sde_i2sb_sigma_0p5_seed42_b8a2_from_tok_pure_latent_spatial_epoch_0002_b28c4/full_eval/curve_summary.json`
- Convergence summary:
  - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round2_sde_i2sb_sigma_0p5_seed42_b8a2_from_tok_pure_latent_spatial_epoch_0002_b28c4/full_eval/round2_convergence.json`

## Current Curve

- settled row count:
  - `1`
- latest checkpoint:
  - `epoch_0001.pt`
- latest transfer:
  - `clip_style = 0.689155`
  - `content_lpips = 0.751470`
- latest all-pairs:
  - `clip_style = 0.691801`
  - `content_lpips = 0.744453`
- latest identity:
  - `clip_style = 0.702381`
  - `content_lpips = 0.716386`
- eval timing:
  - `wall_total = 219.98s`
  - `eval_total = 33.39s`
  - `generation = 117.00s`
  - `vae_decode = 56.78s`

## Interpretation

- this is the first true wave-2 `sigma=0.5` lane that has both:
  - produced a retained remote eval point
  - resumed training afterward without an immediate hard-cap kill
- it is not yet a solver win:
  - the first solver point is materially weaker than the tokenizer handoff frontier on both `transfer` and `all_pairs`
- it is still important evidence:
  - the exact-posterior I2SB path is now real end-to-end on the remote `3060`
  - later checkpoints decide whether `sigma=0.5` becomes a useful solver family or just a stability milestone

## Operational Read

- keep writing every retained checkpoint into the same curve
- do not judge convergence from one point
- the active remote lane has already moved on to `sigma_0p25 / b28c1`
- keep this note frozen as the first `sigma=0.5` reference snapshot unless a new `sigma=0.5` retry is explicitly launched
