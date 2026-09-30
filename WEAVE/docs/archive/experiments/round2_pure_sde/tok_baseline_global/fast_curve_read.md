# tok_baseline_global Fast Curve Read

- Curve CSV:
  - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round2_tok_baseline_global_seed42_b8a2/full_eval/clip_lpips_curve.csv`
- Curve summary:
  - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round2_tok_baseline_global_seed42_b8a2/full_eval/curve_summary.json`

## First Settled Point

- checkpoint:
  - `epoch_0001.pt`
- transfer:
  - `clip_style = 0.680867`
  - `content_lpips = 0.416732`
- all-pairs:
  - `clip_style = 0.708709`
  - `content_lpips = 0.411090`
- identity:
  - `clip_style = 0.820075`
  - `content_lpips = 0.388520`
- eval timing:
  - `wall_total = 111.47s`
  - `full_eval completed = 132.6s`

## Initial Read

- this is the tokenizer control baseline on the new `wikiarts_5_full_notest` train root
- relative to the `sigma_0p5` solver line:
  - structure is much stronger at the first point
  - style strength is materially weaker
- next use:
  - wait for `epoch_0002+`
  - then compare whether pure latent routing can recover style without giving up the strong LPIPS envelope
