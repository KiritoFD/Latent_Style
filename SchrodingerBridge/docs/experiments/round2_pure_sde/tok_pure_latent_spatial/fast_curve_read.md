# tok_pure_latent_spatial Fast Curve Read

- Curve CSV:
  - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c11/full_eval/clip_lpips_curve.csv`
- Curve summary:
  - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c11/full_eval/curve_summary.json`
- Convergence:
  - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_round2_tok_pure_latent_spatial_seed42_b8a2_c11/full_eval/round2_convergence.json`

## Latest Settled Point

- checkpoint:
  - `epoch_0003.pt`
- transfer:
  - `clip_style = 0.705098`
  - `content_lpips = 0.601646`
- all-pairs:
  - `clip_style = 0.715398`
  - `content_lpips = 0.596325`
- identity:
  - `clip_style = 0.756596`
  - `content_lpips = 0.575041`
- eval timing:
  - `wall_total = 106.72s`
  - `eval_total = 29.59s`
  - `generation = 10.36s`
  - `vae_decode = 55.38s`

## Baseline Comparison

- compared with `tok_baseline_global epoch_0001`:
  - transfer style improved:
    - `0.680867 -> 0.702593`
  - all-pairs style improved:
    - `0.708709 -> 0.718218`
  - transfer LPIPS worsened:
    - `0.416732 -> 0.535566`
  - all-pairs LPIPS worsened:
    - `0.411090 -> 0.531715`

## Current Read

- `c11 epoch_0003` is now the latest settled eval evidence.
- the first corrected point preserves the same tradeoff seen in earlier exploratory pure-latent attempts:
  - more style than the global-only tokenizer
  - materially worse LPIPS
- frontier read:
  - `c10 epoch_0001` remains the structure-friendlier tokenizer point
  - `c11 epoch_0002` remains the best all-pairs style point
  - `c11 epoch_0003` is now the best transfer-style point
  - the tokenizer family currently has at least two Pareto points:
    - `epoch_0001`
    - `epoch_0002`
    - `epoch_0003`
- current implication:
  - the latent-native tokenizer is real
  - the current injection / decoder contract still pays too much structure cost
- active live lane note:
  - `c11` landed `epoch_0003`
  - after the third eval and resume, the lane continued into `epoch_4`
  - no `>11 GiB` hard-cap event has appeared through the observed `epoch_4` window
  - so `c11` is now both:
    - the newest settled tokenizer-wave curve point
    - the longest-lived below-cap tokenizer-wave lane so far

## Next Use

- use `c11 epoch_0001` as the newest active tokenizer-wave evidence
- use the current tokenizer frontier for handoff:
  - `c10 epoch_0001` as the structure anchor
  - `c11 epoch_0002` as the best all-pairs style anchor
  - `c11 epoch_0003` as the best transfer-style anchor
- keep watching later retained checkpoints before calling the tokenizer family formally converged
- if `c11` stays below the cap through later epochs, promote it as the tokenizer-wave reference for the solver sweep
- if the line stays style-forward but structure-costly, close tokenizer wave with:
  - `tok_baseline_global` as the structure anchor
  - `tok_pure_latent_spatial c8` as the corrected style-forward tokenizer result
