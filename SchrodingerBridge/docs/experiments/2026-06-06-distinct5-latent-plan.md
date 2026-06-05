# Distinct5 Latent Baseline Plan

Date: 2026-06-06

Scope:

- latent `SaMAM` on `Distinct5-512`
- latent `SaMST` on `Distinct5-512`
- paper purpose:
  - add latent baselines into the `same-cost` comparison
  - test whether either latent baseline becomes paper-relevant after more
    training than the very early same-cost point

## Updated protocol

Each latent baseline now has two separate lanes:

1. `same-cost`
2. `convergence`

The two lanes are evaluated differently on purpose.

### Lane A: same-cost

Goal:

- compare against the current minute-scale LBM story on `Distinct5-512`

Selection rule:

- use only `CLIP-S` and `LPIPS`
- do not gate on `ArtFID`

Reason:

- `ArtFID` is too slow for dense early checkpoint screening
- current evaluator evidence shows the main bottlenecks are metric-side, not
  the model forward itself

Retained-point policy:

- save checkpoints densely enough to recover wall-clock-near points
- first target points:
  - about `2 min`
  - about `10 min`
- use the nearest retained checkpoint to each wall-clock target

### Lane B: convergence

Goal:

- test whether the latent baseline becomes competitive after longer training

Selection rule:

- use only `CLIP-S` and `LPIPS`
- track the best retained point along the longer curve

Convergence criterion:

- stop when additional retained points no longer improve the `CLIP-S / LPIPS`
  trade-off in a meaningful way
- this is a practical paper criterion, not a theorem

## Final expensive closure

Only after the two lanes are screened:

- `SaMAM same-cost best point`
- `SaMAM convergence best point`
- `SaMST same-cost best point`
- `SaMST convergence best point`

Run `ArtFID` only for these final four points.

This keeps the expensive metric confined to the final paper-facing packet.

## Current operational notes

- remote GPU policy remains:
  - single active lane
  - hard cap `< 11.0 GiB`
- `Distinct5-512` latent presets must use WSL-native paths:
  - train latents:
    - `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train`
  - held-out latents:
    - `/mnt/i/wikiart_distinct5_latents_512_ema_test`
  - classview images:
    - `/mnt/i/wikiart_distinct5_samam_512_classview/test`
- the first attempted `SaMAM same-cost` launch failed because older wrapper
  presets still used Windows-style `F:/...` paths; this was repaired locally on
  `2026-06-06`
- the second attempted `SaMAM same-cost` launch reached the real path audit and
  showed that the older `.../test` and `..._classview_real/...` roots do not
  exist on the active remote machine; the plan now follows the audited roots
  above

## Immediate execution order

1. relaunch latent `SaMAM` `Distinct5 same-cost`
2. monitor first-health and checkpoint creation
3. derive nearest retained checkpoints to `2 min` and `10 min`
4. fast-eval those retained points with `CLIP-S + LPIPS` only
5. launch latent `SaMAM` convergence lane
6. repeat the same two-lane process for latent `SaMST`
7. run `ArtFID` only on the final four retained points
8. then return to `LBM` main-model optimization
