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
- the first formal Distinct5 latent `SaMAM` same-cost machine audit is now
  negatively closed under the reviewed `3060` contract:
  - see [2026-06-06-samam-latent-distinct5-11g-gate.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-06-samam-latent-distinct5-11g-gate.md)
  - do not continue spending `Distinct5` GPU budget on this lane until a new
    low-VRAM mechanism is identified
- latent `SaMST` same-cost now has a closed packet too:
  - machine-side launch is healthy under the reviewed `3060` contract
  - quality-side closure is negative because the training packet collapses to
    `nan` losses and zero-direction outputs
  - see [2026-06-06-samst-latent-distinct5-samecost-closure.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-06-samst-latent-distinct5-samecost-closure.md)
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

1. stop expanding latent `SaMAM / SaMST` on `Distinct5` until a concrete
   training-stability mechanism exists
2. shift the next baseline slot to the `SD1.5 LoRA` line on the same
   `Distinct5-512` surface
3. keep the same same-cost / convergence reporting contract
4. return leftover GPU budget to `LBM` main-model optimization once the first
   LoRA same-cost packet is closed
