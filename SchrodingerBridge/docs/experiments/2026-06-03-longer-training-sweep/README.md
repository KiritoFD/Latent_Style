# LBM Longer-Training Sweep

Date: 2026-06-03

## Purpose

Test the simple hypothesis that the current LBM operating points may be
under-trained on Distinct5-512. This packet must isolate training duration from
architecture, tokenizer, loss, and data changes.

## Current baseline facts

Paper-facing Distinct5-512 transfer-only points:

| point | CLIP-S | LPIPS | targetwise ArtFID | train |
| --- | ---: | ---: | ---: | ---: |
| LBM-F e1 | 0.664360 | 0.324528 | 126.826 | 1.2m |
| LBM-K e1 | 0.671167 | 0.372281 | 161.958 | 1.2m |
| IDT | 0.639921 | 0.000000 | 323.661 | ref |

F is the low-LPIPS base. K is the high-style base.

## New configs

- `configs/aaai2027/longer_train_f_seed42_b44_e8.json`
- `configs/aaai2027/longer_train_k_seed42_b44_e8.json`

Both configs:

- use Distinct5-512 EMA latents on remote `/mnt/i`
- use formal RTX 3060 batch size `44`
- train for `8` epochs
- save every epoch
- run deferred full eval for every saved epoch
- keep architecture, tokenizer, losses, queue type, seed, and data split fixed

## Remote launch rule

Do not interrupt Dalton's SaMAM `2750 -> 3000` sidecar. At the time this packet
was created, Dalton reported:

- SaMAM segment `step_003000` still running
- progress approximately `205 / 250`
- GPU about `8432 / 12288 MiB`
- no `3000` checkpoint, eval, metrics, or targetwise ArtFID yet

Faraday may launch this sweep only after the GPU is free or after Dalton
explicitly reports that the SaMAM sidecar no longer needs the device.

## Evaluation gates

Use transfer-only metrics first. Keep the best checkpoint from the full e1--e8
sweep, not necessarily the final epoch.

Preserve if any OR condition holds:

- transfer CLIP-S improves by at least `+0.006` over its own base; or
- transfer LPIPS drops by at least `0.025` while CLIP-S drops by no more than
  `0.003`; or
- targetwise ArtFID improves materially without CLIP-S/LPIPS degradation.

For candidates that pass CLIP-S/LPIPS screening, compute standalone
`aggregate_targetwise_artfid.json`; do not use embedded summary ArtFID as a
paper-table substitute unless the table explicitly says so.

## Interpretation rules

- If later epochs improve, update the Distinct5 frontier and Table 4.
- If e1 remains best, write the result as early convergence, not failure.
- If F improves style but loses LPIPS, treat it as a frontier-shape result.
- If K reduces LPIPS without losing style, prioritize it for the paper.
- If neither improves over IDT-adjusted frontier, do not tune the same axis
  blindly; return to representation/execution design.

## Closure

Closed in:

- [2026-06-06-distinct5-longer-training-closure.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-06-distinct5-longer-training-closure.md)

Final read:

- `F-longer` is a negative closure:
  - best retained balance stays at the earliest closed point
  - later epochs recover only a small amount of style while LPIPS worsens
- `K-longer` is also a negative closure:
  - `e5` is the best retained balance point
  - `e6 .. e8` recover style toward `K e1`, but LPIPS and targetwise ArtFID
    both worsen

Safe summary:

- the current minute-scale Distinct5 operating points are **not** explained away
  by a simple "train the same family longer" objection
- the next GPU slot should return to mechanism or split-validation work, not to
  more same-family `F/K` longer training
