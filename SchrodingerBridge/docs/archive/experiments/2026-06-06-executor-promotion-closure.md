# Executor Promotion Closure

Date: 2026-06-06

Scope:

- packet: `A1`
- config:
  - [executor_promotion_h_e1_seed42_b44.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/executor_promotion_h_e1_seed42_b44.json)
- dataset: `Distinct5-512`
- machine: remote `RTX 3060 WSL`

## Summary

This packet is now fully closed with training plus full evaluation for
`epoch_0001 .. epoch_0003`.

The result is not a promoted improvement.

## Closed artifacts

Run root:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_executor_promotion_h_e1_seed42_b44`

Closed eval payloads:

- `full_eval/epoch_0001/summary.json`
- `full_eval/epoch_0002/summary.json`
- `full_eval/epoch_0003/summary.json`

## Readout

Observed transfer-side read:

- epoch 2:
  - `clip_style = 0.6650`
  - `content_lpips = 0.3401`
- epoch 3:
  - `clip_style = 0.6642`
  - `content_lpips = 0.3418`

Comparison target:

- paper-facing `H` family

Interpretation:

- executor-side promotion does not create a cleaner frontier than the current
  reviewed `H` packet
- style movement stays in roughly the same band
- content preservation is worse than the paper-facing `H e1`

## Closure

Current closure:

- `A1` is a **negative-to-neutral closure**
- it is useful evidence that the executor-refresh idea is not the next
  headline promotion path on the current `H` surface
- it should not be promoted into a larger branch family
