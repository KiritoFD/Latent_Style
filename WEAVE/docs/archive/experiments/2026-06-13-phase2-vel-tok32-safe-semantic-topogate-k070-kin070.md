# Phase 2: vel_tok32_safe_semantic_topogate_k070_kin070

Date: 2026-06-13

## Role

- queued style-release follow-on behind `k070`
- keep the same true-tokenizer + velocity + topology-gate family
- release one more degree of motion freedom:
  - `w_kinetic: 0.85 -> 0.70`

## Why This Exists

- `appalign` proved the family can hold LPIPS near `0.31`
- `i2sb_sigma0.02_tfloor005` proved stochastic exact-I2SB raises style but immediately leaves the paper LPIPS band
- `k070` is therefore the first clean training-side style-release follow-on
- if `k070` is still style-limited while remaining in-band, the next least-invasive knob is to reduce kinetic regularization before touching style-path gains

## Config

- config:
  - [phase2_vel_tok32_safe_semantic_topogate_k070_kin070_seed42_b12a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_semantic_topogate_k070_kin070_seed42_b12a1.json)
- parent packet:
  - [phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1.json)

## Deltas

- keep:
  - `semantic_self_topology_blend = 0.7`
  - `tokenizer_family = pure_latent_spatial`
  - `transport_prediction_mode = velocity`
  - `output_appearance_alignment_mode = tokenizer_latent_affine`
- change:
  - `w_kinetic: 0.85 -> 0.70`

## Intended Read

- success:
  - transfer style rises without reopening the large LPIPS penalty
  - all-pairs remains near the recovered `0.70x / 0.31x` band
- failure:
  - LPIPS rises with no meaningful style gain
  - or the line simply repeats the same plateau at weaker structure control

## 2026-06-14 Cost Stop

- Remote launch: `phase2-structure-k070-kin070-train`.
- Runtime read: epoch `1/24` reached only about `9%` after roughly `2.2min`; projected epoch time was still about `24-25min` before full eval.
- VRAM read: about `6.9 GiB / 12 GiB`, inside the formal cap; the stop was cost/value driven, not OOM driven.
- Eval status: no checkpoint and no `CLIP-S + LPIPS` point were produced.
- Decision: `cost_stopped_no_eval`. Do not treat this as a negative model-quality result. It only says a full-data, full-length retrain for a single `w_kinetic` delta is not worth the remote 3060 time under the current evidence.
- Follow-up rule: revisit kinetic release only through a shorter virtual-length probe, an eval-only surrogate, or a scheduled kinetic decay that can be judged quickly against matched controls.
