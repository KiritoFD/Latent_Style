# Anchored I2SB Endpoint Blend: k070 e3 sigma0p02 blend0p25

## Purpose

Test a single new endpoint anchoring mechanism after pure absolute endpoint
lanes proved style-positive but too destructive, and residual endpoint proved
structure-positive but style-negative.

## Controlled Change

- Matched control:
  `configs/aaai2027/phase2_i2sb_clean_k070_e3_sigma0p02_b8a2_vlen010.json`.
- Candidate:
  `configs/aaai2027/phase2_i2sb_blend025_k070_e3_sigma0p02_b8a2_vlen010.json`.
- Changed:
  `model.endpoint_parameterization=absolute -> blend`,
  `model.endpoint_residual_blend=0.25`.
- Unchanged:
  parent checkpoint, `bridge_sigma=0.02`, I2SB endpoint objective, exact
  Brownian schedule, endpoint time floors, tokenizer, TopoGate, appearance
  alignment, semantic cross-attention, terminal SWD, b8a2 schedule, vlen
  `0.10`, and fast10 transfer eval.

## Mechanism

The endpoint head emits the same bounded raw tensor as the absolute endpoint
lane. The new default-off switch converts it to a delta as:

`delta = lerp(raw - x, raw, endpoint_residual_blend)`.

- `endpoint_residual_blend=0.0` is exactly the old absolute endpoint.
- `endpoint_residual_blend=1.0` is exactly residual endpoint.
- `0.25` keeps most absolute style actuation but adds a content anchor.

## Prior Evidence

| lane | best/last read | transfer CLIP-S | transfer LPIPS | decision |
| --- | --- | ---: | ---: | --- |
| absolute sigma0p02 | e2 peak | 0.709094 | 0.490233 | style-positive, high LPIPS |
| residual sigma0p02 | e2 | 0.673869 | 0.308784 | structure-only, style negative |
| absolute sigma0p01 | e1 peak | 0.713162 | 0.590598 | strongest style, too destructive |
| absolute sigma0p01 | e3 stop | 0.701776 | 0.482099 | style reversal, high LPIPS |

## Eval Contract

- Training-time eval subdir: `full_eval_fast10`.
- Transfer-only, `10` source samples per style.
- `CLIP-S + LPIPS` every retained checkpoint.
- Generated-delta observability enabled.
- Training-time eval remains subprocess-isolated:
  `full_eval_in_process=false`, `full_eval_runtime_model_cache=false`.

## Decision Rule

- Positive: transfer CLIP-S remains near the absolute endpoint band while LPIPS
  is materially lower than absolute sigma0p02 at matched epochs.
- Continue if LPIPS is falling and style remains above `0.700`.
- Stop if style falls below `0.700` for two consecutive checkpoints.
- Stop if style reverses and LPIPS remains above `0.42`, matching the absolute
  endpoint failure mode.

## Launch Notes

- Remote WSL repo:
  `/mnt/i/Github/Latent_Style/SchrodingerBridge`.
- This is a controlled mechanism test, not a new tokenizer/backbone/loss.
- If it works, follow with a small blend sweep around the winning anchor
  strength; if it fails, move to a stronger geometry-aware anchor rather than
  another pure sigma scan.
