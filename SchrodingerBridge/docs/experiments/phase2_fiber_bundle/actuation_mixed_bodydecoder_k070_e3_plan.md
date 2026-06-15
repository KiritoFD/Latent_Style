# Actuation Mixed Body+Decoder Probe

Date: 2026-06-15

## Goal

Test the fiber-bundle diagnosis that the style residual bottleneck is not only
late output rank, but where style enters the transport computation. R16 showed
that a larger output-side basis is active but does not create a meaningful
style breakthrough. This lane moves actuation into the body+decoder feature
path while keeping the rest of the experiment fixed.

## Controlled Delta

- Parent:
  `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`
- Config:
  `configs/aaai2027/phase2_actuation_mixed_bodydecoder_k070_e3_b32bf16_vlen010.json`
- Mechanism delta:
  `model.style_injection_mode=body_decoder`,
  `model.style_injection_form=mixed`.
- Explicitly disabled:
  `model.style_delta_mode=none`, `model.style_delta_scale=0.0`.
- Held fixed: tokenizer, solver, losses, TopoGate, output appearance head,
  k070 e3 parent, `freeze_mode=injection_only`, b32/bf16/channels-last
  throughput lane, `virtual_length_multiplier=0.10`, and training-time
  transfer-only CLIP-S/LPIPS eval.

## Decision Rule

- Primary: transfer CLIP-S, style-first toward Seedream.
- LPIPS budget: values near `0.35` are acceptable if style rises clearly.
- Matched control: compare against k070 e3 parent and the closed R16/S030
  actuation-only lanes, not against unrelated datasets or external boards.
- Positive evidence requires transfer CLIP-S to exceed the R16 e2 full-board
  closure (`0.674395 / 0.352223`) by a meaningful margin and avoid a pure
  LPIPS-cost trade.
- If e1/e2 immediately flatten around k070/R16 levels, inspect runtime
  observability for mixed injection magnitude before interpreting as a theory
  failure.

## Launch Log

Pending remote WSL launch.

## Running Eval Curve

Pending. Expected local mirror:
`docs/experiments/phase2_fiber_bundle/eval/actuation_mixed_bodydecoder_k070_e3_b32bf16_vlen010/clip_lpips_curve.csv`

| epoch | transfer CLIP-S | transfer LPIPS | eval wall | transfer-only |
|---|---:|---:|---:|---:|

## Closure Decision

Pending.
