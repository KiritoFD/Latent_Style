# I2SB + PnP + Fiberwise Mixed Screen Closure

Date: 2026-06-15

## Scope

This run is archived as a mixed-mechanism screen, not as controlled Fiber Bundle evidence.

- Config: `configs/aaai2027/phase2_i2sb_pnp_fiber_sde_k070.json`
- Parent: `exp/aaai2027_phase2_smoe_translator_k070_e3_seed42_b12a1/epoch_0008.pt`
- Changed mechanisms: endpoint/I2SB objective, PnP self-injection, fiberwise SWD, SMoE parent.
- Dataset/eval surface: `wikiarts5_full_notest_train__distinct5_512_classview_test`
- Training-time eval: enabled; two retained checkpoints were evaluated on remote during training.

## Curve

| ckpt | transfer CLIP-S | transfer LPIPS | all-pairs CLIP-S | all-pairs LPIPS | eval wall sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| epoch_0001 | 0.684073 | 0.394578 | 0.707963 | 0.394165 | 230.0 |
| epoch_0002 | 0.683612 | 0.407860 | 0.708105 | 0.406711 | 225.4 |

Source files:

- `docs/experiments/phase2_fiber_bundle/curves/i2sb_pnp_fiber_sde_k070_remote_clip_lpips_curve.csv`
- `docs/experiments/phase2_fiber_bundle/eval/i2sb_pnp_fiber_sde_k070_curve_summary.json`

## Decision

Status: `cost_stopped_mixed_negative`.

The run is style-positive relative to the conservative k070 line, but LPIPS is immediately out of band and worsens from e1 to e2. Because multiple mechanisms changed at once, this result cannot identify whether the failure comes from endpoint parameterization, PnP self-injection, fiberwise SWD, the SMoE parent, or their interaction.

## Next Action

Return to controlled single-mechanism tests. Based on `docs/612-phase2/fiber.md`, the next clean experiment should target the actuation bottleneck: keep tokenizer, solver, losses, schedule, and TopoGate fixed, and change only the style injection/output actuation path. Record generated-delta rank and off-diagonal cosine before drawing a mechanism conclusion.
