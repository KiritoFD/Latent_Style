# Actuation Mixed Body+Decoder Closure

Date: 2026-06-16

## Status

- Run id: `aaai2027_phase2_actuation_mixed_bodydecoder_k070_e3_b32bf16_vlen010`.
- Parent: `k070 epoch_0003`.
- Controlled switch delta: `model.style_injection_mode=body_decoder` with
  `model.style_injection_form=mixed`; `style_delta_mode=none`.
- Held fixed: tokenizer, solver, losses, TopoGate, appearance head, parent,
  `freeze_mode=injection_only`, b32 bf16 throughput lane, and transfer-only
  training eval.
- Closure status: `converged_not_promoted`.

## Evidence

- Retained checkpoints evaluated: `epoch_0001` through `epoch_0015`.
- Curve: `docs/experiments/phase2_fiber_bundle/eval/actuation_mixed_bodydecoder_k070_e3_b32bf16_vlen010/clip_lpips_curve.csv`.
- Formal convergence: `converged=true`, `best_epoch=epoch_0002`,
  `last_pareto_epoch=epoch_0006`, `since_best=13`,
  `since_last_pareto=9`, `tail_flat=true`.

| point | transfer CLIP-S | transfer LPIPS | note |
| --- | ---: | ---: | --- |
| best style `epoch_0002` | 0.673771 | 0.350758 | best style/all-pairs point |
| final `epoch_0015` | 0.672634 | 0.350382 | no late recovery |

## Decision

Close the mixed body+decoder lane as `converged_not_promoted`.

The mechanism was stable and stayed within the style-priority LPIPS budget, but
it did not beat the stronger R16 full-board frontier (`0.674395 / 0.352223`) and
did not approach the `0.74` style target. This supports the `fiber.md`
diagnosis: feature-path injection alone is still swallowed by the generated
delta bottleneck.

## Next Action

Do not continue or stack this lane. The next controlled experiment should
directly target generated-delta collinearity and record off-diagonal delta
cosine alongside CLIP-S/LPIPS, so a style failure can be separated from an
implementation/no-op failure.

## Artifacts

- Curve CSV:
  `docs/experiments/phase2_fiber_bundle/eval/actuation_mixed_bodydecoder_k070_e3_b32bf16_vlen010/clip_lpips_curve.csv`
- Curve summary:
  `docs/experiments/phase2_fiber_bundle/eval/actuation_mixed_bodydecoder_k070_e3_b32bf16_vlen010/curve_summary.json`
- Convergence:
  `docs/experiments/phase2_fiber_bundle/eval/actuation_mixed_bodydecoder_k070_e3_b32bf16_vlen010/round2_convergence.json`
