# Actuation Pre-Decoder Style Section Closure

## Status

- Run id: `aaai2027_phase2_actuation_predec_section_k070_e3_b16a2bf16_vlen010`.
- Parent: `k070 epoch_0003`.
- Mechanism: `model.style_delta_mode=predec_section`, injection-only training.
- Eval contract: `full_eval_fast10`, transfer-only, `10` source samples per style,
  ONNX VAE decode batch `16`, source latent cache, no generated PNGs.
- Retained checkpoints evaluated: `epoch_0001` through `epoch_0018`.

## Fast10 Result

| point | transfer CLIP-S | transfer LPIPS | note |
| --- | ---: | ---: | --- |
| best style `epoch_0003` | 0.680121 | 0.315600 | best style point |
| final `epoch_0018` | 0.680047 | 0.315528 | no late recovery |

- Best-to-final style delta: `-0.000074`.
- Best-to-final LPIPS delta: `-0.000072`.
- Average eval wall excluding the in-process probe outlier e16: `27.58s`.
- Formal convergence JSON still reports `converged=false` under joint Pareto
  because tiny LPIPS-only improvements keep creating numerically different
  points. For this style-priority experiment, the style curve is flat and the
  mechanism is closed as negative.

## Decision

- Decision: `converged_not_promoted_style_flat`.
- Interpretation: moving the style-conditioned section immediately before
  `dec_out` did not break the generated-delta actuation bottleneck. It preserved
  structure but did not increase style strength.
- Consequence: do not spend more 3060 time on this exact pre-decoder section.
  The next style-priority change must either alter the actual output basis more
  strongly or change the bridge/path objective, not just add a shallow
  zero-init pre-dec feature branch.

## Artifacts

- Curve CSV:
  `docs/experiments/phase2_fiber_bundle/curves/actuation_predec_section_k070_e3_remote_fast10_curve.csv`.
- Convergence JSON:
  `docs/experiments/phase2_fiber_bundle/eval/predec_section_k070_e3_round2_convergence.json`.
- Homepage plot data:
  `docs/experiments/phase2_fiber_bundle/plot_points.csv`.
