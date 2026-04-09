# Integrated Evaluation Report

- Generated: 2026-02-13 15:27:53

## Combined Status

- Experiments indexed: 53
- Experiments with metrics: 35
- Strict comparable experiments: 25
- Current strict top score: 0.551533 (full_300_distill_low_only_v1)
- Ablation roots: 3
- Ablation variants tracked: 3
- Ablation completed: 0
- Ablation failed: 2
- Ablation interrupted: 1

## Separate + Integrated Interpretation

- Separate view (historical experiments): style ceiling is around `0.55` in strict comparable runs.
- Separate view (new ablation50): reproducibility pipeline is not yet in stable state (mostly failed/interrupted).
- Integrated view: do not treat new ablation results as competitive evidence until run completion + full_eval outputs exist.

## Balanced Candidate Set (style / cls / lpips)

| run | path | best_style | cls_acc | content_lpips |
|---|---|---:|---:|---:|
| overfit50-style-distill-struct-v2 | overfit50-style-distill-struct-v2 | 0.528368 | 0.850 | 0.551213 |
| overfit50-distill_low_only | overfit50-distill_low_only | 0.518469 | 0.930 | 0.545553 |
| overfit50-strok-style | overfit50-strok-style | 0.516445 | 0.930 | 0.386163 |
| full_300-map16+32 | full_300-map16+32 | 0.509921 | 0.780 | 0.424217 |

## Recommended Next Execution Order

1. Fix ablation launcher paths/env so at least baseline_50e reaches epoch 50 with full_eval.
2. Re-run ablation quick set and regenerate this report.
3. Promote only candidates that satisfy style+cls+lpips gate in this integrated report.
