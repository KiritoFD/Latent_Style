# Topology-Release Eval-Only Closure

Date: 2026-06-14

## Scope

- Parent: `k070 epoch_0003`.
- Tested: inference-only `semantic_self_topology_blend = 0.5, 0.3, 0.0`.
- Control: the trained parent setting is `semantic_self_topology_blend = 0.7`, represented by the existing `k070 epoch_0003` full eval.
- Fixed variables: checkpoint, tokenizer, solver, loss, appearance path, test set, CLIP-S/LPIPS contract, and seed `42`.
- No training was run.

## Results

| blend | transfer CLIP-S | transfer LPIPS | all-pairs CLIP-S | all-pairs LPIPS | decision |
|---:|---:|---:|---:|---:|---|
| 0.7 parent | 0.671820 | 0.314618 | 0.703234 | 0.312550 | control |
| 0.5 | 0.671887 | 0.314608 | 0.703252 | 0.312524 | flat |
| 0.3 | 0.671899 | 0.314675 | 0.703265 | 0.312592 | flat |
| 0.0 | 0.671696 | 0.314660 | 0.703089 | 0.312572 | flat |

## Decision

- Lowering the topology blend at inference does not release meaningful style on the trained `k070` parent.
- The best transfer delta is only `+0.000079` at `blend=0.3`, and all-pairs improves only `+0.000032`; both are below the material threshold.
- Structure is also effectively unchanged, which means the trained representation is not using this inference knob as a style bottleneck.
- Decision: `flat_no_training_value`. Do not spend a training lane on further isolated topology-blend reduction under this parent.

## Next

- Prefer mechanisms that change the style signal path rather than only loosening the topology mask at inference.
- Candidate low-cost directions: eval-only appearance-head scale/gain scans, or a short-probe tokenizer capacity change only if the eval-only screen shows a real style slope.
