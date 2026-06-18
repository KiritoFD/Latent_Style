# 2026-06-18 Family Validity Matrix

This report consolidates the probe evidence behind the phase-618 question:

> when different experiment groups are numerically very close, are we seeing a weak theory change,
> or did the implementation fail to move the model path we actually evaluate?

Generated from current probe artifacts by:

```bash
py -3.12 tools/experiments/build_phase618_family_validity_matrix.py
```

## Global invalidators

| ID | Status | Effect | Action | Source |
| --- | --- | --- | --- | --- |
| phase618_auto_family_mutation | fixed | runs could silently downgrade repaired lowrank bases back to legacy_factorized | treat old auto-launched repaired-family results as suspect and rerun from the corrected base | `docs/experiments/2026-06-18-phase616-auto-family-mutation-audit/README.md` |
| lowrank_code_map_order | fixed | lowrank residual style map was decoded from pre-structured style code and understated style separation | use only post-fix lowrank evidence when judging no-reference carrier strength | `docs/experiments/2026-06-18-lowrank-code-map-order-audit/README.md` |
| topogate_last_block_only | fixed | TopoGate OT descriptor used only the last semantic body block; current multiblock audit reports descriptor_blocks=4 and aggregate_minus_last_mean_abs=0.005184 | treat pre-fix h5/h6 artifacts as stale if they are used as evidence about the intended full-body TopoGate descriptor | `docs/experiments/2026-06-18-topogate-multiblock-audit/README.md` |
| style_injection_anatomy_probe_hook_omission | fixed | config-effect forward deltas could be real while anatomy rows under-reported the incremental branch contribution | use only post-fix style-injection anatomy evidence when deciding whether a new style branch stayed identical to baseline | `docs/experiments/2026-06-18-style-injection-live-init-probe/README.md` |
| style_injection_zero_init_exact_noop | known_behavior | a branch can exist in the graph yet remain exactly identical to baseline at initialization, so close early results are not negative evidence | for fair no-reference actuation tests, enable style_injection_live_init or explicitly treat zero-init runs as wake-up-limited controls | `docs/experiments/2026-06-18-style-injection-live-init-probe/README.md` |

## Family summary

| Suite | Base | Config probe | Training probe | Plain eval changes | Exact no-op | Micro | Weak | Moderate | Large | OT/bridge changes | Verdict | Trust |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| stage1_h0_h6_old_base | remote_h1_old_base | yes | yes | 0 | 6 | 0 | 0 | 0 | 0 | 6 | close results are expected because the family is training-real but plain no-reference eval-inert | limited |
| stage1_h0_h6_repaired_lowrank | baseline_h1_repaired_lowrank | yes | yes | 0 | 6 | 0 | 0 | 0 | 0 | 6 | old OT family is still training-real on the repaired base, but pairwise plain eval remains inert across h0-h6 | medium |
| stage3_style_r1_r10_old_base | remote_h1_old_base | yes | no | 4 | 0 | 0 | 0 | 0 | 4 | n/a | old-base style sweep is scientifically confounded because base repair and bold directions are mixed | invalid |
| stage3_style_r1_r10_repaired_lowrank | baseline_h1_repaired_lowrank | yes | no | 7 | 3 | 0 | 7 | 0 | 0 | n/a | only true repaired-base levers remain; carrier-repair variants collapse to no_effect | high |
| bold_r11_r16_repaired_lowrank | baseline_h1_repaired_lowrank | yes | yes | 6 | 0 | 0 | 6 | 0 | 0 | 6 | runtime changes are real but weak; blend/solver tweaks alone are unlikely to rescue style | high |
| plain_path_distill_lowrank | baseline_h1_repaired_lowrank | yes | yes | 0 | 7 | 0 | 0 | 0 | 0 | 7 | plain-path distill is training-real and runtime-inert by design, directly targeting the train/eval contract gap | medium |
| style_injection_live_init_probe | baseline_h1_repaired_lowrank | yes | no | 2 | 2 | 1 | 0 | 1 | 0 | n/a | this calibration probe shows zero-init style-injection variants can be exact no-ops, while live-init variants are runtime-real with mixed stronger than spatial_carrier | high |

Runtime bucket rule:

- `exact_noop`: `plain_forward_delta = 0`
- `micro_runtime_lever`: `0 < plain_forward_delta <= 0.0001`
- `weak_runtime_lever`: `0.0001 < plain_forward_delta <= 0.002`
- `moderate_runtime_lever`: `0.002 < plain_forward_delta <= 0.02`
- `large_runtime_change`: `plain_forward_delta > 0.02`

## Highest-signal conclusions

1. `stage1_h0_h6_old_base` is **not** a universal implementation no-op.
   The training probe says the family is real; the config/eval probe says the benchmarked plain no-reference path stays inert.
2. `stage3_style_r1_r10_old_base` is confounded and should not be used to judge bold directions.
   On the old base, lowrank variants partly win by repairing the carrier rather than by validating the theory.
3. `stage1_h0_h6_repaired_lowrank` removes the dead plain-style carrier explanation, but the old OT family still stays pairwise plain-eval inert.
   That shifts blame away from the old carrier bug and toward objective weakness or contract weakness.
4. `bold_r11_r16_repaired_lowrank` proves that blend / solver changes are real runtime levers, but weak ones.
   The current body-delta uplift over the repaired base is marginal, not paradigm-changing.
5. `plain_path_distill_lowrank` is the cleanest current lever that explicitly targets the train/eval contract gap.
   The paired probes now show it is training-real while remaining runtime-inert at initialization, so any later gain would reflect learned transfer rather than a hidden graph rewrite.
6. `style_injection_live_init_probe` calibrates a new class of close-result mistakes.
   Zero-init style-injection variants can be exact no-ops, while `mixed + live_init` is runtime-real and `spatial_carrier + live_init` is real but weaker on the plain path.

## Rerun priorities

1. **Highest**: run full training for `plain_path_distill_lowrank` variants.
   This is the strongest current evidence-backed lever that explicitly tries to close the train/eval contract gap.
2. **High**: if style injection is used as a no-reference rescue direction, use `style_injection_live_init=true`.
   Otherwise a close early result can still be an exact-zero-init control rather than real negative evidence.
3. **High**: if old OT evidence is needed, trust only post-multiblock `h5/h6` reruns.
   Any pre-fix h5/h6 artifact should be treated as stale for full-body TopoGate claims.
4. **Medium**: keep repaired-base bold config sweeps as negative evidence, not as primary rescue candidates.
   They are runtime-real but too weak to justify another large sweep before a stronger architecture change.
5. **Do not rerun as science evidence**: old-base style sweeps.
   They are confounded by base repair and should be discarded rather than averaged into conclusions.

## TopoGate note

The current multiblock TopoGate audit reports `descriptor_blocks=4` and `aggregate_minus_last_mean_abs=0.005184`.
That means old h5/h6 results captured before the multiblock fix are stale if they are used to support the intended full-body TopoGate OT descriptor.

## Files

- `global_invalidators.csv`
- `family_validity_matrix.csv`
- `summary.json`
