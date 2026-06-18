# 2026-06-18 Current-Code Recheck: Stage1 Repaired-Lowrank Family

This artifact re-runs the minimum phase-618 diagnostic stack on the **current**
local codebase for the repaired-lowrank stage1 family (`h0/h2/h3/h4/h5/h6`).

Its job is narrow:

> When these groups stay numerically close, is that because the implementation
> never changed the model path, or because the family is mostly changing the
> training contract rather than the benchmarked plain no-reference eval graph?

## Inputs

- base config:
  `docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json`
- variant specs:
  - `config_effect_probe/variant_spec.expanded.json`
  - `training_effect_probe/variant_spec.expanded.json`

## Commands

```powershell
py -3.12 tools/probe_config_effectiveness.py `
  --config docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json `
  --variant-spec docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/config_effect_probe/variant_spec.expanded.json `
  --output-dir docs/experiments/2026-06-18-current-stage1-lowrank-recheck/config_effect_probe `
  --device cpu --batch-size 2 --latent-size 32 --style-id 1

py -3.12 tools/probe_training_variant_effect.py `
  --config docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json `
  --variant-spec docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/training_effect_probe/variant_spec.expanded.json `
  --output-dir docs/experiments/2026-06-18-current-stage1-lowrank-recheck/training_effect_probe `
  --device cpu --batch-size 2 --latent-size 32 --target-style-id 1 --source-style-id 0

py -3.12 tools/probe_conditioning_sensitivity.py `
  --config docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json `
  --output-dir docs/experiments/2026-06-18-current-stage1-lowrank-recheck/conditioning_sensitivity_probe `
  --device cpu --batch-size 2 --latent-size 32 --style-id 1
```

## Files

- `config_effect_probe/summary.json`
- `config_effect_probe/variant_effects.csv`
- `training_effect_probe/summary.json`
- `training_effect_probe/variant_training_effects.csv`
- `conditioning_sensitivity_probe/summary.json`

## Key result 1: all six stage1 OT variants are still eval-graph identical at init

From `config_effect_probe/summary.json`:

| Variant | `max_vs_base_forward_mean_abs` |
| --- | ---: |
| `h0_vertical_fm` | `0.0` |
| `h2_euclidean_ot` | `0.0` |
| `h3_sde_noise` | `0.0` |
| `h4_unbalanced_ot` | `0.0` |
| `h5_topogate_attention` | `0.0` |
| `h6_combined_topogate` | `0.0` |

Interpretation:

1. the stage1 OT / bridge overrides still do **not** alter the executed plain
   no-reference eval graph by themselves
2. this is not a stale-memory claim; it is re-verified on the current code
3. if close curves appear, that closeness cannot be explained by "the probe was
   just old"

## Key result 2: the same six variants still change the training path materially

From `training_effect_probe/summary.json`:

| Variant | Classification | `matched_target_vs_base_mean_abs` | `x_t_vs_base_mean_abs` | `pred_velocity_vs_base_mean_abs` |
| --- | --- | ---: | ---: | ---: |
| `h0_vertical_fm` | `bridge_only_change` | `0.0000` | `0.0765` | `0.0241` |
| `h2_euclidean_ot` | `ot_or_target_change` | `0.6496` | `0.1475` | `0.0501` |
| `h3_sde_noise` | `bridge_only_change` | `0.0000` | `0.0768` | `0.0242` |
| `h4_unbalanced_ot` | `ot_or_target_change` | `0.0841` | `0.0810` | `0.0258` |
| `h5_topogate_attention` | `ot_or_target_change` | `0.0373` | `0.0775` | `0.0252` |
| `h6_combined_topogate` | `ot_or_target_change` | `0.1262` | `0.0857` | `0.0282` |

Interpretation:

1. the family is **not** an implementation-wide exact no-op
2. `h0/h3` mainly change the bridge construction
3. `h2/h4/h5/h6` also change OT matching / objective targets
4. the close-results story remains `training-real, eval-inert`, not "the model
   never changed"

## Key result 3: `topogate_attention_gw` is live on the current code and still uses all body blocks

From `training_effect_probe/summary.json`:

| Variant | `ot_topogate_probe_active` | `ot_topogate_descriptor_blocks` | `ot_topogate_complexity_cost_mean` |
| --- | ---: | ---: | ---: |
| `h5_topogate_attention` | `1.0` | `4.0` | `0.0037316` |
| `h6_combined_topogate` | `1.0` | `4.0` | `0.0037316` |

Interpretation:

1. the multiblock TopoGate descriptor repair is still active on the current tree
2. old "last block only" stale evidence should remain retired
3. close outcomes from `h5/h6` are therefore better read as a weak or indirect
   family effect, not as a descriptor no-op

## Key result 4: the repaired-lowrank base itself is style-live, but the code path is still weaker than the spatial path

From `conditioning_sensitivity_probe/summary.json`:

| Conditioning mode | `forward_a_vs_b_mean_abs` |
| --- | ---: |
| `none` | `0.0000000` |
| `spatial` | `0.0145782` |
| `code` | `0.0022123` |
| `both` | `0.0155500` |

And from the path anatomy rows:

- `anatomy_code_only_delta = 0.0022123`
- `anatomy_spatial_delta = 0.0145782`

Interpretation:

1. the repaired lowrank no-reference base is not dead
2. pure code conditioning is live but weak
3. spatial matched-target conditioning is much stronger
4. this continues to support the phase-618 reading that the missing piece is
   stronger plain-path actuation, not merely "checking whether anything is on"

## Key result 5: topology blend remains a real lever only when the gate is on

From `conditioning_sensitivity_probe/summary.json`:

- `gate0_blend0` vs `gate0_blend1`:
  `forward_same_target_mean_abs = 0.0`
- `gate1_blend0` vs `gate1_blend1`:
  `forward_same_target_mean_abs = 0.0152035`
- `gate1_blend0` vs `gate1_blend05`:
  `forward_same_target_mean_abs = 0.0098075`

Interpretation:

1. `semantic_self_topology_blend` is a real runtime lever when
   `semantic_self_topology_gate=true`
2. it is an exact no-op when the gate is disabled
3. this remains a configuration-validity issue, not a reason to call the whole
   stage1 family dead

## Bottom line

This current-code recheck sharpens the phase-618 conclusion:

- the repaired-lowrank stage1 OT family is still **training-real**
- the family is still **plain-eval inert at init**
- `topogate_attention_gw` is still **implemented and active**
- the repaired lowrank carrier is still **style-live**, but mostly through the
  spatial route

So when stage1 groups end up close, the best current explanation is still:

> the training contract changed, but the benchmarked plain no-reference path
> did not become a strong style actuator

That is a theory / actuation problem, not evidence that the whole
implementation failed to touch the model at all.
