# 2026-06-18 Stage1 Lowrank Distill Contract Probe

This probe asks a targeted follow-up question after the repaired lowrank rerun audit:

> If the old `h0`-`h6` OT family is training-real but plain-eval inert, can we add a
> loss that explicitly distills the conditioned branch back into the plain
> no-reference branch?

The lever under test is:

- `bridge.w_plain_path_distill = 0.5`

This is not intended to change the runtime graph at initialization.
It is intended to change the **training contract** so that the plain branch learns
from the conditioned teacher instead of remaining decoupled.

## Repro

Variant spec:

- `variant_spec.json`

Command:

```powershell
py -3.12 tools/probe_training_variant_effect.py `
  --config docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json `
  --variant-spec docs/experiments/2026-06-18-stage1-lowrank-distill-contract-probe/variant_spec.json `
  --output-dir docs/experiments/2026-06-18-stage1-lowrank-distill-contract-probe/probe `
  --device cpu
```

Outputs:

- `probe/summary.json`
- `probe/variant_training_effects.csv`
- `probe/variant_spec.expanded.json`

Runtime graph check:

```powershell
py -3.12 tools/probe_config_effectiveness.py `
  --config docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json `
  --variant-spec docs/experiments/2026-06-18-stage1-lowrank-distill-contract-probe/variant_spec.json `
  --output-dir docs/experiments/2026-06-18-stage1-lowrank-distill-contract-probe/config_effect_probe `
  --device cpu
```

Additional outputs:

- `config_effect_probe/summary.json`
- `config_effect_probe/variant_effects.csv`

## Main finding

The distill loss is live for every tested family member.

Baseline repaired lowrank `h1`:

- `plain_path_distill = 0.0`
- `plain_path_distill_active = 0.0`

All distill variants:

- `plain_path_distill_active = 1.0`
- `plain_path_distill ~= 0.00215 - 0.00224`

Examples:

- `h1_plain_path_distill_0p50`
  - classification: `conditioning_or_loss_change`
  - `component_delta::plain_path_distill = 0.00224374420940876`
  - `matched_target_vs_base_mean_abs = 0.0`

- `h0_vertical_fm_plain_path_distill_0p50`
  - classification: `bridge_only_change`
  - `x_t_vs_base_mean_abs = 0.07581581920385361`
  - `component_delta::plain_path_distill = 0.0021686286199837923`

- `h2_euclidean_ot_plain_path_distill_0p50`
  - classification: `ot_or_target_change`
  - `matched_target_vs_base_mean_abs = 0.11387591063976288`
  - `component_delta::plain_path_distill = 0.0021519986912608147`

- `h5_topogate_attention_plain_path_distill_0p50`
  - classification: `ot_or_target_change`
  - `matched_target_vs_base_mean_abs = 0.053448665887117386`
  - `component_delta::plain_path_distill = 0.002184363780543208`

Meaning:

1. the distill loss itself is not a no-op
2. it composes cleanly with both:
   - bridge-only variants (`h0`, `h3`)
   - OT / target-changing variants (`h2`, `h4`, `h5`, `h6`)
3. it is currently the clearest training-side mechanism aimed directly at the
   train/eval contract gap highlighted in `docs/618/why_style_weak.md`

## Runtime contract check

The config-effect probe shows that every distill variant is runtime-inert at random
initialization:

- `plain vs_base_forward_mean_abs = 0.0`
- `configured vs_base_forward_mean_abs = 0.0`
- `spatial vs_base_forward_mean_abs = 0.0`
- `code vs_base_forward_mean_abs = 0.0`

for:

- `h1_plain_path_distill_0p50`
- `h0_vertical_fm_plain_path_distill_0p50`
- `h2_euclidean_ot_plain_path_distill_0p50`
- `h3_sde_noise_plain_path_distill_0p50`
- `h4_unbalanced_ot_plain_path_distill_0p50`
- `h5_topogate_attention_plain_path_distill_0p50`
- `h6_combined_topogate_plain_path_distill_0p50`

Interpretation:

1. `plain_path_distill` does not accidentally rewire the plain forward graph
2. it is a deliberate **training-contract-only** lever
3. if a future rerun improves metrics, that improvement will reflect learned transfer
   from the conditioned branch rather than a hidden runtime config change

## Important scope note

This probe is **training-path only**.
And the paired config-effect probe shows the runtime graph is unchanged at init.

Together, that means we know exactly what class of lever this is:

- not a runtime architecture change
- not an accidental no-op
- a training-side distillation contract change

So the right reading is:

- "the new contract lever is implemented and live"

not:

- "the metrics are already fixed"

That second claim still requires full training reruns.

## Consequence

Compared with the earlier bold config-only sweeps:

- lowering blend / changing solver produced weak runtime deltas
- `plain_path_distill` directly attacks the mechanism that kept the plain branch
  disconnected from the conditioned branch

So if we are choosing a next experiment because old groups were too close, this is a
more aligned next-stage lever than another small blend sweep.
