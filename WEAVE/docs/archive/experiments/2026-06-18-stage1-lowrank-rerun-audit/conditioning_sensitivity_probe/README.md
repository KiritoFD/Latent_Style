# Conditioning Sensitivity Probe

This probe is the authority for one narrow but important question on the repaired
low-rank base:

> When no matched target latent is present, is the no-reference style path really
> dead, or does the evaluated graph still execute a structured style map plus a
> low-rank residual code map?

## Repro

```powershell
py -3.12 tools/probe_conditioning_sensitivity.py `
  --config docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json `
  --output-dir docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/conditioning_sensitivity_probe `
  --device cpu
```

Outputs:

- `conditioning_sensitivity.csv`
- `topology_sensitivity.csv`
- `topology_pairwise.csv`
- `path_anatomy.csv`
- `summary.json`

## Probe-contract fix

Before commit `6f224c69a4c42a63874856c72b72ed73a4e4a239`, the runtime already emitted:

- `style_spatial_source_structured_map`
- `style_spatial_source_override_palette`
- `style_spatial_code_map_pre_resolved_abs`

but `tools/probe_conditioning_sensitivity.py` did not export them.

That omission made the `mode=none` and `mode=code` rows look contradictory:

- `style_spatial_source_code_map = 0`
- `style_spatial_source_legacy_zero = 0`
- `style_spatial_code_map_primary = 0`
- `style_spatial_code_map_residual = 1`

After the probe export fix, the source semantics are explicit.

## Main result

On the repaired low-rank base:

- `mode=none`
  - `style_spatial_source_structured_map = 1`
  - `style_spatial_code_map_residual = 1`
  - `forward_a_vs_b_mean_abs = 0`

- `mode=code`
  - `style_spatial_source_structured_map = 1`
  - `style_spatial_code_map_residual = 1`
  - `forward_a_vs_b_mean_abs = 0.0022122871596366167`

- `mode=spatial`
  - `style_spatial_source_target_latent = 1`
  - `style_spatial_code_map_residual = 1`
  - `forward_a_vs_b_mean_abs = 0.014578190632164478`

So the repaired no-reference path is not:

1. `legacy_zero`
2. `code_map` primary fallback
3. a probe artifact

It is a real `structured_map + residual code_map` path, and its code-only actuation is
small but non-zero.

## Quantitative anatomy

Key summary numbers:

- `conditioning_code_forward_delta = 0.0022122871596366167`
- `conditioning_spatial_forward_delta = 0.014578190632164478`
- `conditioning_both_forward_delta = 0.015550028532743454`
- `anatomy_code_first_live_stage = first_hires_block_gate1_a_vs_b_mean_abs`
- `anatomy_code_first_live_stage_delta = 0.0018754458287730813`
- `anatomy_code_only_delta = 0.012378660961985588`
- `anatomy_spatial_delta = 0.02682328224182129`

Interpretation:

1. the old body-dead bug is not present on this repaired base
2. no-reference style actuation is real, but weaker than matched-target spatial input
3. if rerun curves still cluster tightly, that cannot be blamed on the old
   "plain eval path is dead" story

## Consequence for phase-618

This probe closes one observability hole:

- the runtime source labels were correct
- the probe export was incomplete
- the repaired low-rank base remains valid for rerunning old OT variants and for
  testing bold-direction changes

So the next debugging question is no longer "did the no-reference path disappear?"
but "do the chosen training-time objectives move the evaluated no-reference path
enough to matter on CLIP-S / LPIPS?"
