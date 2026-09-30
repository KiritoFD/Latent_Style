# Stage1 exact-family audit (`h0` baseline, full `h1`-`h6` family)

This folder is the corrected full-family rerun of the stage1 config audit.

It exists because the earlier folder
`docs/experiments/2026-06-18-stage1-exact-family-audit/audit`
was invoked with only two config files (`variant_config_count = 2`), which made it
look as if only `h1_linear_fm` differed from `h0_vertical_fm`.

That was an invocation mistake, not a diff-tool limitation.

## Baseline and family

- baseline:
  `docs/experiments/2026-06-18-stage1-exact-family-audit/generated_configs/h0_vertical_fm/config.json`
- family dir:
  `docs/experiments/2026-06-18-stage1-exact-family-audit/generated_configs`

The corrected full-family command was:

```powershell
py -3.12 tools/audit_config_family.py `
  --baseline-config docs/experiments/2026-06-18-stage1-exact-family-audit/generated_configs/h0_vertical_fm/config.json `
  --variant-dir docs/experiments/2026-06-18-stage1-exact-family-audit/generated_configs `
  --output-dir docs/experiments/2026-06-18-stage1-exact-family-audit-h0full `
  --device cpu
```

## What the manifest proves

`variant_manifest.json` now correctly includes:

- `h1_linear_fm`
- `h2_euclidean_ot`
- `h3_sde_noise`
- `h4_unbalanced_ot`
- `h5_topogate_attention`
- `h6_combined_topogate`

Minimal config diffs relative to `h0`:

- `h1`: `bridge.bridge_path_mode = linear`
- `h2`: `bridge.coupling_cost_composition = appearance_only`
- `h3`: `bridge.bridge_sigma = 0.02`, `bridge.bridge_noise_schedule = exact_brownian`
- `h4`: `bridge.coupling_solver = sinkhorn_unbalanced`, `bridge.sinkhorn_unbalanced_tau_src = 0.5`
- `h5`: `appearance_plus_structure`, `topogate_attention_gw`, `coupling_structure_cost_weight = 0.4`
- `h6`: combine `h3`, `h4`, `h5`

## Training-effect result

`training_effect_probe/summary.json` shows:

- `h1` and `h3` are `bridge_only_change`
- `h2`, `h4`, `h5`, `h6` are `ot_or_target_change`
- `h5/h6` set `ot_topogate_probe_active = 1.0`

So the stage1 family is not a training-time no-op.

## Config-effect result

`config_effect_probe/summary.json` shows the benchmarked no-reference eval graph still
stays flat across the family:

- `max_vs_base_forward_mean_abs = 0.0` for every variant

This means:

1. training-time OT / bridge differences are real
2. the evaluated no-reference graph still does not directly expose those switches
3. near-tied stage1 curves should not be read as "the family did nothing"

## Runtime mirror follow-up

After this audit, `src/model.py::_attach_bridge_runtime_fields()` was corrected to
mirror `bridge_noise_schedule` into inference-time model construction, and
`tools/probe_config_effectiveness.py` was rerun into:

- `config_effect_probe_after_runtime_mirror`

That follow-up confirms the runtime model now records:

- `bridge_noise_schedule`
- `bridge_sigma`

but the stage1 family is still eval-inert on the no-reference forward graph, which
points back to the larger training-graph vs eval-graph contract mismatch described in:

- `docs/model/phase616_lancet_bridge_reproducible_audit.md`
