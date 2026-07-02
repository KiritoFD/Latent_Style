# Config Family Audit Smoke

This note validates the new reusable audit wrapper:

- `tools/audit_config_family.py`

It answers a recurring phase-618 debugging question:

> When two runs look close, are we sure the *actual generated configs* still differ in
> the executed eval graph or training graph under the current repaired base?

## 1. Repro

Baseline:

- `docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/remote_base_phase618_ot_rerun_lowrank.json`

Variant:

- `docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json`

Audit command:

```powershell
py -3.12 tools/audit_config_family.py `
  --baseline-config docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/remote_base_phase618_ot_rerun_lowrank.json `
  --variant-config docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json `
  --output-dir docs/experiments/2026-06-18-config-family-audit-smoke `
  --device cpu
```

Generated artifacts:

- `variant_spec.json`
- `variant_manifest.json`
- `config_effect_probe/*`
- `training_effect_probe/*`
- `summary.json`

## 2. What the wrapper extracted

It automatically diffed the two real config files and kept only these effective overrides:

- `bridge.bridge_path_mode = "linear"`
- `bridge.coupling_cost_composition = "structure_only"`
- `bridge.coupling_structure_cost_mode = "self_affinity_gw"`
- `bridge.coupling_structure_cost_weight = 1.0`
- `bridge.sinkhorn_unbalanced_tau_src = 1.0`
- `bridge.sinkhorn_unbalanced_tau_tgt = 1.0`

Importantly, it excluded run-local clutter such as:

- `checkpoint.save_dir`
- `ablation.*`
- `training.resume_*`

So this is the exact kind of reusable family audit we wanted: compare real configs, not
hand-written memory of what a family was *supposed* to vary.

## 3. Main finding

For this specific repaired-base comparison, the wrapper found:

- config-effect:
  - `max_vs_base_forward_mean_abs = 0.0`
- training-effect:
  - `classification = "no_training_effect"`
  - `matched_target_vs_base_mean_abs = 0.0`
  - `x_t_vs_base_mean_abs = 0.0`
  - `target_velocity_vs_base_mean_abs = 0.0`
  - `pred_velocity_vs_base_mean_abs = 0.0`

Meaning:

1. the nominal `h0 -> h1` difference we expected from older stage1 reasoning does **not**
   automatically survive under every repaired-base comparison
2. some "family differences" can collapse all the way to no-op when audited against the
   *current exact base config*
3. this is exactly why close results must be checked against real generated configs and
   real probes, not just historical variant names

## 4. Why this matters

This does **not** prove the whole old-OT family is dead again.

It proves something narrower but very important:

- exact-run config audits are worth automating
- historical assumptions about `h0`, `h1`, etc. can go stale after base repairs
- when corrected reruns later look close, we now have a direct tool to verify whether the
  current family members still differ in:
  - plain no-reference eval graph
  - training-side bridge / OT graph
  - or neither

That makes `tools/audit_config_family.py` a new first-stop diagnostic before drawing
theoretical conclusions from tight experiment clusters.
