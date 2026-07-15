# Phase-618 Validity Auditor Smoke

This directory records a minimal reproducible smoke test for:

- `tools/audit_phase618_run_validity.py`

The goal is to turn the recurring phase-618 diagnosis question into a reusable, file-backed verdict:

> When two experiment groups land almost on top of each other, is that because the theory is weak, or because the implementation / evidence path did not actually change the model in the way we think?

## 1. Repro commands

Confounded old-base style sweep:

```powershell
py -3.12 tools/audit_phase618_run_validity.py `
  --config docs/experiments/2026-06-18-remote-h1-e18-diagnosis/remote_config.json `
  --variant-spec docs/experiments/2026-06-18-style-sweep-base-audit/style_sweep_variant_spec.json `
  --variant-name r8_linear_code_map_lowrank_both `
  --output docs/experiments/2026-06-18-phase618-validity-auditor-smoke/r8_old_base_confounded.json
```

Training-only-by-design plain-path distill:

```powershell
py -3.12 tools/audit_phase618_run_validity.py `
  --config docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json `
  --variant-spec docs/experiments/2026-06-18-stage1-lowrank-distill-contract-probe/variant_spec.json `
  --variant-name h1_plain_path_distill_0p50 `
  --output docs/experiments/2026-06-18-phase618-validity-auditor-smoke/h1_plain_path_distill_0p50.json
```

Runtime-real but weak repaired bold direction:

```powershell
py -3.12 tools/audit_phase618_run_validity.py `
  --config docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json `
  --variant-spec docs/experiments/2026-06-18-bold-eval-graph-preflight/variant_spec.json `
  --variant-name r11_linear_blend_0p00 `
  --output docs/experiments/2026-06-18-phase618-validity-auditor-smoke/r11_linear_blend_0p00.json
```

## 2. Expected verdicts

### `r8_old_base_confounded.json`

- `artifact_status = "confounded"`
- `suite = "stage3_style_r1_r10_old_base"`
- `effect_contract = "runtime_real"`

Interpretation:

- the config really does move the plain eval graph
- but the family is scientifically unusable for style conclusions, because the run mixes base repair with the bold-direction override

### `h1_plain_path_distill_0p50.json`

- `artifact_status = "valid"`
- `suite = "plain_path_distill_lowrank"`
- `effect_contract = "training_only_by_design"`

Interpretation:

- this is not a bug and not a no-op
- the whole point of the knob is to change training behavior while leaving the plain no-reference eval graph unchanged at init

### `r11_linear_blend_0p00.json`

- `artifact_status = "valid"`
- `suite = "bold_r11_r16_repaired_lowrank"`
- `effect_contract = "runtime_and_training_real"`

Interpretation:

- the implementation is alive in both runtime and training probes
- but the family matrix still reads the lever as weak, so close metrics here are real negative evidence against config-only rescue

## 3. Why this matters

These three cases separate the three most important "close result" readings:

1. `confounded`
   - do not use the metric tie as evidence
2. `training_only_by_design`
   - do not expect plain-eval probes to light up
3. `runtime_and_training_real`
   - if metrics are still close, the weakness is much more likely to be theoretical than an implementation no-op

That is the exact decision chain we need before spending more GPU time or declaring a direction dead.
