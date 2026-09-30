# 2026-06-18 Style-Injection Live-Init Probe

This probe asks a narrow but important phase-618 question:

> If we enable a new no-reference style-injection branch and the result stays very close to baseline, is that theory weakness, or did the branch start as an exact zero/near-zero path?

The target base is the repaired lowrank no-reference carrier:

- [baseline_h1_lowrank_config.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json)

The tested variants are in:

- [variant_spec.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-style-injection-live-init-probe/variant_spec.json)

Probe outputs:

- [summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-style-injection-live-init-probe/probe/summary.json)
- [variant_effects.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-style-injection-live-init-probe/probe/variant_effects.csv)

## Why this exists

`docs/618/why_style_weak.md` already notes that many style-side modules start effectively asleep.

Before this patch, all three style-injection builders in `src/model.py` ended with exact zero-init output layers:

- `mixed`
- `carrier_gate`
- `spatial_carrier_gate`

That preserves parent behavior, but it also means a "turn on style injection" config sweep can look unchanged at random init even when the graph shape has changed.

So we added:

- `model.style_injection_live_init`
- `model.style_injection_live_init_std`

These keep default behavior unchanged, but allow future actuation experiments to start from a small live style branch instead of an exact zero path.

## Probe instrumentation fix

During the phase-618 audit we found a second implementation problem in the diagnostic tool itself:

- `tools/probe_conditioning_sensitivity.py` manually traced the body/decoder path
- but it had omitted the runtime calls to `_apply_style_feature_injection(..., site="body")`
- and `_apply_style_feature_injection(..., site="decoder")`

That meant the config-effect probe could correctly see a changed `forward()` path, while the anatomy rows still under-reported the new branch.

This is now fixed, with regression coverage in:

- [test_infra_guardrails.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tests/test_infra_guardrails.py)

So the anatomy numbers below are post-fix evidence, not the stale pre-fix trace.

## Command

```powershell
py -3.12 tools/probe_config_effectiveness.py `
  --config docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json `
  --variant-spec docs/experiments/2026-06-18-style-injection-live-init-probe/variant_spec.json `
  --output-dir docs/experiments/2026-06-18-style-injection-live-init-probe/probe `
  --device cpu
```

## Key results

### 1. Zero-init style injection variants are exact no-ops relative to baseline

For:

- `z1_body_mixed_zero_init`
- `z3_body_spatial_carrier_zero_init`

`variant_effects.csv` shows:

- `plain vs_base_forward_mean_abs = 0.0`
- `configured vs_base_forward_mean_abs = 0.0`
- `spatial vs_base_forward_mean_abs = 0.0`
- `code vs_base_forward_mean_abs = 0.0`

Meaning:

- the new modules exist
- but at random init they preserve the parent graph exactly
- so a close result from these zero-init style-injection variants is not useful evidence against the mechanism

### 2. `mixed` live-init wakes the branch strongly enough to move the plain no-reference graph

For `z2_body_mixed_live_init`:

- `plain vs_base_forward_mean_abs = 0.007001949`
- `configured vs_base_forward_mean_abs = 0.006194224`
- `spatial vs_base_forward_mean_abs = 0.006432245`
- `code vs_base_forward_mean_abs = 0.006471330`

Interpretation:

- a small live-init is enough to make body-level mixed style injection visible on the executed no-reference path
- this is much larger than the `1e-3`-scale blend-only bold sweeps
- so if we want to test "stronger no-reference style actuation" fairly, zero-init is too conservative for this branch

Post-fix anatomy also now shows the incremental effect instead of hiding it:

- `code_only_no_reference h_body_a_vs_b_mean_abs` increases from `0.062431030` to `0.062521532`
- `h_dec_pre_mod_a_vs_b_mean_abs` increases from `0.110204324` to `0.110686623`
- `delta_a_vs_b_mean_abs` increases from `0.012378661` to `0.012438562`

Important nuance:

- the repaired lowrank baseline is already style-live on the code path because the lowrank residual code-map path is active
- so the correct question is not "did h_body go from zero to non-zero?"
- it is "did the variant stay exactly identical to baseline, or did the new branch add measurable incremental signal?"

For `z1_body_mixed_zero_init`, the answer is still "identical to baseline".
For `z2_body_mixed_live_init`, the answer is now clearly "incrementally different from baseline".

### 3. `spatial_carrier_gate` live-init is real, but much weaker on the plain path

For `z4_body_spatial_carrier_live_init`:

- `plain vs_base_forward_mean_abs = 0.000065156`
- `configured vs_base_forward_mean_abs = 0.003182278`
- `spatial vs_base_forward_mean_abs = 0.003185746`
- `code vs_base_forward_mean_abs = 0.000062863`

Interpretation:

- this branch is no longer a hard no-op once live-init is enabled
- but on the current repaired base it is still much weaker than `mixed` for plain no-reference actuation
- so if this path underperforms later, it is less likely to be because it never woke up, and more likely to be because its leverage is genuinely smaller

The post-fix anatomy trace says where it moves:

- code-only path changes are tiny: `h_body +0.000002176`, `delta +0.000001042`
- spatial matched-target path changes are much clearer: `h_body +0.003744096`, `delta +0.000253752`

So `spatial_carrier_gate + live_init` is real, but its leverage is concentrated in the spatial/matched-target route rather than the plain no-reference route.

## What this means for phase 618

This probe gives a concrete new rule for reading close results:

1. If a style-injection variant used the old exact-zero init, close results do **not** count as meaningful negative evidence.
2. For future bold-direction runs that rely on `style_injection_mode`, use `style_injection_live_init=true` unless the explicit goal is to study slow wake-up dynamics.
3. Among the existing no-reference style-injection forms, `mixed + live-init` is currently the strongest candidate for a fair next-stage actuation experiment.

In short:

> yes, there was still an implementation-side reason some "new style branch" experiments could look unchanged even before training really began.
