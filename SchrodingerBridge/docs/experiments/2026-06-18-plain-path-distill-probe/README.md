# 2026-06-18 Plain-Path Distill Probe

This probe checks whether the new `bridge.w_plain_path_distill` path is actually live on the sampled-bridge training graph, and whether our debug/probe stack can see it.

## Why this exists

`docs/618/why_style_weak.md` and `docs/618/bold_directions.md` point to a contract gap:

- training already uses matched-target conditioning
- plain no-reference eval does not
- if we only strengthen the conditioned branch, style can improve in training while plain eval stays flat

The new lever is:

- `bridge.w_plain_path_distill`

It distills the conditioned branch back into the plain no-reference branch:

- OMF path: plain `predict_transport_base(...)` matches conditioned `predict_transport_base(...)`
- sampled bridge path: plain `model(...)` or plain `predict_transport_base(...)` matches the conditioned teacher

## Files

- Variant spec:
  - [variant_spec.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-plain-path-distill-probe/variant_spec.json)
- Probe outputs:
  - [summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-plain-path-distill-probe/training_effect_probe/summary.json)
  - [variant_training_effects.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-18-plain-path-distill-probe/training_effect_probe/variant_training_effects.csv)

## Command

```bash
py -3.12 tools/probe_training_variant_effect.py \
  --config docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json \
  --variant-spec docs/experiments/2026-06-18-plain-path-distill-probe/variant_spec.json \
  --output-dir docs/experiments/2026-06-18-plain-path-distill-probe/training_effect_probe \
  --device cpu \
  --batch-size 2 \
  --latent-size 16
```

## Result

Variant:

- `plain_path_distill_w0p50`

Key evidence from `summary.json`:

- baseline:
  - `plain_path_distill = 0.0`
  - `plain_path_distill_active = 0.0`
- variant:
  - `plain_path_distill = 0.0022437442`
  - `plain_path_distill_active = 1.0`
  - `plain_path_student_abs = 0.1816262007`
  - `classification = conditioning_or_loss_change`

So the new branch is not a no-op.

## Important audit note

The first version of `tools/probe_training_variant_effect.py` only classified training-path changes from a narrow subset of components (`flow` and `terminal_swd`). That incorrectly labeled this variant as `no_training_effect` even though the new loss was active.

The probe has now been widened to compare the full component dictionary, so new losses such as `plain_path_distill` are no longer invisible to the audit.
