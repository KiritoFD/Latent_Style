# Stage1 Config-Effect Probe

This probe answers the missing half of the phase-616 stage1 question:

> Do the `h0`-`h6` variants change the benchmarked no-reference eval graph at all,
> or do they only change training-time OT / bridge construction?

It complements:

- `docs/experiments/2026-06-18-stage1-training-effect-probe`

That training-effect probe already showed that stage1 variants are not training-path
no-ops. This probe shows what happens on the evaluated no-reference path.

## 1. Repro

Baseline:

- `docs/experiments/2026-06-18-remote-h1-e18-diagnosis/remote_config.json`

Variants:

- `docs/experiments/2026-06-18-stage1-training-effect-probe/stage1_variant_spec.json`

Command:

```powershell
py -3.12 tools/probe_config_effectiveness.py `
  --config docs/experiments/2026-06-18-remote-h1-e18-diagnosis/remote_config.json `
  --variant-spec docs/experiments/2026-06-18-stage1-training-effect-probe/stage1_variant_spec.json `
  --output-dir docs/experiments/2026-06-18-stage1-config-effect-probe/probe_random_init `
  --device cpu
```

Outputs:

- `probe_random_init/variant_effects.csv`
- `probe_random_init/summary.json`

## 2. Main finding

Relative to `h1_linear_fm`, every tested stage1 variant is a no-reference eval no-op
at random init:

```text
h0_vertical_fm        -> no_effect
h2_euclidean_ot       -> no_effect
h3_sde_noise          -> no_effect
h4_unbalanced_ot      -> no_effect
h5_topogate_attention -> no_effect
h6_combined_topogate  -> no_effect
```

Observed for all of them:

- `plain vs_base_forward_mean_abs -> 0.0`
- `configured vs_base_forward_mean_abs -> 0.0`
- `spatial vs_base_forward_mean_abs -> 0.0`
- `code vs_base_forward_mean_abs -> 0.0`

And the baseline anatomy remains:

- `anatomy_code_body_dead_spatial_body_live -> true`

Meaning:

1. these stage1 variants do not alter the executed no-reference graph that the
   transfer benchmark actually evaluates
2. the benchmarked `h1` contract still has a body-dead code-only path and a
   body-live matched-target spatial path
3. therefore near-tied stage1 eval curves are expected, even when the training
   target and bridge construction really differ

## 3. Combined diagnosis with the training-effect probe

Together, the two probes now say:

- `probe_training_variant_effect.py`
  - stage1 variants are real at training time
- `probe_config_effectiveness.py`
  - stage1 variants are inert on the benchmarked no-reference eval path

So the precise diagnosis is:

> Stage1 `h0`-`h6` is not an implementation no-op at training time, but it is a
> practical no-op for the current no-reference evaluation contract.

That is the concrete reason why stage1 groups can differ less than metric noise:

- OT / bridge machinery changes training
- but the evaluated graph never sees those differences

## 4. Bottom line

For phase-618 decision-making:

1. do not interpret close `h0`-`h6` eval results as evidence that the OT / bridge
   implementation is dead everywhere
2. do interpret them as evidence that the current stage1 family does not move the
   no-reference eval graph
3. promote only variants that change `plain` eval, such as the low-rank code-map
   repairs or later bold-direction architecture changes
