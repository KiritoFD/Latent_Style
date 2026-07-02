# Stage1 Training-Effect Probe

This folder answers a narrower question than the model/eval-path probes:

> Do the phase-616 stage1 OT / bridge variants actually change the training-time
> matching and bridge construction path on the same fixed batch?

It does **not** ask whether they change the no-reference eval graph. That question is covered by:

- `docs/experiments/2026-06-18-remote-h1-e18-diagnosis/config_effect_probe`

This probe instead compares:

- `matched_target`
- `objective_target`
- `x_t`
- `target_velocity`
- `pred_velocity`
- selected OT / bridge debug metrics

under the same:

- baseline config
- model weights
- synthetic batch
- bridge random seed

## 1. Repro

Baseline:

- `docs/experiments/2026-06-18-remote-h1-e18-diagnosis/remote_config.json`
- this is the pulled `h1_linear_fm` config

Variants:

- `stage1_variant_spec.json`

Command:

```powershell
py -3.12 tools/probe_training_variant_effect.py `
  --config docs/experiments/2026-06-18-remote-h1-e18-diagnosis/remote_config.json `
  --variant-spec docs/experiments/2026-06-18-stage1-training-effect-probe/stage1_variant_spec.json `
  --output-dir docs/experiments/2026-06-18-stage1-training-effect-probe/probe_random_init `
  --device cpu
```

Outputs:

- `probe_random_init/variant_training_effects.csv`
- `probe_random_init/summary.json`

## 2. Main finding

The stage1 variants are **not** all training-path no-ops.

They split into two families:

### 2.1 Bridge-only changes

- `h0_vertical_fm`
- `h3_sde_noise`

Observed:

- `matched_target_vs_base_mean_abs -> 0.0`
- `objective_target_vs_base_mean_abs -> 0.0`
- but:
  - `x_t_vs_base_mean_abs -> ~0.086`
  - `target_velocity_vs_base_mean_abs -> ~0.305`

Meaning:

- these variants keep the same OT match as `h1`
- but they change bridge construction itself:
  - path geometry for `h0`
  - path geometry plus stochastic bridge noise for `h3`

So `h0` and `h3` are real training differences, but they are **bridge-only** differences rather than OT-matching differences.

### 2.2 OT / target-changing variants

- `h2_euclidean_ot`
- `h4_unbalanced_ot`
- `h5_topogate_attention`
- `h6_combined_topogate`

Observed:

- `h2_euclidean_ot`
  - `matched_target_vs_base_mean_abs -> 0.45671725273132324`
  - `x_t_vs_base_mean_abs -> 0.21280144155025482`
- `h4_unbalanced_ot`
  - `matched_target_vs_base_mean_abs -> 0.06527001410722733`
  - `ot_target_gini delta -> +0.039943695068359375`
- `h5_topogate_attention`
  - `matched_target_vs_base_mean_abs -> 0.4015389680862427`
  - `ot_topogate_probe_active -> 1.0`
  - `ot_structure_cost_mean -> 1.0`
  - `ot_latent_affinity_cost_mean -> 5.336275100708008`
- `h6_combined_topogate`
  - `matched_target_vs_base_mean_abs -> 0.41072985529899597`
  - `ot_topogate_probe_active -> 1.0`
  - `ot_dummy_mass -> 0.0`
  - `bridge_sigma -> 0.02`

Meaning:

- these variants really do change the OT plan and therefore the matched training target
- `h4` changes the transport plan through the unbalanced solver
- `h5` changes the structural descriptor from latent self-affinity to `topogate_attention_gw`
- `h6` combines both OT-side and bridge-side changes

Important audit note added later on the repaired low-rank base:

- for `topogate_attention_gw`, `ot_structure_cost_mean = 1.0` is **not** evidence
  that the structure cost collapsed to a constant matrix
- it is a contract artifact of the blended topogate path, where the returned
  structure cost is already mean-normalized before the outer OT composition step
- the more diagnostic fields are:
  - `ot_topogate_complexity_term_var`
  - `ot_latent_affinity_term_var`
  - `ot_total_cost_matrix_var`

On the repaired low-rank rerun base these become:

- `h5_topogate_attention`
  - `ot_topogate_complexity_term_var -> 0.508704662322998`
  - `ot_latent_affinity_term_var -> 0.05051263049244881`
  - `ot_total_cost_matrix_var -> 0.051355328410863876`

Meaning:

- the TopoGate complexity branch is not degenerate
- after normalization, it is actually **more contrastive** than the latent-affinity
  branch
- the old reading "`ot_structure_cost_mean = 1.0`, therefore h5 did nothing" is wrong

So the stage1 OT hypotheses are not merely bookkeeping differences. They are materially distinct at training time.

## 3. What this rules out

This probe rules out one specific failure story:

> "Stage1 groups are close because the OT / bridge configs never changed the training path at all."

That story is too weak now.

More precise status:

1. Stage1 variants do diverge in training-time matching and/or bridge construction.
2. The earlier config-effect audit already showed that some families of changes still fail to alter the no-reference eval graph.
3. Therefore metric closeness can no longer be blamed on a single universal no-op bug.

The current evidence points to a more specific picture:

- some differences are real only in the training graph
- some differences now also reach the eval graph once a no-reference spatial carrier exists
- remaining near-ties should be interpreted as either:
  - train/eval contract mismatch
  - weak downstream style actuation
  - or genuine theory weakness

## 4. Bottom line

Phase-618 now has two complementary audits:

1. `probe_config_effectiveness.py`
   - asks whether a config delta changes the benchmarked no-reference eval graph
2. `probe_training_variant_effect.py`
   - asks whether a variant changes the training-time OT / bridge target construction

Use both before trusting "all groups are the same" as a scientific conclusion.
