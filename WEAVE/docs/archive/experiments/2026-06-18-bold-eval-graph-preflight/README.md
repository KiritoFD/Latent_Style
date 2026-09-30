# Bold eval-graph preflight

This folder checks the highest-priority config-only ideas from:

- `docs/618/why_style_weak.md`
- `docs/618/bold_directions.md`

against the repaired lowrank no-reference base:

- `docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json`

The goal was not "does the config differ on paper", but:

1. does it move the plain no-reference eval graph?
2. does it increase style-id separation in the body path?
3. is the change runtime-only, training-only, or both?

## Variants tested

`variant_spec.json` contains:

- `r11_linear_blend_0p00`
- `r12_linear_blend_0p30`
- `r13_linear_blend_0p00_solver_pc`
- `r14_linear_blend_0p30_solver_pc`
- `r15_vertical_blend_0p00`
- `r16_vertical_blend_0p00_solver_pc`

These correspond to the bold-direction idea family:

- lower / remove content-topology blend
- optionally switch to `solver_pc`
- optionally test vertical path

## Key probe results

### 1. Lowering blend does change the plain eval graph, but only weakly

From `config_effect_probe/summary.json`:

- `r11_linear_blend_0p00`
  - `plain vs_base_forward_mean_abs = 0.000960846`
- `r12_linear_blend_0p30`
  - `plain vs_base_forward_mean_abs = 0.000735832`

So `blend=0.0` and `blend=0.3` are not no-ops.
But the plain no-reference graph only moves at about `1e-3`.

That is much smaller than the matched-target / spatial-context deltas in the same
probe, and too small to expect a "脱胎换骨" result by itself.

### 2. Lowering blend does not materially increase no-reference style separation

Baseline repaired lowrank body-live probe from
`docs/experiments/2026-06-18-styleid-eval-probe/lowrank_base/summary.json`:

- `max_forward_pair_delta = 0.010018897`
- `max_body_pair_delta = 0.059144132`
- `max_decoder_pair_delta = 0.078542545`

Bold config-only variants from `styleid_probes/*/summary.json`:

- `r11_linear_blend_0p00`
  - `max_forward_pair_delta = 0.010065609`
  - `max_body_pair_delta = 0.060304016`
- `r12_linear_blend_0p30`
  - `max_forward_pair_delta = 0.010026167`
  - `max_body_pair_delta = 0.059643842`

Interpretation:

- the style carrier remains body-live
- but `blend=0.0` / `0.3` gives only marginal uplift over the already repaired base
- direction 1 / direction 4 as config-only sweeps are too weak to be the main rescue

### 3. `solver_pc` is real, but mostly in the integrate branch

Before this preflight, `tools/probe_styleid_eval_path.py` only summarized:

- forward pair delta
- body pair delta
- decoder pair delta

That missed solver-only behavior.

We patched the tool so it now also reports:

- `max_predict_transport_base_pair_delta`
- `max_integrate_pair_delta`

Config-effect evidence:

- `r13_linear_blend_0p00_solver_pc`
  - `plain vs_base_forward_mean_abs = 0.000960846`
  - `plain vs_base_integrate_mean_abs = 0.013168968`
- `r14_linear_blend_0p30_solver_pc`
  - `plain vs_base_forward_mean_abs = 0.000735832`
  - `plain vs_base_integrate_mean_abs = 0.013163378`

So `solver_pc` is not a no-op.
It meaningfully changes `integrate()`, while leaving `forward()` /
`predict_transport_base()` almost unchanged.

But style-id separation still does not improve:

- `r11` integrate pair delta: `0.009810952`
- `r13` integrate pair delta: `0.009274727`
- `r12` integrate pair delta: `0.009765103`
- `r14` integrate pair delta: `0.009229702`

Meaning:

1. `solver_pc` changes the evaluated integration path
2. but it does not increase style separation in the current no-reference carrier

### 4. Vertical path changes training, not the plain no-reference graph

Training-effect evidence:

- `r15_vertical_blend_0p00`
  - classification: `bridge_only_change`
  - `x_t_vs_base_mean_abs = 0.086381860`
- `r16_vertical_blend_0p00_solver_pc`
  - classification: `bridge_only_change`
  - `x_t_vs_base_mean_abs = 0.086381860`

But plain no-reference style-id deltas are effectively identical to the linear
blend-0 variants.

So vertical-vs-linear still behaves like a train-graph distinction much more than
an eval-graph style-actuation distinction.

## Conclusion

This preflight rules out a tempting but misleading conclusion:

- "just lower blend and switch to PC solver, style will jump"

What the evidence actually says is:

1. `blend=0.0` / `0.3` are real runtime changes, but weak in plain no-reference eval
2. `solver_pc` is real in `integrate()`, but does not improve style separation
3. vertical path still mostly changes the training bridge contract, not the plain
   no-reference style carrier

Therefore the next serious investment should not be more blend-only sweeps.
It should target the no-reference style carrier itself.

## Important architectural note about direction 5

The codebase already contains a training-time matched-target style encoder:

- `src/lancet_runtime.py::encode_target_style_latent(...)`
- active when `matched_target_style_encoder_mode = residual`

And the current repaired lowrank base already uses:

- `matched_target_conditioning_mode = both`
- `matched_target_style_encoder_mode = residual`

So "从 matched_target 学风格" is already partially implemented for the training graph.

The missing piece is:

- how to transfer that instance-level style information back into the plain
  no-reference eval path

That is a stronger candidate for the next paradigm change than simply re-adding
another matched-target encoder.
