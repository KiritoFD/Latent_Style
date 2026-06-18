# 2026-06-18 Lowrank Code-Map Order Audit

## Finding

In the repaired `pure_latent_spatial` family, the no-reference lowrank residual map was being decoded too early.

Before the fix, the runtime order in `src/lancet_runtime.py` was effectively:

1. adapt placeholder `style_code`
2. decode `style_code_map`
3. run structured tokenizer
4. replace `style_code` with `resolved_style_code`
5. keep using the already-decoded old `style_code_map`

That meant the residual lowrank map was built from a pre-structured code that was still style-invariant for this family.

## Why this is a real bug

For `pure_latent_spatial`, `model._compute_style_code(...)` starts from:

- zero style code
- plus time code

So before the structured tokenizer runs, the code path is not yet style-specific.

The style-specific information appears only after `PureLatentSpatialTokenizer.forward(...)` injects:

- `style_values(style_id)`
- `style_global_raw(style_id)`

into:

- `spatial_map`
- `global_full`

Therefore decoding the lowrank residual map before the structured tokenizer resolves `style_code` leaves that residual carrier effectively blind to `style_id`.

## Pre-fix probe evidence

Using the repaired lowrank base before the model patch, the corrected style-id probe showed:

- `encoded_style_code_a_vs_b_mean_abs = 0.0`
- `adapted_code_a_vs_b_mean_abs = 0.0`
- `resolved_code_a_vs_b_mean_abs = 0.023816`
- `pre_structured_style_code_map_a_vs_b_mean_abs = 0.0`
- `post_resolved_style_code_map_a_vs_b_mean_abs = 0.012525`
- `structured_style_map_a_vs_b_mean_abs = 0.006881`

This is the smoking gun.

Interpretation:

- the current live forward path was using a lowrank residual map with zero inter-style separation
- if the same lowrank decoder were fed the resolved structured code instead, it would produce a style-sensitive map stronger than the structured map itself

The forward-path consequence before the patch was:

- `max_style_map_pair_delta = 0.007225`
- `max_body_pair_delta = 0.025099`
- `max_forward_pair_delta = 0.005959`

So the repaired family was live, but a potentially important residual carrier was still wired in the wrong order.

## Fix applied

We patched:

- `src/lancet_runtime.py`
- `src/model.py`

so that when a structured tokenizer returns `resolved_style_code`, the lowrank residual map is re-decoded from that resolved code before being merged into:

- the main no-reference forward style map
- the output appearance alignment context

We also updated:

- `tools/probe_styleid_eval_path.py`

so its manual trace matches the fixed runtime path while still recording:

- pre-structured code-map delta
- post-resolved code-map delta

## Post-fix probe result

After the patch, the same repaired lowrank base now shows:

- `max_style_map_pair_delta = 0.014215`
- `max_body_pair_delta = 0.059144`
- `max_decoder_pair_delta = 0.078543`
- `max_forward_pair_delta = 0.010019`

Compared with the pre-fix values:

- style-map delta roughly doubled
- body delta more than doubled
- forward delta increased from about `0.00596` to about `0.01002`

So this was not cosmetic cleanup. The residual carrier now actually affects no-reference style separation.

## Config-effect verification

We ran `tools/probe_config_effectiveness.py` with:

- base: repaired lowrank config
- variants:
  - `disable_lowrank_code_map` via `model.style_code_spatial_scale = 0.0`
  - `boost_lowrank_code_map` via `model.style_code_spatial_scale = 0.7`

Results from `probe/summary.json`:

### disable_lowrank_code_map

- `plain forward delta = 0.042390`
- `plain predict_transport_base delta = 0.042390`
- `plain integrate delta = 0.043425`

This shows the lowrank residual is now a first-order contributor to plain eval behavior.

### boost_lowrank_code_map

- `plain forward delta = 0.000952`
- `plain predict_transport_base delta = 0.000952`
- `plain integrate delta = 0.002548`

This smaller delta suggests the current base is already in a partially saturated regime for this carrier, but the important point is that the lowrank branch is no longer a hidden no-op.

## Consequence for experiment interpretation

When several phase-618 groups looked nearly identical, at least one real reason was:

- the launcher could silently mutate the family back to legacy
- even when the repaired family was present, the lowrank residual map was still decoded from the wrong code ordering

So "the hypothesis failed" was too early a conclusion.

The implementation was still leaving meaningful style-carrying capacity on the floor.

## Files

- `variant_spec.json`
- `probe/summary.json`
- `probe/variant_effects.csv`
