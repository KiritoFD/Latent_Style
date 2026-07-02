# 2026-06-18 Current-State Conditioning Probe

This directory captures the current local random-init probe after a second
`tools/probe_conditioning_sensitivity.py` anatomy fix.

## Why this rerun exists

The earlier anatomy fix already restored the missing runtime
`_apply_style_feature_injection(..., site="body"/"decoder")` path.

But for the repaired lowrank family there was still one more probe fidelity bug:

1. the `code_only_no_reference` anatomy branch did not call
   `model._compute_style_code(...)`
2. it did not replay `model._structured_style_from_sidecar(...)`
3. it therefore skipped the structured-map + resolved-code-map path that the
   live runtime actually uses

That made the anatomy row overstate the code-only branch and under-report how
the structured style map participates in the repaired lowrank family.

## Current result

Using:

- config:
  `docs/experiments/2026-06-18-stage1-lowrank-rerun-audit/baseline_h1_lowrank_config.json`
- command:
  `py -3.12 tools/probe_conditioning_sensitivity.py --config ... --output-dir docs/experiments/2026-06-18-current-state-conditioning-probe/lowrank_base_current`

the corrected summary now reports:

- `conditioning_code_forward_delta = 0.0022138171`
- `anatomy_code_only_delta = 0.0022138171`
- `conditioning_spatial_forward_delta = 0.0145761650`
- `anatomy_spatial_delta = 0.0145761650`

So the anatomy trace now matches the live forward delta exactly for both the
code-only and spatial branches on this repaired lowrank base.

## Interpretation

The repaired lowrank code-only path is still real, but weaker than the stale
anatomy reading suggested:

- `code_only_no_reference.style_map_a_vs_b_mean_abs = 0.0025964859`
- `code_only_no_reference.delta_a_vs_b_mean_abs = 0.0022138171`

This means:

1. the matched-target code-only path is not dead
2. it does use a small structured style-map contribution
3. but it is still much weaker than the spatial matched-target branch

For comparison:

- `spatial_matched_target.style_map_a_vs_b_mean_abs = 1.1975014210`
- `spatial_matched_target.delta_a_vs_b_mean_abs = 0.0145761650`

## What changed in our interpretation

Do not keep using the older `anatomy_code_only_delta ~ 1e-2` reading as
evidence that the repaired lowrank code-only path is already a strong rescue.

The corrected reading is:

- code-only path: live but weak
- spatial matched-target path: clearly stronger
- old close-result clusters are still better explained by
  `train_eval_contract_gap` than by “the model never changed”
