# 616 Clean Vertical Target Probe

Date: 2026-06-16

## Scope

This packet is the clean-base frequency-domain target-geometry probe derived from `docs/616/design.md`.

Matched control:

- [phase616_cleanbase_i2sb_k085_b8a2_e24.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_cleanbase_i2sb_k085_b8a2_e24.json)

Matched candidates:

- [phase616_clean_vertical_target_source_low_k085_b8a2_e24.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_vertical_target_source_low_k085_b8a2_e24.json)
- [phase616_clean_vertical_target_wavelet_k085_b8a2_e24.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_vertical_target_wavelet_k085_b8a2_e24.json)

Everything else stays fixed:

- `solver_i2sb`
- endpoint I2SB objective
- `pure_latent_spatial` tokenizer path
- clean 616 contract
- batch/accum/eval contract

## Hypothesis

The next clean question after OT repair is not "more randomness" but "is the supervision geometry itself cleaner if the low/high split is more explicit?"

Expected interpretation:

- `source_low_target_high`
  - should reduce low-frequency drift early
  - may over-anchor if the kernel split is too soft
- `wavelet_source_low_target_high`
  - should give a sharper base/fiber separation than the 5x5 kernel split
  - should be the stronger candidate if the doc's frequency-boundary diagnosis is right

## Required White-Box Metrics

These rows must exist in the training CSV for these probes:

- `training_target_projection_active`
- `training_target_projection_mode_source_low_target_high`
- `training_target_projection_mode_wavelet_source_low_target_high`
- `training_target_projection_mode_pure_vertical_flow`
- `training_target_projection_low_anchor`
- `training_target_projection_low_drift`
- `training_target_projection_target_delta`
- `training_target_projection_high_energy_ratio`
- `base_structural_drift`
- `fiber_energy_ratio`
- `low_freq_leak`
- `target_base_shift`
- `ot_plan_entropy`
- `ot_barycentric_entropy`
- `ot_target_mass_entropy`
- `structured_style_tokenizer_spatial_svd_entropy`
- `structured_style_tokenizer_style_value_offdiag_cosine`
- `structured_style_tokenizer_translation_delta_offdiag_cosine`

Probe interpretation note:

- the runtime `base_structural_drift / fiber_energy_ratio / low_freq_leak / target_base_shift` probes are now aligned to the active split mode
- this matters for `wavelet_source_low_target_high`; without this, the training target would use wavelet while the diagnostics still measured a 5x5 kernel split
- the tokenizer rank/diversity probes are cheap enough to stay in the default training path, so the clean vertical-target packet can now distinguish "projection helped geometry" from "projection merely forced tokenizer collapse"

## Result Table Helper

After runs finish, use:

- [build_phase616_projection_probe_table.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/build_phase616_projection_probe_table.py)

This produces one compact CSV combining the latest transfer eval row and the latest training probe row for each control/candidate run.

## Launcher

- [run_phase616_clean_vertical_target_probe_round1.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase616_clean_vertical_target_probe_round1.sh)

## Decision Rule

- if either projection lowers `base_structural_drift` and `low_freq_leak` without collapsing transfer style immediately, keep it active
- if `wavelet_source_low_target_high` dominates `source_low_target_high` on both LPIPS and the projection probes, prefer wavelet for the next clean lane
- if both collapse style while barely improving drift, the split is too hard and the next follow-on should soften the low anchor instead of changing solver/tokenizer first
