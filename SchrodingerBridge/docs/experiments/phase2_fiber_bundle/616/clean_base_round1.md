# 616 Clean Base

Date: 2026-06-16

## Purpose

This packet creates the canonical 616 clean base that the later OT, frequency, and tokenizer rounds should inherit from.

It exists because the current live `phase616_ot_vertical_scratch_b8a2_e24` lane is still a controlled intervention on the retained `k070 -> k085_appalign` family, not a fully purged code-path base.

## Configs

- clean base:
  [phase616_cleanbase_i2sb_k085_b8a2_e24.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_cleanbase_i2sb_k085_b8a2_e24.json)
- clean OT foundation follow-on:
  [phase616_clean_ot_vertical_k085_b8a2_e24.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_vertical_k085_b8a2_e24.json)
- launcher for the clean OT follow-on:
  [run_phase616_clean_ot_vertical_round1.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase616_clean_ot_vertical_round1.sh)
- matched clean vertical-target probes:
  [phase616_clean_vertical_target_source_low_k085_b8a2_e24.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_vertical_target_source_low_k085_b8a2_e24.json)
  [phase616_clean_vertical_target_wavelet_k085_b8a2_e24.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_vertical_target_wavelet_k085_b8a2_e24.json)
- launcher for the clean vertical-target probe packet:
  [run_phase616_clean_vertical_target_probe_round1.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase616_clean_vertical_target_probe_round1.sh)
- packet note:
  [clean_vertical_target_probe_round1.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/616/clean_vertical_target_probe_round1.md)

## Contract

`model.contract_family = phase616` activates a loader-time guardrail.

That contract currently forbids:

- output appearance alignment
- proximal runtime branches
- style-delta runtime branches
- content anchor losses
- cycle consistency
- full-eval RGB postprocess
- full-eval latent postprocess
- pre-integrate moment match
- output moment match

This is a code-level boundary, not a documentation-only convention.

## Intended Use

Use this packet for the first truly clean 616 training lanes after the current legacy-base OT control closes.

Interpretation:

- use the current live lane for evidence about `OT repair + pure_vertical_flow` on the retained legacy base
- use the clean packet for the next architecture-valid 616 experiments that must not inherit `appalign`

## Validation

Local validation completed on 2026-06-16:

- config load for the clean base and clean OT follow-on
- contract failure check for injected `output_appearance_alignment_mode=tokenizer_latent_affine`
- local synthetic forward/backward smoke for both configs via
  [smoke_experiment_config.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/smoke_experiment_config.py)
- local synthetic forward/backward smoke for the clean `source_low_target_high` and `wavelet_source_low_target_high` probes

## Probe Bundle Status

As of the late 2026-06-16 infra pass, the clean base now also enables the 616 white-box debug packet by default:

- `numeric_debug = true`
- `gpu_monitor_enabled = true`
- tokenizer rank and diversity probes:
  - `structured_style_tokenizer_spatial_svd_entropy`
  - `structured_style_tokenizer_spatial_top1_singular_ratio`
  - `structured_style_tokenizer_style_value_offdiag_cosine`
  - `structured_style_tokenizer_translation_delta_offdiag_cosine`
- OT mass / plan probes:
  - `ot_plan_entropy`
  - `ot_barycentric_entropy`
  - `ot_target_mass_entropy`

These probes were verified by local smoke on the clean OT and clean wavelet vertical-target configs before remote sync.
