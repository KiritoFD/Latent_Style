# Phase 2 Queue Manifest Note

Date: 2026-06-13

## Purpose

- provide one machine-readable phase-2 queue index
- avoid re-deriving the preferred packet order from multiple dated notes
- keep the current formal lane, structure-side reentry candidates, and diagnostic-only I2SB candidates in one place

## Manifest

- CSV:
  - [phase2_queue_manifest.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_queue_manifest.csv)

## Interpretation

- `lane_class`
  - `formal_lane`
  - `structure_reentry`
  - `i2sb_diagnostic_only`
- `priority_in_class`
  - lower number means earlier preferred execution within that class
- `preferred`
  - marks the packet that should be chosen first when the class becomes active
- `formal_eligible`
  - whether the packet is allowed to occupy the only remote formal lane under the current `612-phase2` policy
- `tokenizer_profile`
  - distinguishes refreshed `tok32` packets from older `64/4` tokenizer diagnostics

## Current Read

- current formal packet:
  - `vel_tok32_safe_rescan_r2`
  - still blocked by remote WSL2 host state, not by model-side rejection
- current preferred structure-side packet:
  - `vel_tok32_semantic_topogate_k085`
- current preferred exact-I2SB theory-check packet:
  - `i2sb_tok32_semantic_topogate_sigma0p02_residual`

## Update Rule

- whenever a phase-2 packet changes queue priority, status, or preferredness:
  - update this CSV
  - then update [README.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/612-phase2/README.md) only if the policy text itself changed
