# Phase 2 Queue Manifest Note

Date: 2026-06-13

## Purpose

- provide one machine-readable phase-2 queue index
- avoid re-deriving the preferred packet order from multiple dated notes
- keep the current formal lane, structure-side reentry candidates, and diagnostic-only I2SB candidates in one place

## Manifest

- CSV:
  - [phase2_queue_manifest.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_queue_manifest.csv)
- validator:
  - [validate_phase2_queue_manifest.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/validate_phase2_queue_manifest.py)
- latest validation snapshot:
  - [phase2_queue_manifest_validation.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_queue_manifest_validation.json)
- current state snapshot:
  - [phase2_queue_state_snapshot.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_queue_state_snapshot.json)
- state snapshot builder:
  - [report_phase2_queue_state.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/report_phase2_queue_state.py)

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
  - distinguishes:
    - `tok32_safe_rescan`
    - `tok32_refresh`
    - `legacy64_endpoint`

## Current Read

- current formal packet:
  - `vel_tok32_safe_rescan_r2`
  - latest settled authority point is now:
    - `epoch_0008`
    - transfer `0.672774 / 0.389067`
    - all-pairs `0.700669 / 0.384913`
  - current read:
    - still in-band
    - still below the old safe shelf on style
    - LPIPS drifted back toward the ceiling
    - the lane is now closed as an in-band style plateau
- current preferred structure-side packet:
  - `vel_tok32_safe_semantic_topogate_k085`
  - first `b20a1` launch hit the runtime guard at `11093 MiB`
  - preferred relaunch now uses the `b16a1` packet
  - current live read:
    - `epoch_0002` is now the latest settled point
    - transfer `0.671915 / 0.361009`
    - all-pairs `0.700605 / 0.357866`
    - the lane is still clean, but the latest point has slipped back below the formal safe shelf
  - important read:
    - `epoch_0001` was a real first recovery candidate
    - but the queue should now be judged from the latest settled point, not from the earlier peak
- next queued structure-side probe:
  - `vel_tok32_safe_semantic_topogate_k085_appalign`
  - same safe parent and same topology-gate family
  - only adds the conservative tokenizer-guided output appearance head
  - intended to test whether the remaining transfer shortfall is low-order brightness / contrast mismatch
- next structural family fallback after that:
  - `vel_tok32_safe_pnp_selfinject`
  - queue contract:
    - active `structure_reentry` rows now carry same-lane watch fields
    - the automatic sequence is `topogate -> appalign -> pnp -> topo_anchor`
    - `queued_reference` rows remain documentation-only and are skipped by successor handoff
- current preferred exact-I2SB theory-check packet:
  - `i2sb_tok32_safe_semantic_topogate_sigma0p02_residual`
- current validation state:
  - `phase2_queue_manifest_validation.json -> ok = true`
- formal-lane recovery thresholds now also live in the manifest:
  - `watch_min_settled_epoch`
  - `watch_min_allpairs_style_recovery`
  - `watch_max_allpairs_lpips_for_recovery`
  - `watch_min_transfer_style_recovery`
  - `watch_max_transfer_lpips_for_recovery`
  - `watch_handoff_mode`

## Update Rule

- whenever a phase-2 packet changes queue priority, status, or preferredness:
  - update this CSV
  - then update [README.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/612-phase2/README.md) only if the policy text itself changed

## Operator Flow

- validate the manifest before trusting it:
  - [validate_phase2_queue_manifest.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/validate_phase2_queue_manifest.py)
- generate a one-shot combined queue/health/watcher snapshot:
  - [report_phase2_queue_state.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/report_phase2_queue_state.py)
- resolve the current preferred formal packet:
  - [resolve_phase2_queue_packet.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/resolve_phase2_queue_packet.py)
- manifest-driven recovery watcher example:
  - [watch_phase2_wsl_recover_and_launch.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_phase2_wsl_recover_and_launch.py)
  - local resolve-only dry path:
    - `python ...\\watch_phase2_wsl_recover_and_launch.py --manifest-csv ...\\phase2_queue_manifest.csv --validation-json ...\\phase2_queue_manifest_validation.json --lane-class formal_lane --resolve-only`
