# Experiments Tooling Index

This directory contains the operational scripts for `SchrodingerBridge`
training, eval, packet sync, plotting, and experiment bookkeeping.

Use this file as the stable entrypoint instead of guessing from filenames.

## Current Priority Surface

- round-1 tokenizer/backbone/solver sweep:
  - [run_round1_family_queue.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_round1_family_queue.py)
  - [audit_round1_queue_state.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/audit_round1_queue_state.py)
  - [report_round1_convergence.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/report_round1_convergence.py)
  - [sync_round1_remote_fast_eval_packet.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/sync_round1_remote_fast_eval_packet.py)
- current round-1 authority docs:
  - [2026-06-10-round1-full-sweep-master.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-10-round1-full-sweep-master.md)
  - [round1_full_sweep/README.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/README.md)

## Tool Groups

### Round-1 remote lane

- launch one family train lane:
  - [launch_remote_round1_family_train.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_round1_family_train.py)
- one-shot live runtime read:
  - [report_round1_family_runtime_status.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/report_round1_family_runtime_status.py)
- launch remote fast eval on retained checkpoints:
  - [launch_remote_round1_family_fast_eval.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_round1_family_fast_eval.py)
- segmented train/eval for memory-sensitive families:
  - [run_remote_round1_family_segmented.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_remote_round1_family_segmented.py)

### Round-2 remote lane

- default policy:
  - round-2 pure-SDE launchers reject DINO-conditioned configs unless `--allow-dino` is supplied explicitly
- phase-2 queue source of truth:
  - [phase2_queue_manifest.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_queue_manifest.csv)
  - [phase2_queue_manifest_validation.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_queue_manifest_validation.json)
  - [2026-06-13-phase2-queue-manifest-note.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-queue-manifest-note.md)
- validate and resolve the current preferred phase-2 packet:
  - [validate_phase2_queue_manifest.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/validate_phase2_queue_manifest.py)
  - [resolve_phase2_queue_packet.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/resolve_phase2_queue_packet.py)
- one-shot status read for a single active remote lane:
  - [report_remote_experiment_status.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/report_remote_experiment_status.py)
- run inference-only phase2 solver_pc review on an existing checkpoint:
  - [run_phase2_eval_only_pc_solver.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase2_eval_only_pc_solver.py)
- launch the same eval-only phase2 solver_pc review on the remote host when the formal lane is idle:
  - [launch_remote_phase2_eval_only_pc_solver.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_phase2_eval_only_pc_solver.py)
- watch the formal phase2 velocity lane and hand off to eval-only solver_pc when the documented closure rule is met:
  - [watch_phase2_velocity_handoff.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_phase2_velocity_handoff.py)
  - supports `--handoff-mode stop_only` when the active phase2 packet should close on LPIPS / plateau without automatically launching the legacy solver_pc follow-up
- manifest-driven WSL2 recovery watcher:
  - [watch_phase2_wsl_recover_and_launch.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_phase2_wsl_recover_and_launch.py)
  - now supports:
    - `--manifest-csv`
    - `--validation-json`
    - `--lane-class`
    - `--resolve-only`
  - the preferred phase-2 usage is to resolve the formal lane from the manifest instead of hard-coding config / run-name / watcher thresholds
- launch one family train lane:
  - [launch_remote_round2_family_train.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_round2_family_train.py)
- audit round-2 pure-latent / I2SB contract compliance:
  - [audit_round2_contracts.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/audit_round2_contracts.py)
- tokenizer-winner follow-on launch:
  - [launch_remote_round2_followon_train.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_round2_followon_train.py)
- one-shot eval-curve watcher:
  - [watch_round2_eval_curve.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_round2_eval_curve.py)
- compare an active round-2 curve against a chosen reference point:
  - [report_round2_reference_gap.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/report_round2_reference_gap.py)
- segmented train/eval fallback for resume-time VRAM spikes:
  - [run_remote_round2_family_segmented.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_remote_round2_family_segmented.py)
  - [launch_round2_family_segmented_detached.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_round2_family_segmented_detached.py)
- detached follow-up chain:
  - [launch_round1_family_followups_detached.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_round1_family_followups_detached.py)
  - [watch_round1_family_runtime_status.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_round1_family_runtime_status.py)
  - [watch_sync_round1_remote_fast_eval_packet.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_sync_round1_remote_fast_eval_packet.py)
  - [watch_launch_round1_queue_when_idle.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_launch_round1_queue_when_idle.py)

### Round-1 manifest and queue control

- family status retag/promotion:
  - [retag_round1_manifest_family.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/retag_round1_manifest_family.py)
  - [promote_next_round1_non_dino_candidate.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/promote_next_round1_non_dino_candidate.py)
- shared helpers:
  - [round1_manifest_utils.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/round1_manifest_utils.py)
  - [round1_paths.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/round1_paths.py)
  - [csv_utils.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/csv_utils.py)

### Round-1 review and closure

- local image-backed reruns:
  - [run_local_round1_family_bestfew_rerun.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_local_round1_family_bestfew_rerun.py)
  - [run_local_round1_family_review.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_local_round1_family_review.py)
- remote/local packet pulls:
  - [pull_remote_round1_family_localreview.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/pull_remote_round1_family_localreview.py)
  - [build_best_few_handoff.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/build_best_few_handoff.py)
  - [build_round1_localreview_prep_note.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/build_round1_localreview_prep_note.py)
- external board review:
  - [build_round1_family_external_vlm_manifests.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/build_round1_family_external_vlm_manifests.py)
  - [run_round1_family_external_vlm_packet.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_round1_family_external_vlm_packet.py)
  - [watch_vlm_snapshot_summaries.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_vlm_snapshot_summaries.py)

### WikiArts5 baseline reproduction

- dataset builders:
  - [build_wikiarts5_full_dataset.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/build_wikiarts5_full_dataset.py)
  - [build_wikiarts5_flat_view.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/build_wikiarts5_flat_view.py)
- WSL baseline runners:
  - [run_samam_wikiarts5_wsl.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_samam_wikiarts5_wsl.sh)
  - [run_samst_wikiarts5_wsl.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_samst_wikiarts5_wsl.sh)
  - [watch_resume_wikiarts5_segmented_until_converged.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_resume_wikiarts5_segmented_until_converged.py)
  - [watch_resume_wikiarts5_samst_until_converged.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_resume_wikiarts5_samst_until_converged.py)

### Eval, plots, and data products

- fast-curve construction:
  - [build_clip_lpips_curve_from_eval_root.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/build_clip_lpips_curve_from_eval_root.py)
  - [audit_round1_eval_timings.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/audit_round1_eval_timings.py)
- plotting:
  - [plot_round1_runtime_curve.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/plot_round1_runtime_curve.py)
  - [plot_round1_training_csv.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/plot_round1_training_csv.py)
  - [compare_distinct5_eval_curve.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/compare_distinct5_eval_curve.py)

### DINO and auxiliary precompute

- scope:
  - historical round-1 support only
  - not part of the active round-2 pure-latent / true-I2SB mainline unless a later board result justifies reviving it

- local cache build:
  - [run_local_round1_dino_cache_build.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_local_round1_dino_cache_build.py)
- remote cache build:
  - [launch_remote_round1_dino_cache_build.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_round1_dino_cache_build.py)
- warm-start/pretrain config and launch:
  - [prepare_round1_tokenizer_warmstart_config.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/prepare_round1_tokenizer_warmstart_config.py)
  - [launch_remote_round1_tokenizer_warmstart.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_round1_tokenizer_warmstart.py)
  - [prepare_round1_tokenizer_reconstruction_pretrain_config.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/prepare_round1_tokenizer_reconstruction_pretrain_config.py)
  - [launch_remote_round1_tokenizer_reconstruction_pretrain.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_round1_tokenizer_reconstruction_pretrain.py)

## Boundaries

- Keep generated run outputs under `SchrodingerBridge/aaai2027/` or the result
  roots they were designed for; do not add more loose runtime files beside the
  tooling.
- `tools/experiments` is for operational code, not for long narrative notes.
  Put summaries and decisions under `docs/experiments/`.
- If a helper becomes family-specific and no longer general, prefer moving it
  behind a family doc link rather than treating it as a global entrypoint.
