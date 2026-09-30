# Round 1 Full Sweep Folder

This folder is the narrative surface for the round-1 tokenizer / backbone /
solver sweep.

For the current live status, read in this order:

1. [2026-06-10-round1-full-sweep-master.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-10-round1-full-sweep-master.md)
2. [2026-06-11-round1-node-summary-and-idle-cleanup.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-11-round1-node-summary-and-idle-cleanup.md)
3. [round1_family_manifest.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/round1_family_manifest.csv)

## Family folders

- tokenizer:
  - [tok_a_dino_dict](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/tok_a_dino_dict)
  - [tok_b_cross_image](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/tok_b_cross_image)
  - [tok_c_residual_adapter](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/tok_c_residual_adapter)
  - [tok_d_vlm_prompt](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/tok_d_vlm_prompt)
- attention:
  - [attn_sa_mod](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/attn_sa_mod)
  - [attn_gw_ot](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/attn_gw_ot)
  - [attn_gated_spade](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/attn_gated_spade)
  - [attn_pnp_selfinject](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/attn_pnp_selfinject)
- solver:
  - [solver_tangent_rk](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/solver_tangent_rk)
  - [solver_pc](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/solver_pc)
  - [solver_unsb_cycle](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/round1_full_sweep/solver_unsb_cycle)

## Expected per-family artifacts

- `plan.md`
- `remote_run.md`
- `fast_curve_read.md`
- `local_deep_review.md`
- `closure.md`
- `decision.md`

Machine-readable artifacts live beside the run roots:

- `metrics.csv`
- `summary.json`
- `clip_lpips_curve.csv`
- `round1_convergence.json`
- shortlist manifests
- local frozen `VLM` snapshots

## Operational entrypoints

- queue and manifest:
  - [run_round1_family_queue.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_round1_family_queue.py)
  - [audit_round1_queue_state.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/audit_round1_queue_state.py)
  - [retag_round1_manifest_family.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/retag_round1_manifest_family.py)
  - [promote_next_round1_non_dino_candidate.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/promote_next_round1_non_dino_candidate.py)
- remote authority path:
  - [report_round1_family_runtime_status.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/report_round1_family_runtime_status.py)
  - [launch_remote_round1_family_train.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_round1_family_train.py)
  - [launch_remote_round1_family_fast_eval.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_round1_family_fast_eval.py)
  - [sync_round1_remote_fast_eval_packet.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/sync_round1_remote_fast_eval_packet.py)
  - [watch_sync_round1_remote_fast_eval_packet.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_sync_round1_remote_fast_eval_packet.py)
- local heavy review only:
  - [watch_local_round1_family_fast_eval.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_local_round1_family_fast_eval.py)
  - [launch_local_round1_family_fast_eval_detached.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_local_round1_family_fast_eval_detached.py)
  - [local_gpu_lock.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/local_gpu_lock.py)
  - [run_local_round1_family_review.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_local_round1_family_review.py)
- wider tool index:
  - [tools/experiments/README.md](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/README.md)

## Current queue policy

- remote-side `CLIP-S + LPIPS` is the convergence authority
- local GPU is for delayed heavy review and image-backed reruns
- DINO tokenizer families are intentionally tail-blocked unless explicitly
  reopened
- if the generic queue would fall through into DINO-only `planned` families,
  audit and re-promote a smoke-ok non-DINO candidate first
