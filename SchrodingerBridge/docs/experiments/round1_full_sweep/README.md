# Round 1 Full Sweep Folder

This folder is reserved for the round-1 tokenizer / backbone / solver sweep.

Expected per-family artifacts:

- `plan.md`
- `remote_run.md`
- `fast_curve_read.md`
- `local_deep_review.md`
- `closure.md`
- `decision.md`

Machine-readable artifacts live beside the run roots:

- `clip_lpips_curve.csv`
- `round1_convergence.json`
- shortlist manifests
- local frozen `VLM` snapshots

Generic round-1 helpers:

- remote train:
  - [launch_remote_round1_family_train.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_round1_family_train.py)
- remote fast eval watcher:
  - [launch_remote_round1_family_fast_eval.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_round1_family_fast_eval.py)
  - preferred authority path for all-ckpt `CLIP-S + LPIPS` during formal remote training
- remote packet sync:
  - [sync_round1_remote_fast_eval_packet.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/sync_round1_remote_fast_eval_packet.py)
  - [watch_sync_round1_remote_fast_eval_packet.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_sync_round1_remote_fast_eval_packet.py)
  - refreshes tracked local authority packets only when a settled point or convergence state actually advances
- local fast eval watcher:
  - [watch_local_round1_family_fast_eval.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_local_round1_family_fast_eval.py)
  - [launch_local_round1_family_fast_eval_detached.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_local_round1_family_fast_eval_detached.py)
  - protected by:
    - [local_gpu_lock.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/local_gpu_lock.py)
  - intended for delayed local heavy review / backfill, not as the primary formal convergence authority
- remote bestfew image-backed rerun:
  - [launch_remote_round1_family_bestfew_rerun.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_round1_family_bestfew_rerun.py)
- local pull:
  - [pull_remote_round1_family_localreview.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/pull_remote_round1_family_localreview.py)
- local review:
  - [run_local_round1_family_review.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_local_round1_family_review.py)
- bestfew pipeline:
  - [run_round1_family_bestfew_pipeline.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_round1_family_bestfew_pipeline.py)
  - [launch_local_round1_family_review_detached.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_local_round1_family_review_detached.py)
  - [run_round1_family_stageclose_when_ready.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_round1_family_stageclose_when_ready.py)
- external-baseline `VLM` snapshots:
  - [build_round1_family_external_vlm_manifests.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/build_round1_family_external_vlm_manifests.py)
  - [run_round1_family_external_vlm_packet.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_round1_family_external_vlm_packet.py)
  - [launch_round1_family_external_vlm_detached.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_round1_family_external_vlm_detached.py)
  - [watch_vlm_snapshot_summaries.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_vlm_snapshot_summaries.py)
- next-family launcher:
  - [run_round1_family_queue.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_round1_family_queue.py)
  - [watch_launch_round1_queue_when_idle.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_launch_round1_queue_when_idle.py)
