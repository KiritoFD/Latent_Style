# Phase 2: eval_only_pc_solver

Date: 2026-06-13

## Goal

- test whether `solver_pc` can pull structure back from a style-strong `xpred + pattn` checkpoint
- keep this branch inference-only:
  - no retraining
  - no remote main-lane preemption
- evaluate the Phase 2 hypothesis:
  - training for style
  - inference for structure

## Entry Surface

- override config:
  - [phase2_eval_pc_solver_xpred_pattn.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_eval_pc_solver_xpred_pattn.json)
- eval wrapper:
  - [run_phase2_eval_only_pc_solver.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase2_eval_only_pc_solver.py)
- remote launcher:
  - [launch_remote_phase2_eval_only_pc_solver.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_phase2_eval_only_pc_solver.py)
- handoff watcher:
  - [watch_phase2_velocity_handoff.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_phase2_velocity_handoff.py)
- merged-runtime support:
  - `run_evaluation.py` now accepts `--config_override`
  - `LGTInference` now merges checkpoint config with an override config before building the inference model

## Why This Matters

- the old placeholder with `num_epochs = 0` was not a clean eval-only contract
- it also mixed training semantics into what should be pure inference
- the new path is explicit:
  - choose a checkpoint
  - choose a solver override
  - run the existing evaluation stack directly

## Immediate Candidate

- source checkpoint:
  - the style-heavy `xpred + pattn` line named by the Phase 2 pivot docs
  - confirmed remote path exists:
    - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/inmortal-exp/aaai2027_inmortal_xpred_kmanifold_pattn_seed42_b16_e12_continue/epoch_0011.pt`
- first read target:
  - can `solver_pc` pull LPIPS down materially while keeping enough style to beat the current velocity line on the same transfer/all-pairs surface?
- remote launch policy:
  - use a single-shot eval-only launcher
  - refuse launch while the formal 3060 lane is still occupied
  - hand off only after the current velocity line closes or is explicitly stopped

## Handoff Command

- when the current formal lane is released, the intended first remote call is:
  - `python SchrodingerBridge/tools/experiments/launch_remote_phase2_eval_only_pc_solver.py --checkpoint /mnt/i/Github/Latent_Style/SchrodingerBridge/exp/inmortal-exp/aaai2027_inmortal_xpred_kmanifold_pattn_seed42_b16_e12_continue/epoch_0011.pt --force-regen`

## Automated Handoff Gate

- dry-run watcher:
  - `python SchrodingerBridge/tools/experiments/watch_phase2_velocity_handoff.py`
- default closure rule encoded there:
  - if either latest transfer/all-pairs `LPIPS >= 0.70`, immediate `fail_stop`
  - if either latest transfer/all-pairs `LPIPS >= 0.40`, immediate `archival_stop`
  - otherwise require:
    - `latest_settled_epoch >= 6`
    - `best_in_newest_2 = false`
    - no style-side recovery:
      - `latest_all_pairs_clip_style < 0.7005`
      - `latest_transfer_clip_style < 0.6725`
- execution mode:
  - `python SchrodingerBridge/tools/experiments/watch_phase2_velocity_handoff.py --execute --force-regen`
- execution behavior:
  - stop the remote velocity lane
  - wait for GPU idle
  - launch `eval_only_pc_solver`

## Local Wiring Proof

- checkpoint used for merge-path smoke:
  - `G:\GitHub\Latent_Style\SchrodingerBridge\exp\local_wsl_wikiart512_hist_b32_e8\epoch_0008.pt`
- override used:
  - [phase2_eval_pc_solver_xpred_pattn.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_eval_pc_solver_xpred_pattn.json)
- proof result:
  - `solver_family = solver_pc`
  - `solver_corrector_steps = 2`
  - `solver_corrector_mode = latent_lowpass`
  - `solver_corrector_lowpass_kernel = 5`
  - `solver_corrector_clamp = 0.0`
  - `integrated_shape = [1, 4, 32, 32]`
- interpretation:
  - checkpoint config + override merge is live
  - eval-only solver settings now actually reach the inference runtime
  - this is no longer a dead placeholder config

## Status

- local plumbing complete
- remote launcher complete
- first execution note:
  - the first handoff launch failed because the wrapper forwarded `--device` into `run_evaluation.py`
  - that wrapper bug has been fixed locally
- current remote state:
  - the formal velocity lane has already been released
  - `eval_only_pc_solver` was relaunched at `2026-06-13 04:10`
  - health read passed with GPU memory around `1033 MiB`
  - the eval finished at `2026-06-13 04:12`
  - output:
    - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/phase2_eval_only_pc_solver/epoch_0011/summary.json`
  - result:
    - transfer `0.729014 / 0.621056`
    - all-pairs `0.735295 / 0.611310`
    - eval wall `115.15s`
- queue meaning:
  - this read has now been interpreted as negative evidence
  - solver-only correction does not recover structure enough to pass the Phase 2 gate
  - the branch closes as archival evidence and hands the decision back to the next training-side queue
