# Stylesig Fast-Eval Watcher

Date: 2026-06-10

This note records the recovery action for the active `stylesig` line.

Problem:

- the remote train kept advancing through later checkpoints
- but no `full_eval_fast_snapshot/` tree was appearing
- so the mainline mechanism lane had training evidence but no decision-grade
  quality evidence

Chosen fix:

- keep the remote train alive
- do not interrupt the single active lane
- attach a post-train fast-eval watcher that waits for training to finish and
  then backfills the fresh retained checkpoints into:
  - `full_eval_fast_snapshot`

Watcher launcher:

- [launch_remote_knee_spatial_carriergate_bodydecoder_qedgegated_pattn_stylesig_fast_eval_watcher.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_knee_spatial_carriergate_bodydecoder_qedgegated_pattn_stylesig_fast_eval_watcher.py)

Watched run:

- `aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_pattn_stylesig_seed42_b8a2`

Remote watcher log:

- `/mnt/i/Github/Latent_Style/exp/inmortal-exp/knee_spatial_carriergate_bodydecoder_qedgegated_pattn_stylesig_fast_eval_watcher.log`

Intended output root:

- `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_pattn_stylesig_seed42_b8a2/full_eval_fast_snapshot`

Current verified state:

- the watcher has been launched
- the watcher log exists
- the watcher is currently reporting:
  - `train_alive=True`
- the corresponding remote watcher process is also visible in `ps`
