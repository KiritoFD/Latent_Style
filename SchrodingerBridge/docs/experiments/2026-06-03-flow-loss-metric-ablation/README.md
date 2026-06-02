# Flow-Loss Metric Ablation Bundle

Date: 2026-06-03

This directory is the execution bundle for the first matched paper-closing
ablation after the continuous reviewer lane was wired in.

## Scope

- dataset: `Distinct5-512`
- base family: `H`
- only changed variable: `bridge.loss_type`
- hardware: remote `RTX 3060`
- formal batch: `44`
- first seed block: `42`

## Configs

- `configs/aaai2027/flow_loss_h_base_mse_seed42.json`
- `configs/aaai2027/flow_loss_h_base_huber_seed42.json`
- `configs/aaai2027/flow_loss_h_base_l1_seed42.json`

## Remote run contract

- repo worktree:
  - `I:\Github\Latent_Style_TokenizerClean`
- remote WSL repo path:
  - `/mnt/i/Github/Latent_Style_TokenizerClean/SchrodingerBridge`
- shared eval cache:
  - `/mnt/i/Github/Latent_Style/eval_cache`

## First launch

First formal arm to launch:

- task name:
  - `SB_FlowLoss_H_MSE_S42`
- config:
  - `configs/aaai2027/flow_loss_h_base_mse_seed42.json`
- output dir:
  - `exp/aaai2027_flow_loss_h_base_mse_seed42_b44`
- log:
  - `exp/aaai2027_flow_loss_h_base_mse_seed42_b44/remote_train.log`

## Review-cycle coupling

After each seed-level completion:

1. update `docs/experiments/aaai2027_master_experiment_log.csv`
2. append a compact reviewer cycle to:
   - `docs/reviews/aaai2027_review_score_log.csv`
   - `docs/reviews/aaai2027_review_registry.csv`

After the full `mse / huber / l1` block:

1. refresh the full reviewer consensus
2. decide whether the latent-metric thesis stays broad or shrinks to the
   endpoint-side `W1` story
