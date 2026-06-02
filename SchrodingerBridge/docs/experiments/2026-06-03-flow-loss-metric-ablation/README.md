# Flow-Loss Metric Ablation Bundle

Date: 2026-06-03

This directory is the execution bundle for the first matched paper-closing
ablation after the continuous reviewer lane was wired in.

## Scope

- dataset: `Distinct5-512`
- base family: `H`
- only changed variable: `bridge.loss_type`
- interpretation boundary: this probes the robustness of the velocity-regression
  penalty (`MSE / Huber / L1`) under a matched backbone and dataset; it is not
  yet a direct test of the broader endpoint-side `W1` vs. Euclidean-matching
  thesis.
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

## MSE seed42 closure

The first formal arm has now completed.

- train task:
  - `SB_FlowLoss_H_MSE_S42`
- run root:
  - `exp/aaai2027_flow_loss_h_base_mse_seed42_b44`
- retry eval log:
  - `exp/aaai2027_flow_loss_h_base_mse_seed42_b44/remote_full_eval_retry.log`

Observed issue:

- the first automatic full-eval attempt failed after generation because
  `run_evaluation.py` resolved the default offline CLIP source relative to the
  active worktree (`Latent_Style_TokenizerClean`) instead of the shared
  `eval_cache`.
- the fix landed in commit:
  - `1acaa17ee` `Fix offline CLIP cache resolution for worktree evals`

Recovered metrics after the eval retry:

| epoch | full clip_style | full content_lpips | transfer clip_style | transfer content_lpips | wall_total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `epoch_0001` | `0.6964` | `0.3313` | `0.6650` | `0.3391` | `41.6s` |
| `epoch_0002` | `0.6989` | `0.3595` | `0.6686` | `0.3687` | `94.7s` |
| `epoch_0003` | `0.6961` | `0.3422` | `0.6647` | `0.3512` | `94.3s` |

Immediate read:

- the matched `H`-base `MSE` arm behaves like the previous Distinct5 family:
  style peaks early and LPIPS is already competitive at `epoch_0001`.
- this means the MSE arm is healthy enough to unlock the queued `Huber` and
  `L1` arms.

## Huber seed42 live status

Remote owner check at `2026-06-03 01:59 +08:00` confirms the second formal arm
is already healthy on the remote clean worktree.

- train task:
  - `SB_FlowLoss_H_HUBER_S42`
- run root:
  - `exp/aaai2027_flow_loss_h_base_huber_seed42_b44`
- log:
  - `exp/aaai2027_flow_loss_h_base_huber_seed42_b44/remote_train.log`
- schtasks state:
  - `Running`
- latest sampled progress:
  - `Epoch 2/3`, `112/113` steps
- sampled device health:
  - `GPU 95%`, `VRAM 9723 MiB`, `Power 154.87 W`

This status block is intentionally operational, not interpretive. Final metric
comparison waits for the completed `mse / huber / l1` bundle.

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
3. if the broader thesis is kept, schedule a separate endpoint-objective
   ablation rather than over-interpreting the present `bridge.loss_type` probe
