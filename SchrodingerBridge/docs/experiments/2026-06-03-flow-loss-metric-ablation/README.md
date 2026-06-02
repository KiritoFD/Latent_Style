# Flow-Loss Metric Ablation Bundle

Date: 2026-06-03

This directory is the execution bundle for the first matched paper-closing
ablation after the continuous reviewer lane was wired in.

## Current status

The originally launched `mse / huber / l1` trio is now archived as a
**near-null operational control** because the post-run config audit showed:

- `objective_mode = omf`
- `w_flow = 0.0`

So `loss_type` never became the active compared term.

The repaired packet now lives here:

- theory packet:
  - `repaired_endpoint_metric_ablation_packet_20260603.md`
- remote launch manifest:
  - `repaired_endpoint_metric_launch_manifest_20260603.md`
- repaired configs:
  - `configs/aaai2027/endpoint_metric_h_omf_flow_mse_seed42.json`
  - `configs/aaai2027/endpoint_metric_h_omf_flow_huber_seed42.json`
  - `configs/aaai2027/endpoint_metric_h_omf_flow_l1_seed42.json`

These repaired arms keep `objective_mode = omf` but activate the endpoint term
with `w_flow = 1.0` and disable terminal SA-SWD with
`terminal_swd_weight = 0.0`, so the compared `loss_type` finally sits on the
active endpoint-matching path.

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

## Critical audit finding

On `2026-06-03`, we re-resolved the three configs through the project config
loader and found the following effective bridge settings:

| config | objective_mode | loss_type | w_flow |
| --- | --- | --- | ---: |
| `flow_loss_h_base_mse_seed42.json` | `omf` | `mse` | `0.0` |
| `flow_loss_h_base_huber_seed42.json` | `omf` | `huber` | `0.0` |
| `flow_loss_h_base_l1_seed42.json` | `omf` | `l1` | `0.0` |

Under the current `omf` objective implementation, `loss_type` enters the
active loss path only through the `w_flow > 0` branch. Since the resolved
configs keep `w_flow = 0.0`, this entire `mse / huber / l1` bundle is currently
best interpreted as a **non-probing or near-null ablation**, not as direct
evidence about whether `MSE`, `Huber`, or `L1` is better for the intended flow
term.

Practical consequence:

- the completed `MSE` and `Huber` rows remain useful as run-health and
  baseline-stability evidence
- the completed `L1` row belongs in the same category
- they do **not** close the paper's latent-metric or flow-loss thesis
- the next valid version of this ablation must either:
  - set `w_flow > 0`, or
  - move to the non-`omf` objective path that actually uses `loss_type` on the
    velocity regression term

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
- this means the MSE arm is healthy enough to validate the remote path, but not
  enough to support the intended loss-kernel thesis by itself.

## Huber seed42 closure

The second formal arm has now completed cleanly, including automatic full-eval.

- train task:
  - `SB_FlowLoss_H_HUBER_S42`
- run root:
  - `exp/aaai2027_flow_loss_h_base_huber_seed42_b44`
- log:
  - `exp/aaai2027_flow_loss_h_base_huber_seed42_b44/remote_train.log`
- completion:
  - `2026-06-03 02:07:54 +08:00` wrote `Training completed.`
  - `schtasks` returned to `Ready`
  - last result `0`

Recovered metrics:

| epoch | full clip_style | full content_lpips | transfer clip_style | transfer content_lpips | wall_total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `epoch_0001` | `0.6965` | `0.3322` | `0.6650` | `0.3402` | `97.7s` |
| `epoch_0002` | `0.6986` | `0.3560` | `0.6680` | `0.3651` | `95.7s` |
| `epoch_0003` | `0.6961` | `0.3409` | `0.6644` | `0.3498` | `95.3s` |

Immediate read:

- the Huber arm shows the same early-peak structure as the MSE arm
- best raw style is again `epoch_0002`
- best LPIPS is again `epoch_0001`
- the matched Huber results do **not** currently show a decisive advantage over
  MSE; the practical reading is parity, not a thesis-closing win
- after the config audit above, even this parity reading must be treated as
  descriptive only, because the ablation target was not activated as intended

This is exactly why the broader latent-metric story must remain narrow until the
full `mse / huber / l1` block is closed and compared together.

## L1 seed42 closure

The third formal arm has now completed cleanly, including automatic full-eval.

- train task:
  - `SB_FlowLoss_H_L1_S42`
- run root:
  - `exp/aaai2027_flow_loss_h_base_l1_seed42_b44`
- log:
  - `exp/aaai2027_flow_loss_h_base_l1_seed42_b44/remote_train.log`

Recovered metrics:

| epoch | full clip_style | full content_lpips | transfer clip_style | transfer content_lpips | wall_total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `epoch_0001` | `0.6964` | `0.3313` | `0.6651` | `0.3391` | `96.3s` |
| `epoch_0002` | `0.6988` | `0.3608` | `0.6685` | `0.3703` | `95.6s` |
| `epoch_0003` | `0.6960` | `0.3415` | `0.6645` | `0.3507` | `95.3s` |

Immediate read:

- `L1` lands almost exactly on top of the `MSE` and `Huber` runs
- that observed parity is now fully consistent with the config audit above:
  the three-way kernel switch did not activate the intended loss path
- the completed trio should therefore be treated as a **near-null operational
  control**, not as evidence for or against a broader manifold-aware metric
  claim

The next valid step is not to average these rows into a paper claim. It is to
repair the ablation design first.

## Replacement block after the config audit

The next paper-facing metric experiment should be a repaired block, not more
seeds on the invalidated trio.

Minimum valid options:

1. `OMF + active flow term`
   - keep `objective_mode = omf`
   - set `w_flow > 0`
   - then rerun `mse / huber / l1`

2. `True velocity-regression block`
   - switch to the non-`omf` objective path
   - ensure `loss_type` is applied directly to the active velocity regression
   - keep terminal SWD, kinetic, and eval scope otherwise matched

Reviewer-safe interpretation:

- the current trio is archived as an operational near-null control
- the repaired block is the first valid test of whether a local loss kernel
  matters at all for this family

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
