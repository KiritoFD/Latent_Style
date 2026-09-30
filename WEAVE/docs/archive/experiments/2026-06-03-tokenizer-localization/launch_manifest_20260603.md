# Tokenizer Localization Launch Manifest

Date: 2026-06-03

This manifest converts the Distinct5 tokenizer-localization packet into a
remote-launch contract for the standing 3060 owner.

## Shared controls

- base family:
  - `configs/distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3.json`
- shared checkpoint:
  - reviewed `L e1`
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3_b44_remote/epoch_0001.pt`
- dataset:
  - `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train`
- test split:
  - `/mnt/i/wikiart_distinct5_samam_512_classview/test`
- hardware:
  - remote RTX 3060
- formal batch:
  - `44`
- seed:
  - `42`
- epoch budget:
  - `3`
- eval bundle:
  - per-epoch strict full eval for `epoch_0001`, `epoch_0002`, `epoch_0003`

## Arms

### 1. Fresh style branch, frozen executor

- config:
  - `configs/aaai2027/tokenizer_localization_l_e1_stylebranch_seed42_b44.json`
- task name:
  - `SB_TokenLoc_L_E1_STYLE_S42`
- output dir:
  - `exp/aaai2027_tokenizer_localization_l_e1_stylebranch_seed42_b44`
- train log:
  - `exp/aaai2027_tokenizer_localization_l_e1_stylebranch_seed42_b44/remote_train.log`

### 2. Frozen style branch, fresh executor

- config:
  - `configs/aaai2027/tokenizer_localization_l_e1_executoronly_seed42_b44.json`
- task name:
  - `SB_TokenLoc_L_E1_EXEC_S42`
- output dir:
  - `exp/aaai2027_tokenizer_localization_l_e1_executoronly_seed42_b44`
- train log:
  - `exp/aaai2027_tokenizer_localization_l_e1_executoronly_seed42_b44/remote_train.log`

## Pre-launch truth checks

Do not launch until all of the following are true on remote:

1. reviewed `L e1` checkpoint exists at the exact path above;
2. the two new configs exist in the remote clean worktree;
3. the remote clean worktree includes the new `executor_only` freeze mode;
4. logs will land under the listed output dirs.

## Success gate

Each arm counts as completed only if all of the following exist:

1. `remote_train.log` ends with training completion
2. `full_eval/epoch_0001/summary.json`
3. `full_eval/epoch_0002/summary.json`
4. `full_eval/epoch_0003/summary.json`

## Interpretation contract

Paper-safe interpretation after the run:

- if the style-branch arm wins clearly, tokenizer-side control remains the
  stronger bottleneck candidate in this packet;
- if the executor-only arm wins clearly, the reviewed `L e1` control was more
  usable than the current executor allowed;
- if both improve, the bottleneck remains joint;
- if neither improves materially, this localization route becomes negative
  evidence rather than a manuscript-facing mechanism closure.
