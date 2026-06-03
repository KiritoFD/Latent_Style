Distinct5 tokenizer-localization remote preflight

Date: 2026-06-03
Owner surface: remote RTX 3060 / experiment-note only

Packet checked:
- `G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-03-tokenizer-localization/README.md`
- `G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-03-tokenizer-localization/launch_manifest_20260603.md`

Remote code root:
- `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge`

Remote sync truth

- direct remote `git fetch origin --prune` did not advance the remote worktree to the newest pushed head during this pass;
- remote `HEAD` remained:
  - `867d17439b89eff6726d04f6207509129aba02e5`
- the Distinct5 tokenizer-localization packet files required from local commit `abefc9263` were manually synced into the remote clean worktree:
  - `configs/aaai2027/tokenizer_localization_l_e1_seed42_b44_base.json`
  - `configs/aaai2027/tokenizer_localization_l_e1_stylebranch_seed42_b44.json`
  - `configs/aaai2027/tokenizer_localization_l_e1_executoronly_seed42_b44.json`
- the remote clean worktree `src/trainer.py` now contains the needed `executor_only` freeze-mode branch:
  - aliases include `renderer_only`, `fresh_executor`, `freeze_style_branch`
  - valid freeze-mode set includes `executor_only`
  - trainable-parameter branch includes `if mode == "executor_only": ...`

Pre-launch truth checks

1. Reviewed `L e1` checkpoint
- status: found
- exact path:
  - `I:\Github\Latent_Style\SchrodingerBridge\exp\distinct5_512_ema_variant_l_content_adaptive_annealed_queue_e3_b44_remote\epoch_0001.pt`

2. New Distinct5 `L e1` packet configs in remote clean worktree
- status: found
- exact paths:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\configs\aaai2027\tokenizer_localization_l_e1_seed42_b44_base.json`
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\configs\aaai2027\tokenizer_localization_l_e1_stylebranch_seed42_b44.json`
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\configs\aaai2027\tokenizer_localization_l_e1_executoronly_seed42_b44.json`

3. Distinct5 packed latent root and pairing cache
- status: found
- exact paths:
  - `I:\wikiart_distinct5_samam_512_latents_ema\train`
  - `I:\wikiart_distinct5_samam_512_latents_ema\train\.latent_cache`
  - `I:\wikiart_distinct5_samam_512_latents_ema\train\.latent_cache\prototype_pairing_top8.pt`

4. Distinct5 test split
- status: found
- exact path:
  - `I:\wikiart_distinct5_samam_512_classview\test`

Preflight result

- passed for the updated Distinct5 `L e1` packet
- no silent fallback to the superseded legacy256 chain was used

Launch result

Both manifest arms were launched from the remote clean worktree using the synced Distinct5 packet files.

1. Fresh style branch, frozen executor
- task:
  - `SB_TokenLoc_L_E1_STYLE_S42`
- output dir:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\aaai2027_tokenizer_localization_l_e1_stylebranch_seed42_b44`
- train log:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\aaai2027_tokenizer_localization_l_e1_stylebranch_seed42_b44\remote_train.log`
- current state at launch note time:
  - running on remote 3060

2. Frozen style branch, fresh executor
- task:
  - `SB_TokenLoc_L_E1_EXEC_S42`
- output dir:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\aaai2027_tokenizer_localization_l_e1_executoronly_seed42_b44`
- train log:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\aaai2027_tokenizer_localization_l_e1_executoronly_seed42_b44\remote_train.log`
- current state at launch note time:
  - launched and waiting for the style-branch arm to finish `full_eval/epoch_0003/summary.json` before starting its own GPU training, to avoid overlapping formal 3060 use

Evidence at note time

- style-branch log shows live training:
  - packed latent cache loaded from `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/packed/...`
  - reviewed `L e1` resume checkpoint loaded
  - `Freeze mode=style_branch`
- executor-only log shows live queue state:
  - `WAITING_FOR_STYLE`

Recovery update

- original style-branch auto full-eval crash point:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\aaai2027_tokenizer_localization_l_e1_stylebranch_seed42_b44\remote_train.log`
  - failing surface:
    - `/home/xy/venvs/samam312/bin/python /mnt/i/Github/Latent_Style_TokenizerClean/SchrodingerBridge/src/utils/run_evaluation.py ...`
  - first observed error:
    - `ModuleNotFoundError: No module named 'diffusers'`

- recovery task:
  - task:
    - `SB_TokenLoc_L_E1_STYLE_EVAL_RECOVER_S42`
  - wrapper:
    - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\aaai2027_tokenizer_localization_l_e1_stylebranch_seed42_b44\launch_full_eval_recovery.cmd`
  - recovery log:
    - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\aaai2027_tokenizer_localization_l_e1_stylebranch_seed42_b44\remote_full_eval_recovery.log`
  - preserved earlier recovery attempt:
    - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\aaai2027_tokenizer_localization_l_e1_stylebranch_seed42_b44\remote_full_eval_recovery_attempt1.log`

- recovered style full-eval outputs now present:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\aaai2027_tokenizer_localization_l_e1_stylebranch_seed42_b44\full_eval\epoch_0001\summary.json`
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\aaai2027_tokenizer_localization_l_e1_stylebranch_seed42_b44\full_eval\epoch_0002\summary.json`
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\aaai2027_tokenizer_localization_l_e1_stylebranch_seed42_b44\full_eval\epoch_0003\summary.json`

- executor-only handoff:
  - queue log:
    - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\aaai2027_tokenizer_localization_l_e1_executoronly_seed42_b44\remote_train.log`
  - confirmed waiting through:
    - `[WAITING_FOR_STYLE] 2026/06/03 09:25:52`
  - confirmed start after style `epoch_0003` summary landed:
    - `[START] 2026/06/03 09:26:52`
  - executor-only arm entered active training from the remote clean worktree

Latest live status after the handoff

- verified by the active remote owner (`Linnaeus`) from the same executor log:
  - training reached `epoch_0003.pt`
  - auto full-eval has started
  - log evidence includes:
    - `Running full eval for ...executoronly.../epoch_0001.pt -> .../full_eval/epoch_0001`
    - timestamp observed in the remote log:
      - `2026-06-03 09:30:03`
- current blocking milestone:
  - wait for executor-only `full_eval/epoch_0001..0003/summary.json` to land
    before opening the matched localization packet for adversarial review

Executor-only full-eval crash and recovery path

- latest verified failure surface:
  - `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp\aaai2027_tokenizer_localization_l_e1_executoronly_seed42_b44\remote_train.log`
- exact failing point in the executor log:
  - `Running full eval for exp/aaai2027_tokenizer_localization_l_e1_executoronly_seed42_b44/epoch_0001.pt -> exp/aaai2027_tokenizer_localization_l_e1_executoronly_seed42_b44/full_eval/epoch_0001`
- latest verified error:
  - `ImportError: CLIPModel requires the PyTorch library but it was not found in your environment.`

Interpretation

- executor-only training itself is complete through `epoch_0003.pt`;
- the blocked surface is executor-only evaluation, not training or checkpoint
  creation;
- this mirrors the earlier style-branch eval-environment failure pattern rather
  than invalidating the training run.

Current recovery action

- `Linnaeus` has been reassigned to recover the executor-only full-eval on the
  same remote machine while preserving the original output tree;
- preferred recovery path:
  - same-machine Windows `py -3` full-eval recovery, since that route already
    recovered the style-branch summaries successfully;
- target artifacts remain:
  - `full_eval/epoch_0001/summary.json`
  - `full_eval/epoch_0002/summary.json`
  - `full_eval/epoch_0003/summary.json`
