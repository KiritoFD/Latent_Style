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
