# Distinct5-512 SDXL-fix Eval Repair

Date: 2026-06-05

Scope: repair the local `Distinct5-512` SDXL-fix latent run after training
completed but deferred `full_eval` failed on the Windows `classview` test root.

## Failure Cause

The training run itself completed through `epoch_0008.pt`, but the deferred
`full_eval` launched from:

- [local_distinct5_512_sdxl_fix_k_b32_e8.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/archive/20260605_local_distinct5_sdxl_fix/local_distinct5_512_sdxl_fix_k_b32_e8.json)

still pointed `training.test_image_dir` at:

- `F:/wikiart_distinct5_samam_512_classview/test`

On Windows this root resolves to unreadable reparse-point entries for Python
image I/O, so `run_evaluation.py` failed as soon as it tried to open the first
source image.

Failing log:

- [local_distinct5_sdxl_fix_train.err.log](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/local_distinct5_sdxl_fix_train.err.log)

## Repair Applied

1. Updated the local SDXL-fix config to use the real readable test root:
   - `F:/wikiart_distinct5_512_images/test`
2. Hardened
   [run_evaluation.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/utils/run_evaluation.py)
   so Windows eval falls back from known `_classview` aliases to the real image
   root when available.
3. Added a reusable repair runner:
   - [run_local_sdxl_fix_eval_repair.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_local_sdxl_fix_eval_repair.py)

## Repair Execution

Manual proof run that closed the first checkpoint:

```powershell
python SchrodingerBridge/src/utils/run_evaluation.py `
  --checkpoint G:\GitHub\Latent_Style\SchrodingerBridge\exp\local_distinct5_512_sdxl_fix_k_b32_e8\epoch_0001.pt `
  --output G:\GitHub\Latent_Style\SchrodingerBridge\exp\local_distinct5_512_sdxl_fix_k_b32_e8\full_eval\epoch_0001 `
  --test_dir F:\wikiart_distinct5_512_images\test `
  --cache_dir G:\GitHub\Latent_Style\eval_cache `
  --clip_hf_cache_dir G:\GitHub\Latent_Style\eval_cache\hf `
  --profile_timing `
  --force_regen
```

Background repair for the remaining curve:

- pid file:
  - [local_sdxl_fix_eval_repair.pid](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/local_sdxl_fix_eval_repair.pid)
- stdout:
  - [local_sdxl_fix_eval_repair.out.log](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/local_sdxl_fix_eval_repair.out.log)
- stderr:
  - [local_sdxl_fix_eval_repair.err.log](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/local_sdxl_fix_eval_repair.err.log)

Launch form:

```powershell
python SchrodingerBridge/tools/experiments/run_local_sdxl_fix_eval_repair.py --start-epoch 2
```

## Current Status

Closed summaries already verified:

- `full_eval/epoch_0001/summary.json`
- `full_eval/epoch_0002/summary.json`
- `full_eval/epoch_0003/summary.json`
- `full_eval/epoch_0004/summary.json`
- `full_eval/epoch_0005/summary.json`

Compare-chain watcher still active:

- [watch_local_sdxl_eval_and_compare.log](/G:/GitHub/Latent_Style/SchrodingerBridge/_codex_tmp/watch_local_sdxl_eval_and_compare.log)

Expected comparison output after `epoch_0008` lands:

- `SchrodingerBridge/docs/experiments/local_distinct5_sdxl_fix_vs_ema_20260605/distinct5_eval_curve_comparison.md`

This note records repair status only. The question
"does SDXL-fix latent training beat the EMA latent baseline on Distinct5-512?"
remains open until the repaired curve finishes and the comparison artifact is
materialized.
