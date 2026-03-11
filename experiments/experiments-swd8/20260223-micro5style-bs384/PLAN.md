# Batch Experiment Plan

- Base config: `I:\Github\Latent_Style\Cycle-NCE\src\config.json`
- Pack root: `I:\Github\Latent_Style\Cycle-NCE\experiments\20260223-micro5style-bs384`
- Total experiments: `10`

## Run Command

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\src\run_pack_20260223_micro5style_bs384.ps1
```

## Matrix

| exp_id | phase | run_name | key changes |
|---|---|---|---|
| exp00 | phase1_60e | exp00-phase1_60e-baseline_bs384_lr2e4_bd64_sd256 | `runs\exp00-phase1_60e-baseline_bs384_lr2e4_bd64_sd256\config.json` |
| exp01 | phase1_60e | exp01-phase1_60e-capacity_floor_bd32_lc32_sd128 | `runs\exp01-phase1_60e-capacity_floor_bd32_lc32_sd128\config.json` |
| exp02 | phase1_60e | exp02-phase1_60e-capacity_ceiling_bd128_lc128_sd256_bs256 | `runs\exp02-phase1_60e-capacity_ceiling_bd128_lc128_sd256_bs256\config.json` |
| exp03 | phase1_60e | exp03-phase1_60e-style_ctrl_sd512_bs288 | `runs\exp03-phase1_60e-style_ctrl_sd512_bs288\config.json` |
| exp04 | phase1_60e | exp04-phase1_60e-lr4e4_speedup | `runs\exp04-phase1_60e-lr4e4_speedup\config.json` |
| exp05 | phase1_60e | exp05-phase1_60e-lr8e4_stress | `runs\exp05-phase1_60e-lr8e4_stress\config.json` |
| exp06 | phase1_60e | exp06-phase1_60e-l1_relax_wdl1_0p01 | `runs\exp06-phase1_60e-l1_relax_wdl1_0p01\config.json` |
| exp07 | phase1_60e | exp07-phase1_60e-swd_patch_1_3 | `runs\exp07-phase1_60e-swd_patch_1_3\config.json` |
| exp08 | phase2_150e | exp08-phase2_150e-deep_cosine_lr5e4_min1e6 | `runs\exp08-phase2_150e-deep_cosine_lr5e4_min1e6\config.json` |
| exp09 | phase2_150e | exp09-phase2_150e-flat_lr3e4 | `runs\exp09-phase2_150e-flat_lr3e4\config.json` |

## Notes

- All runs are isolated with their own `checkpoint.save_dir`.
- Phase 1 runs target 60 epochs; phase 2 runs target 150 epochs.
- Default batch size is 384, with OOM-safe overrides for heavy runs (exp02/exp03/exp08/exp09).
- `data.preload_to_gpu` is forced to `false` for this pack.
- `training.use_gradient_checkpointing` is forced to `true` for this pack.
- Use the sequential PowerShell runner under `src/`.