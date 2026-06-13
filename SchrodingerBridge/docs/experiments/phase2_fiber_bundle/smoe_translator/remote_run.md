# SMoE Translator Remote Run

## Launch 2026-06-14 04:42 Asia/Shanghai

- Family: `smoe_translator_k070_e3`.
- Config: `SchrodingerBridge/configs/aaai2027/phase2_smoe_translator_k070_e3_seed42_b12a1.json`.
- Task: `phase2-phase2_smoe_translator_k070_e3_seed42_b12a1-train`.
- Remote cwd: `/mnt/i/Github/Latent_Style`.
- Remote python: `/home/xy/venvs/samam312/bin/python`.
- Parent checkpoint: `/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`.
- Data root: `/mnt/i/wikiarts_5_full_notest_latents_ema/train`.
- Test root: `/mnt/i/wikiart_distinct5_samam_512_classview/test`.
- Train log: `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_phase2_smoe_translator_k070_e3_seed42_b12a1_train.log`.
- Output root: `/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_smoe_translator_k070_e3_seed42_b12a1`.

## First Health

- First heartbeat was inspected during launcher health window.
- Python PID: `456`.
- Initial GPU memory: `6969 MiB / 12288 MiB`.
- GPU utilization: `94%`.
- Log reached `Epoch 1/24`.
- Runtime guard: max memory `11000 MiB`; min-memory guard is warning-only because epoch-end full eval intentionally offloads the trainer.

## Decision

- The run is accepted as the SMoE matched-control lane despite using less than the preferred `9.0-10.8 GiB` formal band.
- Reason: increasing batch size would alter the tokenizer-only control schedule and contaminate the mechanism comparison.
- If the lower memory footprint persists, record it as an efficiency observation, not as a reason to change this lane.
