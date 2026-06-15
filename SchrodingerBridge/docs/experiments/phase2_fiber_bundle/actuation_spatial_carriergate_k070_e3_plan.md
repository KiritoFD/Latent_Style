# Actuation Spatial Carrier Gate Probe

Date: 2026-06-15

## Goal

Test the `fiber.md` actuation-bottleneck diagnosis without using overdrive, PC lowpass, endpoint/I2SB, DINO, or metric-affecting affine calibration.

## Controlled Delta

- Parent: `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`
- Config: `configs/aaai2027/phase2_actuation_spatial_carriergate_k070_e3_vlen010.json`
- Only mechanism delta: `model.style_injection_mode="body_decoder"` with `style_injection_form="spatial_carrier_gate"`.
- Freeze: `training.freeze_mode="injection_only"`.
- Tokenizer: unchanged.
- Solver: unchanged `euler_legacy`.
- Corrector: `solver_corrector_mode="none"`.
- Overdrive: disabled.
- Metric postprocess: disabled.

## Schedule

- Short probe: `virtual_length_multiplier=0.10`, `num_epochs=8`.
- Save/eval every epoch.
- Remote 3060 VRAM target: under `11.0 GiB`; reduce batch/accum only if needed, not mechanism parameters.

## Decision Rule

- If transfer/all-pairs style improves while LPIPS stays in band, promote to full-length convergence run.
- If LPIPS worsens in e1/e2, inspect identity initialization and style-injection scale before continuing.
- If curves remain flat, archive as evidence that a shallow carrier gate is swallowed by the current output bottleneck.

## Launch Log

### 2026-06-15 14:26 CST, rejected launch

- Remote partial launch was stopped because the parent checkpoint path resolved under `SchrodingerBridge/exp` and the run logged `No checkpoint found, start from scratch`.
- Bad partial output and launcher logs were removed before relaunch:
  - `exp/aaai2027_phase2_actuation_spatial_carriergate_k070_e3_vlen010`
  - `logs/phase2_actuation_spatial_carriergate_k070_e3_vlen010.launch.log`
  - `logs/phase2_actuation_spatial_carriergate_k070_e3_vlen010.pid`
- Config fix: `training.resume_checkpoint="../exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt"`.

### 2026-06-15 14:34 CST, clean remote launch

- Remote host: `administrator@100.115.18.62:2222`, WSL root `/mnt/i/Github/Latent_Style/SchrodingerBridge`.
- PID: `1691`.
- Log: `logs/phase2_actuation_spatial_carriergate_k070_e3_vlen010.launch.log`.
- Remote smoke resolved parent to `/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt` and verified it exists.
- First health check:
  - Loaded parent: `Partially loaded resume ... epoch_0003.pt | loaded=282 skipped=0 missing=28 unexpected=0`.
  - Trainable scope: `freeze_mode=injection_only`, `trainable_count=28`.
  - GPU: `1840 MiB`, `0%` at the sampled instant, below the `11.0 GiB` guardrail.
  - Progress: epoch `1/8`, step `15/157`.
