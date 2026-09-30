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

## Running Eval Curve

| epoch | transfer CLIP-S | transfer LPIPS | all-pairs CLIP-S | all-pairs LPIPS | IDT CLIP-S | train time | eval wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.671841 | 0.314633 | 0.703257 | 0.312567 | 0.828923 | 141.2s | 214.8s |
| 2 | 0.671824 | 0.314633 | 0.703222 | 0.312550 | 0.828815 | 104.5s | 206.4s |
| 3 | 0.671615 | 0.314685 | 0.703043 | 0.312613 | 0.828756 | 104.3s | 211.0s |
| 4 | 0.671725 | 0.314793 | 0.703094 | 0.312718 | 0.828567 | 104.5s | 208.4s |

### 2026-06-15 14:46 CST, early read

- The first two all-ckpt eval points are essentially flat: transfer style changes by `-0.000017` and transfer LPIPS changes by `+0.000001`.
- This is safe but not style-positive. It supports the actuation-bottleneck diagnosis: the shallow spatial carrier gate is likely being swallowed by the existing backend/output bottleneck.
- Do not close yet; wait for epoch 3 to avoid overreacting to a two-point curve. If epoch 3 remains flat, archive this probe and move to a stronger, still-controlled actuation experiment.

### 2026-06-15 14:52 CST, three-point read

- Epoch 3 confirms the early flat read: transfer style is now `0.671615`, below epoch 1 by `-0.000226`, while LPIPS is slightly worse by `+0.000053`.
- All-pairs style also declines from `0.703257` to `0.703043`.
- Interpretation: this shallow `spatial_carrier_gate` does not create an effective new fiber coordinate under the current backend. It is likely gated/normalized away before the final delta head.
- Action: allow the already-started epoch 4 eval to finish as one tail point, then stop and close as negative evidence unless epoch 4 creates a new Pareto point.

## Closure Decision

### 2026-06-15 15:01 CST, stopped

- Epoch 4 tail did not create a new Pareto point: transfer `0.671725 / 0.314793`, worse than epoch 1 on both style and LPIPS.
- Remote training PID `1691` was stopped. A leftover epoch 5 eval child (`run_evaluation.py`, PID `2102`) was also stopped; GPU returned to `57 MiB / 0%`.
- Decision: `archive_negative`.
- Reason: the change is implementation-safe and in-band, but it does not improve style. This supports the fiber-bundle diagnosis that merely adding a shallow spatial carrier gate upstream is insufficient when the final delta head/backend remains the active bottleneck.
- Next mechanism: keep tokenizer/solver/loss fixed and test a stronger actuation path that bypasses the low-rank final-delta bottleneck more directly, preferably a residual style-delta side head or style-conditioned multi-head delta basis with identity/zero init.
