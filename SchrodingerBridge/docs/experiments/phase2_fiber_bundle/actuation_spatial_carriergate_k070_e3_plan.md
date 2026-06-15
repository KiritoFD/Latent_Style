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
