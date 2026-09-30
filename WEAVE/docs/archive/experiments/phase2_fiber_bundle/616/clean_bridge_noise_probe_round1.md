# 616 Clean Bridge-Noise Probe Round 1

## Summary

- OT repair is already on the retained `self_affinity_gw` line.
- training target geometry is already on the retained avgpool `pure_vertical_flow` line.
- the remaining clean gap is that training bridge noise was still injected in full spectrum, even when the target endpoint had already been projected into a content-anchored base plus target fiber.

This round isolates exactly that gap.

## Hypothesis

`pure_vertical_flow` fixed the supervised endpoint geometry, but not the stochastic bridge perturbation itself. If low-frequency Brownian energy is still entering the training bridge, the network is being asked to denoise structural corruption that the endpoint contract was explicitly trying to remove.

The first clean repair is therefore:

- keep OT fixed
- keep target projection fixed
- change only the training bridge noise spectrum

## Mechanism

New default-off bridge fields:

- `bridge.training_bridge_noise_projection_mode = none | source_low_target_high | wavelet_source_low_target_high | pure_vertical_flow | pure_vertical_flow_wavelet`
- `bridge.training_bridge_noise_projection_kernel`
- `bridge.training_bridge_noise_projection_preserve_rms`

Current round uses:

- control: `none`
- candidate: `pure_vertical_flow`
- `preserve_rms = true`

RMS preservation is part of the contract so the comparison does not get confounded by simply shrinking total noise power.

## Probe Contract

The training CSV now records:

- `training_bridge_noise_projection_active`
- `training_bridge_noise_projection_mode_*`
- `training_bridge_noise_projection_pre_rms`
- `training_bridge_noise_projection_post_rms`
- `training_bridge_noise_projection_low_rms`
- `training_bridge_noise_projection_high_rms`

The round still closes on the existing 616 signals:

- `transfer CLIP-S`
- `LPIPS`
- `ot_target_gini`
- `ot_target_max_mass`
- `base_structural_drift`
- `fiber_energy_ratio`
- `low_freq_leak`

## Run Set

- control:
  [phase616_clean_bridge_noise_probe_control_faststep60_e1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_bridge_noise_probe_control_faststep60_e1.json)
- candidate:
  [phase616_clean_bridge_noise_probe_vertical_faststep60_e1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_bridge_noise_probe_vertical_faststep60_e1.json)

Launch helpers:

- local runner:
  [run_phase616_clean_bridge_noise_probe_round1.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase616_clean_bridge_noise_probe_round1.sh)
- remote launcher:
  [launch_phase616_clean_bridge_noise_probe_round1_remote.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_phase616_clean_bridge_noise_probe_round1_remote.sh)

## Decision Rule

- if projected-noise training lowers `base_structural_drift` or `low_freq_leak` without immediate style collapse, retain it for the next 616 line
- if it only lowers those probes by effectively weakening the bridge despite RMS preservation, do not promote it
- if transfer and white-box probes disagree, extend with a longer matched rerun before closure

## Status

- `2026-06-17`: mechanism implemented locally as default-off
- `2026-06-17`: matched control/candidate configs prepared
- `2026-06-17`: authoritative isolated rerun configs prepared so the closure does not depend on reused control save roots
- next: launch the authoritative pair once the current OT descriptor lane is closed

Authoritative isolated launch surface:

- control:
  [phase616_clean_bridge_noise_probe_control_faststep60_e1_authoritative.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_bridge_noise_probe_control_faststep60_e1_authoritative.json)
- candidate:
  [phase616_clean_bridge_noise_probe_vertical_faststep60_e1_authoritative.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_bridge_noise_probe_vertical_faststep60_e1_authoritative.json)
- local runner:
  [run_phase616_clean_bridge_noise_probe_round2_authoritative.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase616_clean_bridge_noise_probe_round2_authoritative.sh)
- remote launcher:
  [launch_phase616_clean_bridge_noise_probe_round2_authoritative_remote.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_phase616_clean_bridge_noise_probe_round2_authoritative_remote.sh)
