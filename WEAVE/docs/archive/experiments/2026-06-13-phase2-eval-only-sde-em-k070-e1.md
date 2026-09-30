# Phase 2: eval_only_sde_em_k070_e1

Date: 2026-06-13

## Role

- queued velocity-native stochastic diagnostic behind the current `k070` structure lane
- keep the same recovered `pure_latent_spatial + velocity` parent
- test whether a small Euler-Maruyama style stochastic step can lift style without reopening the structure problem

## Why This Exists

- the exact-I2SB `sigma0.02` line already proved one thing clearly:
  - endpoint-side stochastic transport can raise style
  - but on Distinct5 it immediately left the paper LPIPS band
- the current active family is different:
  - `pure_latent_spatial`
  - `velocity`
  - `semantic_self_topology_blend = 0.7`
  - recovered structure near `0.7036 / 0.3331` on `k070 epoch_0001`
- the clean next theory question is therefore not "can endpoint I2SB explode style again?"
- it is:
  - can the velocity family itself benefit from a very small stochastic bridge step while staying near the recovered LPIPS band

## Config

- override:
  - [phase2_eval_sde_em_k070_e1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_eval_sde_em_k070_e1.json)
- parent checkpoint:
  - `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0001.pt`
- generic remote launcher:
  - [launch_remote_phase2_eval_only_override.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_phase2_eval_only_override.py)
- generic local/remote runner:
  - [run_phase2_eval_only_override.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase2_eval_only_override.py)

## Deltas

- keep:
  - `transport_prediction_mode = velocity`
  - current `k070` checkpoint weights
  - `latent_lowpass` structure correction
- change only at inference:
  - `solver_family = solver_unsb_cycle`
  - `solver_corrector_steps = 2`
  - `solver_corrector_step_size = 0.06`
  - `solver_stochastic_noise_scale = 0.01`

## Intended Read

- positive diagnostic:
  - style rises relative to plain `k070 e1`
  - LPIPS stays inside the recovered band
  - the probe justifies a cleaner velocity-native stochastic branch instead of reopening endpoint transport
- negative diagnostic:
  - LPIPS moves up without real style gain
  - or the probe simply recreates the old "noise with no board improvement" pattern

## Execution Rule

- do not preempt the active remote `k070` training lane for this probe
- this is a queued diagnostic for the next idle window or post-close-gate decision point
- unlike the older `topogate e2` SDE config, this probe is anchored to the current recovered parent and current tokenizer family
