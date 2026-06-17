# 616 Clean Stats + Bridge Combo Round 1: Terminal-Affine Control vs Bridge-Vertical Combo

Date: 2026-06-17

## Why this round exists

The two standalone authoritative closures now say:

- `terminal_affine` is the retained style-lift auxiliary line
- `pure_vertical_flow` bridge-noise projection is geometry-positive but
  frontier-negative when used alone

The next minimal integration question is therefore:

- can the structure-side gain from bridge-noise cleanup survive when the model
  is evaluated on the stronger `terminal_affine` style-recovery line?

This is the smallest justified combo after the single-mechanism closures.

## Matched contract

Fixed across both rows:

- `transport_stats_mode = terminal_affine`
- retained OT control:
  - `coupling_structure_cost_mode = self_affinity_gw`
  - `coupling_cost_composition = appearance_plus_structure`
  - `coupling_solver = sinkhorn_unbalanced`
- retained target geometry:
  - `training_target_projection_mode = pure_vertical_flow`
- one epoch / 60 steps / transfer-only eval

Changed axis:

- control:
  - `training_bridge_noise_projection_mode = none`
- candidate:
  - `training_bridge_noise_projection_mode = pure_vertical_flow`
  - `training_bridge_noise_projection_preserve_rms = true`

## Decision rule

Promote the combo only if it can keep most of the `terminal_affine` style lift
while recovering some of the structure-side gain previously seen in the
bridge-noise standalone probe.

Concrete read:

- style regression should be materially smaller than the standalone
  bridge-noise penalty
- LPIPS / structure probes should move in the candidate's favor

## Configs and launcher

- control:
  [phase616_clean_stats_terminal_affine_control_faststep60_e1_authoritative.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_stats_terminal_affine_control_faststep60_e1_authoritative.json)
- candidate:
  [phase616_clean_stats_terminal_affine_plus_bridge_vertical_faststep60_e1_authoritative.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_stats_terminal_affine_plus_bridge_vertical_faststep60_e1_authoritative.json)
- local runner:
  [run_phase616_clean_stats_bridge_combo_round1_authoritative.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase616_clean_stats_bridge_combo_round1_authoritative.sh)
- remote launcher:
  [launch_phase616_clean_stats_bridge_combo_round1_authoritative_remote.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_phase616_clean_stats_bridge_combo_round1_authoritative_remote.sh)

## Live status

- `2026-06-17`: configs, runner, launcher, and closure doc prepared
- `2026-06-17 11:40 CST`: remote WSL launch started successfully on
  `100.115.18.62 / Ubuntu-26.04`
  - task: `phase616_clean_stats_bridge_combo_round1_authoritative`
  - transport stats bank loaded successfully before training
  - prelaunch guard passed:
    - `prelaunch_gpu_memory_used_mib = 432`
  - first health check passed:
    - `health_gpu_memory_used_mib = 2274`
  - control lane entered training on the retained `terminal_affine` stats line

### Authoritative control eval: `terminal_affine`

- `2026-06-17 11:46 CST` control eval closed successfully.
- authoritative transfer result:
  - `CLIP-S = 0.695357`
  - `LPIPS = 0.706042`
- eval timing:
  - `eval_wall_total_sec = 223.80`
  - `eval_total_sec = 40.01`
  - `generation_sec = 120.51`
  - `vae_decode_sec = 57.22`

GPU summary:

- mean VRAM `2.56 GiB`
- peak VRAM `6.64 GiB`
- mean util `35.3%`
- mean power `68.5 W`

Step-50 probe snapshot:

- `base_structural_drift = 0.04626`
- `fiber_energy_ratio = 0.39265`
- `low_freq_leak = 1.90169`
- `training_bridge_noise_projection_active = 0.0`
- `transport_stats_active = 1.0`
- `transport_stats_bank_loaded = 1.0`

### Authoritative candidate eval: `terminal_affine + bridge-noise vertical`

- `2026-06-17 11:51 CST` candidate eval closed successfully.
- authoritative transfer result:
  - `CLIP-S = 0.699773`
  - `LPIPS = 0.706801`
- matched delta versus control:
  - `style = +0.004416`
  - `LPIPS = +0.000759`
- eval timing:
  - `eval_wall_total_sec = 212.61`
  - `eval_total_sec = 35.42`
  - `generation_sec = 115.26`
  - `vae_decode_sec = 57.04`

GPU summary:

- mean VRAM `2.66 GiB`
- peak VRAM `6.64 GiB`
- mean util `40.2%`
- mean power `72.7 W`

Step-50 probe snapshot:

- `base_structural_drift = 0.04394`
- `fiber_energy_ratio = 0.40106`
- `low_freq_leak = 1.85335`
- `training_bridge_noise_projection_active = 1.0`
- `training_bridge_noise_projection_pre_rms = 1.00002`
- `training_bridge_noise_projection_post_rms = 1.00002`
- `transport_stats_active = 1.0`
- `transport_stats_bank_loaded = 1.0`

## Authoritative verdict

This matched combo round answers the key follow-on question from the standalone
bridge-noise closure:

- yes, the structure-cleaning effect can be paired with the retained
  `terminal_affine` style-lift line without repeating the large standalone
  style collapse

What improved:

- style moved up slightly: `+0.00442`
- `base_structural_drift` improved: `0.04626 -> 0.04394`
- `low_freq_leak` improved: `1.90169 -> 1.85335`
- RMS-preserved bridge projection remained genuinely active

What did not improve cleanly:

- LPIPS regressed slightly: `+0.00076`

Interpretation:

- the standalone bridge-noise mechanism was too suppressive on its own
- but as a controlled addition to `terminal_affine`, it becomes a plausible
  retained line rather than an immediate rejection
- this is a **soft positive** result, not a breakout

Promotion decision:

- retain `terminal_affine + bridge-noise vertical projection` as the current
  best combined 616 line among the mechanisms already implemented
- do not claim a frontier breakthrough from this alone because the LPIPS change
  is slightly negative and the absolute gain is small

Next mainline implication:

- stop spending primary budget on more OT / standalone bridge-noise variants
- move the next 616 implementation wave to the tokenizer-side mechanism from
  `design.md`, using this combo line as the new matched control surface
