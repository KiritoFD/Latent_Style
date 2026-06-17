# 616 Clean Bridge-Noise Probe Round 2: Authoritative Control + Vertical Projection

Date: 2026-06-17

## Why this round exists

The 2026-06-17 implementation audit and the clean OT round-8 closure now agree
on one important remaining geometry gap:

- `training_target_projection_mode = pure_vertical_flow` was active
- `training_bridge_noise_projection_mode` still remained `none` in the retained lines

That means the endpoint contract was already trying to remove low-frequency
horizontal contamination, while the stochastic bridge perturbation could still
inject full-spectrum structural corruption.

This round isolates only that missing half.

## Narrowed rerun contract

Fixed across both rows:

- retained OT control:
  - `coupling_structure_cost_mode = self_affinity_gw`
  - `coupling_cost_composition = appearance_plus_structure`
  - `coupling_solver = sinkhorn_unbalanced`
- retained target geometry:
  - `training_target_projection_mode = pure_vertical_flow`
- same one-epoch / 60-step / transfer-only eval contract
- same tokenizer / backbone / solver family

Changed axis:

- control:
  - `training_bridge_noise_projection_mode = none`
- candidate:
  - `training_bridge_noise_projection_mode = pure_vertical_flow`
  - `training_bridge_noise_projection_preserve_rms = true`

RMS preservation stays on so we do not confuse geometric cleanup with a trivial
reduction in total noise power.

## Success criterion

Promote bridge-noise vertical projection only if the candidate can improve at
least one of:

- `transfer CLIP-S`
- `transfer LPIPS`
- `base_structural_drift`
- `low_freq_leak`

without showing evidence that the gain is only coming from an effectively
weaker bridge after RMS preservation.

Important read rule:

- after the OT round-8 verdict, bridge-noise geometry is now a **mainline**
  616 variable, not a side diagnostic

## Configs and launcher

- control:
  [phase616_clean_bridge_noise_probe_control_faststep60_e1_authoritative.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_bridge_noise_probe_control_faststep60_e1_authoritative.json)
- candidate:
  [phase616_clean_bridge_noise_probe_vertical_faststep60_e1_authoritative.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_bridge_noise_probe_vertical_faststep60_e1_authoritative.json)
- local runner:
  [run_phase616_clean_bridge_noise_probe_round2_authoritative.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase616_clean_bridge_noise_probe_round2_authoritative.sh)
- remote launcher:
  [launch_phase616_clean_bridge_noise_probe_round2_authoritative_remote.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_phase616_clean_bridge_noise_probe_round2_authoritative_remote.sh)

## Live status

- `2026-06-17`: authoritative pair prepared
- `2026-06-17`: launcher hardened with `python/python3` autodetect to match the
  later 616 remote launch surface
- `2026-06-17 09:00 CST`: remote WSL launch started successfully on
  `100.115.18.62 / Ubuntu-26.04`
  - task: `phase616_clean_bridge_noise_probe_round2_authoritative`
  - first active process observed under GPU monitor
  - prelaunch guard passed:
    - `prelaunch_gpu_memory_used_mib = 379`
  - first health check passed:
    - `health_gpu_memory_used_mib = 2322`
  - control lane entered training on the retained clean 616 base
- `2026-06-17 09:02 CST`: control train stage closed cleanly and entered fast eval
  - train closure:
    - `loss = 13.9741`
    - `flow = 0.4372`
    - `kinetic_energy = 13.0889`
    - `ot_cost = 2.7054`
    - `ot_target_gini = 0.059`
    - `ot_target_max_mass = 0.353`
    - `base_structural_drift = 0.2069`
    - `fiber_energy_ratio = 0.442`
    - `low_freq_leak = 3.4725`
    - `target_base_shift = 0.0126`
    - `epoch_time_sec = 91.5`
    - `avg_optimizer_step_time_sec = 1.551`
    - `gpu = 56.1/97.0%`
    - `vram = 5.09/6.49 GB`
    - `power = 93.2/141.8 W`

### Authoritative control eval

- `2026-06-17 09:05 CST` control eval closed successfully.
- authoritative transfer result:
  - `CLIP-S = 0.658573`
  - `LPIPS = 0.784350`
- eval timing:
  - `eval_wall_total_sec = 215.04`
  - `eval_total_sec = 38.75`
  - `generation_sec = 114.38`
  - `vae_decode_sec = 57.19`

GPU summary:

- mean VRAM `2.54 GiB`
- peak VRAM `6.49 GiB`
- mean util `35.8%`
- mean power `69.3 W`

Step-50 probe snapshot:

- `ot_cost = 2.5690`
- `ot_target_gini = 0.0456`
- `ot_target_max_mass = 0.3438`
- `base_structural_drift = 0.05126`
- `fiber_energy_ratio = 0.41325`
- `low_freq_leak = 2.00185`
- `training_bridge_noise_projection_active = 0.0`

### Authoritative candidate eval: `pure_vertical_flow` bridge noise

- `2026-06-17 09:11 CST` candidate eval closed successfully.
- authoritative transfer result:
  - `CLIP-S = 0.625777`
  - `LPIPS = 0.751436`
- matched delta versus control:
  - `style = -0.032796`
  - `LPIPS = -0.032915`
- eval timing:
  - `eval_wall_total_sec = 215.41`
  - `eval_total_sec = 35.72`
  - `generation_sec = 118.05`
  - `vae_decode_sec = 56.87`

GPU summary:

- mean VRAM `2.65 GiB`
- peak VRAM `6.50 GiB`
- mean util `38.3%`
- mean power `72.7 W`

Step-50 probe snapshot:

- `ot_cost = 2.5690`
- `ot_target_gini = 0.0456`
- `ot_target_max_mass = 0.3438`
- `base_structural_drift = 0.04174`
- `fiber_energy_ratio = 0.40713`
- `low_freq_leak = 1.80650`
- `training_bridge_noise_projection_active = 1.0`
- `training_bridge_noise_projection_pre_rms = 1.00002`
- `training_bridge_noise_projection_post_rms = 1.00002`
- `training_bridge_noise_projection_low_rms = 0.19663`
- `training_bridge_noise_projection_high_rms = 0.97885`

## Authoritative verdict

What this rerun proves:

- the bridge-noise projection path is genuinely active and observable
- RMS preservation worked as intended, so the comparison is not explained by a
  trivial drop in total noise magnitude
- projecting bridge noise into the `pure_vertical_flow` split **does** reduce
  structure-side corruption signals:
  - `base_structural_drift`: `0.05126 -> 0.04174`
  - `low_freq_leak`: `2.00185 -> 1.80650`

But on transfer it also causes a meaningful style regression:

- `CLIP-S`: `0.65857 -> 0.62578`
- `LPIPS`: `0.78435 -> 0.75144`

Interpretation:

- this is positive evidence that the mechanism is doing the intended geometric
  thing on the bridge
- but in its current standalone form it is **too suppressive** for a
  style-priority program
- the result should therefore be read as:
  - `geometry-positive`
  - `frontier-negative`

Promotion decision:

- do **not** promote bridge-noise vertical projection as the new standalone
  retained mainline
- retain it as a diagnosed structure-cleaning mechanism with a clear style cost

Queue impact:

1. keep the bridge-noise projection implementation and probes
2. keep `terminal_affine` as the retained style-lift auxiliary line
3. the next high-value follow-on should test whether the structure gain from
   bridge-noise projection can be paired with a style-recovery mechanism rather
   than replacing the mainline by itself
