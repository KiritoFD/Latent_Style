# Clean Stats Probe Round 2: Authoritative Control + Terminal-Affine Rerun

Date: 2026-06-17

## Why this round exists

Round 1 already answered the high-level transfer question:

- control `none`: `CLIP-S = 0.675156`, `LPIPS = 0.714525`
- candidate `terminal_affine`: `CLIP-S = 0.701927`, `LPIPS = 0.695852`
- candidate `normalized_solver`: `CLIP-S = 0.685882`, `LPIPS = 0.715152`

So the strongest signal is already clear:

- `terminal_affine` is a strong positive matched delta
- `normalized_solver` is not the leading promotion candidate on this base

But round 1 still had a closure-quality issue:

- the live `loss_dict` and `numeric_debug.jsonl` contained non-zero
  `transport_stats_*`
- the epoch CSV writer had not yet been fixed to persist those fields

That means round 1 is valid transfer evidence, but not the final authoritative
white-box closure for the stats-track mechanism.

## Narrowed rerun contract

This rerun intentionally narrows scope to the only pair that matters for the
next promotion decision:

- control:
  `transport_stats_mode = none`
- candidate:
  `transport_stats_mode = terminal_affine`

We do **not** rerun `normalized_solver` first because:

- it already lost clearly to `terminal_affine` on transfer
- it also looked worse on train-side structural probes
- spending the next 5-6 minutes on the likely winner is cleaner than burning
  the lane on a lower-value third branch

Everything else remains fixed:

- `contract_family = phase616`
- retained OT control:
  - `coupling_structure_cost_mode = self_affinity_gw`
  - `coupling_cost_composition = appearance_plus_structure`
  - `coupling_solver = sinkhorn_unbalanced`
- retained target geometry:
  - `training_target_projection_mode = pure_vertical_flow`
- one epoch
- `stop_after_global_steps = 60`
- lightweight transfer-only eval

## Success criterion

Promote `terminal_affine` to the retained 616 stats-track mechanism only if the
authoritative rerun confirms:

- transfer stays clearly above the control
- LPIPS stays below the control
- `transport_stats_*` fields are now present in the training CSV
- no hidden degradation appears in `ot_target_gini`, `ot_target_max_mass`,
  `base_structural_drift`, `fiber_energy_ratio`, or `low_freq_leak`

If the rerun reproduces the earlier direction, the next 616 mainline should
treat:

- `terminal_affine` as retained
- `normalized_solver` as deferred / non-leading
- the next unresolved variable as either bridge-noise projection or the
  wavelet split operator

## Configs and launcher

- control:
  [phase616_clean_stats_probe_control_none_faststep60_e1_authoritative.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_stats_probe_control_none_faststep60_e1_authoritative.json)
- candidate:
  [phase616_clean_stats_probe_terminal_affine_faststep60_e1_authoritative.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_stats_probe_terminal_affine_faststep60_e1_authoritative.json)
- local runner:
  [run_phase616_clean_stats_probe_round2_authoritative.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase616_clean_stats_probe_round2_authoritative.sh)
- remote launcher:
  [launch_phase616_clean_stats_probe_round2_authoritative_remote.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_phase616_clean_stats_probe_round2_authoritative_remote.sh)

## Live status

- `2026-06-17 07:16` remote WSL authoritative rerun launched successfully on
  `100.115.18.62 / Ubuntu-26.04`.
- control train finished cleanly before candidate handoff:
  - `loss = 13.8326`
  - `flow = 0.4346`
  - `kinetic_energy = 12.9485`
  - `ot_cost = 2.7054`
  - `base_structural_drift = 0.20419`
  - `fiber_energy_ratio = 0.44407`
  - `low_freq_leak = 3.41610`
  - `target_base_shift = 0.01258`
  - `epoch_time_sec = 104.02`
  - `avg_optimizer_step_time_sec = 1.7630`
- most importantly, the new authoritative `training_*.csv` now visibly contains
  the fixed `transport_stats_*` columns and values:
  - `transport_stats_active = 0.0`
  - `transport_stats_bank_loaded = 0.0`
  - `transport_stats_mode_terminal_affine = 0.0`
  - `transport_stats_mode_normalized_solver = 0.0`
  - `transport_stats_valid_styles = 5.0`
  - `transport_stats_missing_bank = 1.0`

Interpretation:

- the writer-path bug is no longer present in the authoritative rerun
- once control eval and the `terminal_affine` lane finish, this round will be a
  real closure-quality white-box comparison rather than a transfer-only read

### Authoritative control eval

- `2026-06-17 07:22` control eval closed successfully.
- authoritative transfer result:
  - `CLIP-S = 0.674232`
  - `LPIPS = 0.702816`
- eval timing:
  - `eval_wall_total_sec = 247.65`
  - `eval_total_sec = 44.71`
  - `generation_sec = 135.80`
  - `vae_decode_sec = 61.87`

This is now the authoritative baseline for the narrowed stats-track closure.
The remaining decision hinges on whether `terminal_affine` reproduces its
earlier positive delta against this new control under the fixed CSV path.

### Authoritative candidate eval: `terminal_affine`

- `2026-06-17 07:28` candidate eval closed successfully.
- authoritative transfer result:
  - `CLIP-S = 0.701788`
  - `LPIPS = 0.704400`
- matched delta versus authoritative control:
  - `style = +0.027556`
  - `LPIPS = +0.001583`
- eval timing:
  - `eval_wall_total_sec = 242.87`
  - `eval_total_sec = 39.73`
  - `generation_sec = 136.84`
  - `vae_decode_sec = 61.03`

GPU summary:

- control:
  - mean VRAM `2.54 GiB`
  - peak VRAM `9.40 GiB`
  - mean util `40.4%`
  - mean power `67.8 W`
- candidate:
  - mean VRAM `2.58 GiB`
  - peak VRAM `6.56 GiB`
  - mean util `37.7%`
  - mean power `71.1 W`

Authoritative white-box observability is now present in the candidate run:

- `transport_stats_active = 1.0`
- `transport_stats_bank_loaded = 1.0`
- `transport_stats_mode_terminal_affine = 1.0`
- `transport_stats_mode_normalized_solver = 0.0`
- `transport_stats_valid_styles = 5.0`
- `transport_stats_missing_bank = 0.0`
- `transport_stats_mean_delta = 0.45110`
- `transport_stats_std_delta = 0.12793`

Matched step-50 probe comparison against the authoritative control direction:

- `ot_target_gini` stayed aligned at about `0.04559`
- `ot_target_max_mass` stayed aligned at about `0.34380`
- `base_structural_drift` stayed low at `0.05963`
- `fiber_energy_ratio` stayed moderate at `0.41859`
- `low_freq_leak` stayed low at `2.15899`

## Authoritative verdict

What this rerun proves:

- the `transport_stats_*` CSV writer path is now fixed and usable for closure
- `terminal_affine` still gives a strong, repeatable style lift on the retained
  616 OT / vertical base
- the positive style delta survives the authoritative rerun
- the LPIPS improvement from round 1 did **not** survive exactly; it regressed
  to near-parity with a slight loss (`+0.00158`)

Interpretation:

- `terminal_affine` is retained as the best current stats-track mechanism
- but it should be treated as a **style-lift terminal remap**, not as the core
  solution to 616's training-geometry problem
- this mechanism is promotion-worthy as a retained auxiliary line, not as the
  main explanation for the model's next structural breakthrough

Queue impact:

1. keep `terminal_affine` available as the retained stats-track branch
2. keep `normalized_solver` deferred / non-leading
3. shift the next mainline 616 experiment to bridge-noise geometry, because OT
   has now been downgraded and stats-track has been positively but only
   partially resolved
