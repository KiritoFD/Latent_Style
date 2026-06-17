# Clean Stats Probe Round 1

Date: 2026-06-17

## Purpose

The 2026-06-17 implementation audit established a precise gap between the 616
theory and the earlier clean runs:

- OT repair and `pure_vertical_flow` were real
- but the 616 "global stats track + local fiber track" was not yet present in
  the clean runtime path

This round is the first matched probe for that missing mechanism.

We keep the current retained OT control fixed:

- `self_affinity_gw`
- `appearance_plus_structure`
- `sinkhorn_unbalanced`
- `pure_vertical_flow`

Then change only the runtime transport stats path.

## Hypothesis

`docs/616/design.md` argues that one part of the current failure surface is
photometry / DC-stat mismatch:

- OT still couples through a metric that is imperfect on global appearance
- the clean target-geometry split protects structure
- but the transport still lacks a dedicated track for style-level mean/std
  statistics

This round tests whether a clean latent stats lane helps without reintroducing
the retired eval-only affine hacks.

## Candidates

Control:

- `transport_stats_mode = none`

Candidate A:

- `transport_stats_mode = terminal_affine`
- leave solver state unchanged
- only remap the final latent to the target style-bank statistics

Candidate B:

- `transport_stats_mode = normalized_solver`
- normalize the source latent into a solver track with zero-mean / unit-std
- run the bridge there
- restore target style-bank stats at the output

## Matched contract

Fixed:

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

Changed only:

- `model.transport_stats_mode`

## New observability

This round should produce the existing OT / fiber probes plus the new runtime
stats probes:

- `transport_stats_active`
- `transport_stats_bank_loaded`
- `transport_stats_mode_terminal_affine`
- `transport_stats_mode_normalized_solver`
- `transport_stats_source_mean_abs`
- `transport_stats_source_std_mean`
- `transport_stats_target_mean_abs`
- `transport_stats_target_std_mean`
- `transport_stats_mean_delta`
- `transport_stats_std_delta`
- `transport_stats_missing_bank`

Interpretation:

- if the candidate improves transfer while keeping OT hubness / leakage flat,
  that is evidence that the missing 616 stats lane mattered
- if the candidate only changes the stats probes but does not move transfer, the
  mechanism may be clean but low-value
- if transfer regresses while the stats probes look "cleaner", then the global
  stats lane is overconstraining the fiber mechanism

## Builder and launcher

- bank builder:
  [build_phase616_style_stats_bank.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/build_phase616_style_stats_bank.py)
- configs:
  - control:
    [phase616_clean_stats_probe_control_none_faststep60_e1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_stats_probe_control_none_faststep60_e1.json)
  - candidate A:
    [phase616_clean_stats_probe_terminal_affine_faststep60_e1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_stats_probe_terminal_affine_faststep60_e1.json)
  - candidate B:
    [phase616_clean_stats_probe_normalized_solver_faststep60_e1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_stats_probe_normalized_solver_faststep60_e1.json)
- launcher:
  [run_phase616_clean_stats_probe_round1.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase616_clean_stats_probe_round1.sh)
- remote launcher:
  [launch_phase616_clean_stats_probe_round1_remote.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_phase616_clean_stats_probe_round1_remote.sh)

Recommended command inside WSL / remote EXT4 workspace:

```bash
bash tools/experiments/run_phase616_clean_stats_probe_round1.sh
```

The launcher builds the style-stats bank first, then runs the three matched
configs with the standard GPU monitor wrapper.

## Decision rule

- promote a candidate only if transfer improves without worsening
  `ot_target_gini`, `ot_target_max_mass`, `low_freq_leak`, or
  `base_structural_drift`
- if `terminal_affine` helps but `normalized_solver` regresses, keep the
  terminal-only path as the narrower retained mechanism
- if both candidates regress, record the entire stats-track as
  `negative_for_promotion` on the current OT/vertical base rather than
  silently reverting to old moment/affine hacks

## Status

- `2026-06-17`: config set, bank builder, and launcher prepared locally.
- `2026-06-17 04:04` remote WSL launch started successfully on
  `100.115.18.62 / Ubuntu-26.04`.
  - active run:
    `aaai2027_phase616_clean_stats_probe_control_none_faststep60_e1`
  - remote run dir:
    `/home/xy/Latent_Style/SchrodingerBridge_phase616/SchrodingerBridge/exp/aaai2027_phase616_clean_stats_probe_control_none_faststep60_e1`
  - launch evidence:
    - style-stats bank build started from packed latent caches
    - training entered `src/run.py --config configs/aaai2027/phase616_clean_stats_probe_control_none_faststep60_e1.json`
    - live state reached `training_before_first_settled_eval`
    - sampled mid-launch remote GPU read:
      about `6.3 GiB` VRAM with active training process present
- `2026-06-17 04:09` control completed train + fast eval and the queue advanced
  to `terminal_affine`.
  - control train:
    - epoch wall about `89.6 s`
    - `ot_target_gini = 0.05935`
    - `ot_target_max_mass = 0.35310`
    - `base_structural_drift = 0.21157`
    - `fiber_energy_ratio = 0.42744`
    - `low_freq_leak = 3.52814`
  - control transfer eval:
    - `CLIP-S = 0.67513`
    - `LPIPS = 0.71452`
    - full-eval wall about `217.25 s`
- `2026-06-17 04:11` `terminal_affine` finished train and entered fast eval.
  - early train-only comparison versus control is directionally positive:
    - loss `13.97` vs `14.11`
    - `base_structural_drift = 0.20695` vs `0.21157`
    - `fiber_energy_ratio = 0.44233` vs `0.42744`
    - `low_freq_leak = 3.47252` vs `3.52814`
- instrumentation note:
  - the current running round started before `transport_stats_*` columns were
    added to [training.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/utils/training.py)
  - therefore this live round can still answer the high-level transfer question,
    but it is not a fully authoritative closure for the new stats probes
  - the CSV schema has now been fixed locally so the next matched rerun will
    capture:
    - `transport_stats_active`
    - `transport_stats_bank_loaded`
    - mode flags
    - source/target stat deltas
- `2026-06-17` audit correction:
  - the problem was not only the CSV header schema
  - the actual bug was that
    [utils/training.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/utils/training.py)
    exposed `transport_stats_*` in `TRAIN_LOG_COLUMNS` but did not write them in
    `append_training_log()`
  - remote `numeric_debug.jsonl` proves the live `loss_dict` already contained
    correct non-zero `transport_stats_*` values during both `terminal_affine`
    and `normalized_solver`
  - therefore the transfer summaries remain usable, but the original
    `training_*.csv` rows for this round are not authoritative for
    stats-observability closure
- Next:
  - let `terminal_affine` and `normalized_solver` finish this first queue
  - harvest their transfer deltas
  - then rerun the same matched trio once with the fixed CSV schema so the
    transport-stats probes become first-class closure evidence
  - place the results on the homepage frontier plot if a candidate closes

## Round-1 transfer closure

The first queue did finish and the transfer side is now known:

### Control: `transport_stats_mode = none`

- transfer:
  - `CLIP-S = 0.675156`
  - `LPIPS = 0.714525`
- train-side closure row:
  - `base_structural_drift = 0.211573`
  - `fiber_energy_ratio = 0.427437`
  - `low_freq_leak = 3.528142`
  - `ot_target_gini = 0.059350`
  - `ot_target_max_mass = 0.353099`

### Candidate A: `terminal_affine`

- transfer:
  - `CLIP-S = 0.701927`
  - `LPIPS = 0.695852`
- matched delta versus control:
  - `style = +0.026771`
  - `LPIPS = -0.018673`
- train-side closure row:
  - `base_structural_drift = 0.211923`
  - `fiber_energy_ratio = 0.430539`
  - `low_freq_leak = 3.536195`
  - `ot_target_gini = 0.059350`
  - `ot_target_max_mass = 0.353099`

### Candidate B: `normalized_solver`

- transfer:
  - `CLIP-S = 0.685882`
  - `LPIPS = 0.715152`
- matched delta versus control:
  - `style = +0.010726`
  - `LPIPS = +0.000627`
- train-side closure row:
  - `base_structural_drift = 0.218046`
  - `fiber_energy_ratio = 0.466059`
  - `low_freq_leak = 3.965846`
  - `ot_target_gini = 0.059350`
  - `ot_target_max_mass = 0.353099`

## Interim decision

Interim status:

- `terminal_affine`: `strong_positive_transfer_signal`
- `normalized_solver`: `not_leading_candidate`

Interpretation:

- `terminal_affine` is already the only stats-track branch with a clear
  promotion-worthy transfer gain on the current 616 OT/vertical base
- `normalized_solver` does not justify being the first authoritative rerun
  target because it gives a much smaller style gain, slightly worse LPIPS, and
  worse train-side structure/leakage probes

Therefore the next authoritative rerun is intentionally narrowed to:

- control `none`
- candidate `terminal_affine`

See:

- [clean_stats_probe_round2_authoritative.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/616/clean_stats_probe_round2_authoritative.md)
