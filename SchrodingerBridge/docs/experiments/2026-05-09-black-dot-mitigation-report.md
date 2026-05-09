# 2026-05-09 Black Dot Mitigation Report

## Scope

This round focused on one concrete failure mode: localized black artifacts / dark holes that appeared in the aggressive full-band SWD regime. We also cleaned the workspace by moving older experiment assets under [`exp/`](</g:/GitHub/Latent_Style/SchrodingerBridge/exp:1>).

Current active runs intentionally kept at repo root:

- training: [`o10_m1`](</g:/GitHub/Latent_Style/SchrodingerBridge/o10_m1:1>), [`o10_m2`](</g:/GitHub/Latent_Style/SchrodingerBridge/o10_m2:1>), [`o10_m3`](</g:/GitHub/Latent_Style/SchrodingerBridge/o10_m3:1>)
- evaluation: [`o10m1`](</g:/GitHub/Latent_Style/SchrodingerBridge/o10m1:1>), [`o10m2`](</g:/GitHub/Latent_Style/SchrodingerBridge/o10m2:1>), [`o10m3`](</g:/GitHub/Latent_Style/SchrodingerBridge/o10m3:1>)

Archived historical experiments:

- run artifacts: [`exp/runs`](</g:/GitHub/Latent_Style/SchrodingerBridge/exp/runs:1>)
- old sweep/config bundles: [`exp/configs`](</g:/GitHub/Latent_Style/SchrodingerBridge/exp/configs:1>)
- early legacy projects: [`exp/legacy`](</g:/GitHub/Latent_Style/SchrodingerBridge/exp/legacy:1>)
- move manifest: [`exp/archive_manifest.json`](</g:/GitHub/Latent_Style/SchrodingerBridge/exp/archive_manifest.json:1>)

## Prior Debug Conclusion

From the earlier artifact tracing reports:

- [`exp/configs/orthogonal_phase_space_sweep_debug/reports/l5_epoch10_batch00.json`](</g:/GitHub/Latent_Style/SchrodingerBridge/exp/configs/orthogonal_phase_space_sweep_debug/reports/l5_epoch10_batch00.json:1>)
- [`exp/configs/orthogonal_phase_space_sweep_debug/reports/d3_epoch20_batch00.json`](</g:/GitHub/Latent_Style/SchrodingerBridge/exp/configs/orthogonal_phase_space_sweep_debug/reports/d3_epoch20_batch00.json:1>)

we already had a stable working hypothesis:

1. The black dots were not primarily NaN/Inf events.
2. The hottest path was `terminal_swd -> dec_out.weight -> pred_velocity -> pred_endpoint`.
3. Attention was very hard (`mean_top1_prob` near `0.96`), but the artifact looked less like checkerboard upsampling and more like local over-driving of endpoint amplitudes under hard semantic routing.
4. So the most promising mitigation direction was not "more clamps everywhere", but rebalancing force allocation:
   - lower `terminal_swd_weight`
   - increase `w_kinetic`
   - add a little `w_cycle`
   - soften `semantic_attn_temperature`

## New Mitigation Runs

Three 10-epoch mitigations were trained from the current main codebase:

- [`exp/configs/orthogonal_phase_space_sweep_10_validation/g0_blackdot_m1_softbalance10.json`](</g:/GitHub/Latent_Style/SchrodingerBridge/exp/configs/orthogonal_phase_space_sweep_10_validation/g0_blackdot_m1_softbalance10.json:1>)
- [`exp/configs/orthogonal_phase_space_sweep_10_validation/g0_blackdot_m2_damped10.json`](</g:/GitHub/Latent_Style/SchrodingerBridge/exp/configs/orthogonal_phase_space_sweep_10_validation/g0_blackdot_m2_damped10.json:1>)
- [`exp/configs/orthogonal_phase_space_sweep_10_validation/g0_blackdot_m3_stylekeep10.json`](</g:/GitHub/Latent_Style/SchrodingerBridge/exp/configs/orthogonal_phase_space_sweep_10_validation/g0_blackdot_m3_stylekeep10.json:1>)

### Parameter summary

| run | `terminal_swd_weight` | `w_kinetic` | `w_cycle` | `semantic_attn_temperature` | intent |
|---|---:|---:|---:|---:|---|
| `M1` | 12.0 | 0.40 | 0.15 | 0.12 | balanced softening |
| `M2` | 10.0 | 0.45 | 0.20 | 0.12 | stronger damping / safest point |
| `M3` | 14.0 | 0.40 | 0.10 | 0.12 | keep more style pressure |

Artifacts:

- train logs: [`o10_m1/train_stdout.log`](</g:/GitHub/Latent_Style/SchrodingerBridge/o10_m1/train_stdout.log:1>), [`o10_m2/train_stdout.log`](</g:/GitHub/Latent_Style/SchrodingerBridge/o10_m2/train_stdout.log:1>), [`o10_m3/train_stdout.log`](</g:/GitHub/Latent_Style/SchrodingerBridge/o10_m3/train_stdout.log:1>)
- eval summaries: [`o10m1/summary.json`](</g:/GitHub/Latent_Style/SchrodingerBridge/o10m1/summary.json:1>), [`o10m2/summary.json`](</g:/GitHub/Latent_Style/SchrodingerBridge/o10m2/summary.json:1>), [`o10m3/summary.json`](</g:/GitHub/Latent_Style/SchrodingerBridge/o10m3/summary.json:1>)

## Results

### Main metrics

| run | style transfer `clip_style` | `clip_content` | `LPIPS` | photo->art `clip_style` | photo->art `clip_content` | photo->art `LPIPS` |
|---|---:|---:|---:|---:|---:|---:|
| previous `L5` | 0.6966 | 0.7330 | 0.5442 | 0.6877 | 0.7202 | 0.5645 |
| previous `D3` | 0.6984 | 0.7302 | 0.5521 | 0.6875 | 0.7124 | 0.5714 |
| `M1` | 0.6790 | 0.7920 | 0.4763 | 0.6612 | 0.7818 | 0.5023 |
| `M2` | 0.6713 | 0.8280 | 0.4416 | 0.6500 | 0.8085 | 0.4707 |
| `M3` | 0.6880 | 0.7718 | 0.4991 | 0.6748 | 0.7620 | 0.5159 |

### Visual read

- `M1`: materially cleaner than the older `L5/D3` region, but still a bit soft in brighter atmospheric scenes.
- `M2`: the cleanest and most stable point of this round. It gives up some style strength, but the gain in structural continuity and LPIPS is real.
- `M3`: best "style retention" of the new trio, but it starts drifting back toward the older darker / harsher artifacts.

Reference grids:

- [`o10m1/summary_grid.png`](</g:/GitHub/Latent_Style/SchrodingerBridge/o10m1/summary_grid.png:1>)
- [`o10m2/summary_grid.png`](</g:/GitHub/Latent_Style/SchrodingerBridge/o10m2/summary_grid.png:1>)
- [`o10m3/summary_grid.png`](</g:/GitHub/Latent_Style/SchrodingerBridge/o10m3/summary_grid.png:1>)

## Mature Conclusion For Now

The black-dot problem is controllable through training-side force balancing.

The strongest working conclusion at this point is:

1. **It is not mainly an inference-only bug.**  
   Inference scaling can hide symptoms, but the checkpoint quality is determined in training.

2. **The dominant lever is reducing endpoint over-drive.**  
   Lowering `terminal_swd_weight`, raising `w_kinetic`, and adding a small `w_cycle` clearly improves stability.

3. **Softening semantic routing helps.**  
   Raising `semantic_attn_temperature` from `0.08` to `0.12` appears compatible with cleaner outputs.

4. **There is a clear trade-off frontier.**
   - `M2` is the best current anti-artifact point.
   - `M3` is the best current compromise if we insist on keeping more style energy.

### Recommended next move

If the goal is to keep pushing while staying sane:

- promote `M2` as the current stability baseline
- next local sweep should be narrow, around:
  - `terminal_swd_weight = 10.0 ~ 11.0`
  - `w_kinetic = 0.42 ~ 0.48`
  - `w_cycle = 0.15 ~ 0.20`
  - `semantic_attn_temperature = 0.11 ~ 0.13`

That is a smaller and cleaner search space than the earlier broad sweeps.

## Index / Tracking

The registry has been refreshed at:

- [`docs/experiments/experiment_registry.json`](</g:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/experiment_registry.json:1>)
- [`docs/experiments/experiment_registry.csv`](</g:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/experiment_registry.csv:1>)

It now captures both:

- current root-level runs (`o10_m1`, `o10m1`, etc.)
- archived runs under [`exp/runs`](</g:/GitHub/Latent_Style/SchrodingerBridge/exp/runs:1>)
