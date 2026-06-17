# Clean OT Probe Round 1

Date: 2026-06-16

## Purpose

This is the first strict `phase616` clean-base OT probe after stopping the slower scratch lane.

We keep the clean base fixed and change only the OT structure cost:

- control: `lowedge`
- candidate: `self_affinity_gw`

Common mechanism bundle:

- `contract_family = phase616`
- `solver_family = solver_i2sb`
- `transport_prediction_mode = endpoint`
- `output_appearance_alignment_mode = none`
- `coupling_solver = sinkhorn_unbalanced`
- `training_target_projection_mode = pure_vertical_flow`

## Why this round exists

The stopped scratch lane answered the speed question more than the OT question:

- it was running on the non-clean retained legacy base
- epoch wall time was too long for a first white-box OT diagnostic lane
- the probe schema landed mid-run, so observability was incomplete

This probe round fixes all three issues:

- strict clean base
- short fast lane via `b16 a1 vlen=0.10`
- finalized OT/tokenizer/fiber probe schema from step 1

## Configs

- control: [phase616_clean_ot_probe_lowedge_b16a1_vlen010_e6.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_probe_lowedge_b16a1_vlen010_e6.json)
- candidate: [phase616_clean_ot_probe_selfaffgw_b16a1_vlen010_e6.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_probe_selfaffgw_b16a1_vlen010_e6.json)
- launcher: [run_phase616_clean_ot_probe_round1.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase616_clean_ot_probe_round1.sh)

## New OT Mechanism

`self_affinity_gw` is a GW-like structural cost, not a full exact GW solver.

Implementation sketch:

- build low/edge/high latent tokens per image
- adaptive-pool them to an `8 x 8` token grid
- form each image's self-affinity matrix
- flatten the upper triangle into a topology descriptor
- compare descriptors with pairwise L2 cost inside minibatch OT

Interpretation:

- if it beats `lowedge` with lower `ot_target_gini` or similar LPIPS at higher style, the new OT geometry is helping
- if it only raises cost variance or hubness, then the descriptor is too noisy or too weakly normalized

## Closure Signals

Primary:

- transfer `CLIP-S`
- transfer `LPIPS`

White-box:

- `ot_target_gini`
- `ot_target_mass_entropy`
- `ot_target_max_mass`
- `ot_plan_entropy`
- `ot_barycentric_entropy`
- `ot_cost_mean/var`
- `ot_structure_cost_mean/var`
- `base_structural_drift`
- `fiber_energy_ratio`
- `low_freq_leak`
- `target_base_shift`
- `structured_style_tokenizer_spatial_svd_entropy`
- `structured_style_tokenizer_style_value_offdiag_cosine`

Runtime:

- `avg_optimizer_step_time_sec`
- `epoch_time_sec`
- `samples_per_sec`
- `cuda_peak_allocated_gb`
- `cuda_peak_reserved_gb`
- `gpu_vram_used_gb_mean/peak`
- `gpu_util_mean/peak`
- `gpu_power_w_mean/peak`

## Speed Policy

This round is explicitly speed-constrained:

- `batch_size = 16`
- `accumulation_steps = 1`
- `virtual_length_multiplier = 0.10`
- `num_epochs = 6`

The goal is not final convergence. The goal is to determine whether the clean OT signal is good enough to deserve a longer lane.

## Live Read

Updated on 2026-06-17 after the first matched control/candidate pass.

### Lowedge Control

The clean-base `lowedge` control was stopped early after enough negative evidence.

Observed transfer curve:

| epoch | CLIP-S | LPIPS | eval wall | generated-rank | offdiag-cos |
|---|---:|---:|---:|---:|---:|
| 1 | 0.6896 | 0.6971 | 221.2 s | 1.0257 | 0.9830 |
| 2 | 0.6993 | 0.6540 | 171.4 s | 1.0307 | 0.9798 |
| 3 | 0.6869 | 0.5291 | 171.0 s | 1.0329 | 0.9786 |
| 4 | 0.6994 | 0.4649 | 170.1 s | 1.0265 | 0.9826 |

Read:

- epoch wall and eval wall were acceptable for a probe lane
- transfer quality stayed far outside the retained frontier
- generated-delta rank stayed pinned near `1.03` with offdiag cosine near `0.98`
- this is strong evidence of style-direction collapse under the clean `lowedge` OT geometry

Decision:

- mark `lowedge` clean control as `negative-for-promotion`
- keep it as the matched baseline for this OT round
- do not spend the remaining epochs on it

### Self-Affinity GW Candidate

The clean `self_affinity_gw` candidate was promoted past the lowedge control, then stopped early once the next frequency-split probe was ready.

First matched evidence against the stopped control at `epoch_0001`:

| family | epoch | CLIP-S | LPIPS | eval wall | generated-rank | offdiag-cos |
|---|---|---:|---:|---:|---:|---:|
| lowedge | 1 | 0.6896 | 0.6971 | 221.2 s | 1.0257 | 0.9830 |
| self_affinity_gw | 1 | 0.6998 | 0.6458 | 213.9 s | 1.1150 | 0.9283 |

Matched delta versus `lowedge epoch_0001`:

- `CLIP-S`: `+0.0102`
- `LPIPS`: `-0.0513`
- `generated effective rank`: `+0.0893`
- `generated offdiag cosine`: `-0.0548`

This is the first clean positive signal that a more GW-like structural OT descriptor helps both transfer quality and internal style diversity at the same speed band.

Follow-up evidence:

| family | epoch | CLIP-S | LPIPS | eval wall | generated-rank | offdiag-cos |
|---|---|---:|---:|---:|---:|---:|
| self_affinity_gw | 2 | 0.7015 | 0.5480 | 175.4 s | 1.0378 | 0.9755 |

Interpretation:

- epoch 2 still beats the `lowedge` control on both transfer axes
- but the internal diversity signal partially collapses again by epoch 2
- by the epoch-3 train row, `fiber_energy_ratio` improved to about `0.690`, while the direction-diversity signal was no longer expanding

Decision:

- keep `self_affinity_gw` as the current retained OT mechanism
- do not spend more epochs on the plain avgpool split
- move immediately to the next controlled mechanism: wavelet vertical split

### Probe Plumbing Status

The `self_affinity_gw` lane is also the first clean probe lane with the extended training CSV schema fully active.

Observed epoch-1 train row:

- `ot_target_gini = 0.0593`
- `ot_target_mass_entropy = 1.1616`
- `ot_plan_entropy = 0.6177`
- `ot_barycentric_entropy = 0.6177`
- `base_structural_drift = 0.1419`
- `fiber_energy_ratio = 0.5083`
- `low_freq_leak = 2.8591`
- `target_base_shift = 0.0126`
- `structured_style_tokenizer_style_value_offdiag_cosine = 0.0053`

By epoch 2, the live train log reads:

- `loss = 6.4057`
- `ot_cost = 2.7128`
- `fiber_energy_ratio = 0.6620`
- `low_freq_leak = 1.4790`
- `avg_optimizer_step_time_sec = 0.943`
- `epoch_time_sec = 111.3`
- `samples_per_sec = 16.96`
- GPU band about `7.53 / 7.86 GiB` mean/peak with power about `130 / 140 W`

That is the current best 616 OT lane in both observability quality and throughput quality.

### Wavelet Vertical Split

The follow-up wavelet lane was started as:

- `coupling_structure_cost_mode = self_affinity_gw`
- `coupling_solver = sinkhorn_unbalanced`
- `training_target_projection_mode = pure_vertical_flow_wavelet`

Reason:

- `design.md` explicitly warns that `avgpool 5x5` is too crude a base/fiber separator
- the matched `self_affinity_gw` result was positive enough to retain OT
- the next clean variable is therefore the low/high split operator, not a new OT or solver change

Live health check on launch:

- step band returned to about `1.0-1.15 s/it` after warmup
- VRAM band about `6.94 GiB`
- power band about `136-145 W`

Status update on 2026-06-17:

- this lane was stopped before closure
- reason: too slow for the current OT decision stage, and it mixed the next low/high split variable into the same round
- decision: move back to OT-only matched probes and revisit wavelet only after a faster OT answer exists
