# Clean OT Probe Round 2

Date: 2026-06-17

## Purpose

Round 1 established two things:

- `self_affinity_gw` is better than the old `lowedge` control.
- the follow-up `wavelet` lane is too slow for the current decision stage.

Round 2 therefore narrows back down to **OT-only** again:

- keep `contract_family = phase616`
- keep `solver_family = solver_i2sb`
- keep `transport_prediction_mode = endpoint`
- keep `coupling_solver = sinkhorn_unbalanced`
- keep `training_target_projection_mode = pure_vertical_flow`
- change only the OT structure cost

## Why the previous live lane was stopped

The `self_affinity_gw + pure_vertical_flow_wavelet` lane was stopped before closure.

Reason:

- it mixed the next low/high split variable into the OT round
- it took too long relative to the current need, which is quick white-box evidence
- the next decision should come from short matched probes, not another longer training lane

Status:

- treat that wavelet lane as `stopped_for_speed`
- keep its partial record as diagnostic-only
- do not use it as the main OT conclusion for 616

## New OT mechanism

New candidate: `lowedge_self_affinity_gw`

Definition:

- `lowedge` branch contributes low/edge/high summary statistics
- `self_affinity_gw` branch contributes pooled self-affinity topology
- both descriptors are L2-normalized separately
- the final OT structure descriptor concatenates the two with a fixed weight

Interpretation:

- this is the most direct implementation of the 616 note that OT should see both
  structure masks/edges and topology affinity
- unlike the stopped wavelet line, this round keeps the target projection fixed

## Matched configs

- control: [phase616_clean_ot_probe_selfaffgw_faststep60_e1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_probe_selfaffgw_faststep60_e1.json)
- candidate: [phase616_clean_ot_probe_lowedge_selfaffgw_faststep60_e1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_probe_lowedge_selfaffgw_faststep60_e1.json)
- launcher: [run_phase616_clean_ot_probe_round2.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase616_clean_ot_probe_round2.sh)

## Speed contract

This round is intentionally shorter than round 1:

- `batch_size = 16`
- `accumulation_steps = 1`
- `virtual_length_multiplier = 0.05`
- `stop_after_global_steps = 60`
- one epoch only
- one lightweight transfer eval at epoch end

The goal is not convergence. The goal is to answer:

1. does hybrid OT reduce hubness / cost variance relative to `self_affinity_gw`
2. does it lower structural leakage without collapsing style direction again

## Readout

Primary:

- transfer `CLIP-S`
- transfer `LPIPS`

White-box:

- `ot_target_gini`
- `ot_target_mass_entropy`
- `ot_target_max_mass`
- `ot_plan_entropy`
- `ot_cost_mean/var`
- `ot_structure_cost_mean/var`
- `base_structural_drift`
- `fiber_energy_ratio`
- `low_freq_leak`
- `target_base_shift`

Decision rule:

- if candidate beats or matches control on hubness / leakage while holding transfer quality, keep it
- if it only raises structure cost variance or slows the lane without cleaner probes, retire it quickly

## Results

Round completed on 2026-06-17. Both short probes ran to completion.

### Control: `self_affinity_gw`

Train closure:

- epoch wall: `89.0 s`
- `avg_optimizer_step_time_sec = 1.509`
- `ot_cost = 2.7054`
- `ot_target_gini = 0.059`
- `ot_target_max_mass = 0.353`
- `fiber_energy_ratio = 0.431`
- `low_freq_leak = 3.5362`

Transfer eval:

- `CLIP-S = 0.6698`
- `LPIPS = 0.7773`
- generated effective rank `= 1.0202`
- generated offdiag cosine `= 0.9867`

### Candidate: `lowedge_self_affinity_gw`

Train closure:

- epoch wall: `89.7 s`
- `avg_optimizer_step_time_sec = 1.521`
- `ot_cost = 2.6123`
- `ot_target_gini = 0.058`
- `ot_target_max_mass = 0.354`
- `fiber_energy_ratio = 0.447`
- `low_freq_leak = 3.4851`

Transfer eval:

- `CLIP-S = 0.6561`
- `LPIPS = 0.8089`
- generated effective rank `= 1.0148`
- generated offdiag cosine `= 0.9902`

### Matched delta: candidate minus control

White-box train probes:

- `ot_cost`: `-0.0931`
- `ot_target_gini`: `-0.0010`
- `fiber_energy_ratio`: `+0.0160`
- `low_freq_leak`: `-0.0511`

Transfer eval:

- `CLIP-S`: `-0.0137`
- `LPIPS`: `+0.0317`
- generated effective rank: `-0.0054`
- generated offdiag cosine: `+0.0035`

## Decision

Decision: `negative_for_promotion`

Interpretation:

- the hybrid OT descriptor does make the minibatch OT objective look slightly cleaner
- however the transfer surface gets worse on both style and structure
- the eval-side delta observability also regresses, which means the cleaner train OT signal did not survive into actual style diversity

Additional warning sign:

- candidate eval-side tokenizer routing became much sharper (`attn_effective_count` dropped from about `6.63` to `2.40`, while `attn_top1_mean` rose from about `0.52` to `0.81`)
- this suggests the hybrid descriptor is over-constraining matching and pushing the tokenizer toward a narrower routing regime

Follow-up implication:

- retain `self_affinity_gw` as the current best OT repair
- retire `lowedge_self_affinity_gw` from the active path
- the next 616 step should not keep stacking OT structure constraints; it should move to the next clean variable with `self_affinity_gw` fixed
