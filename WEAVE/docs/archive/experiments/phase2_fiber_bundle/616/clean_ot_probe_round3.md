# Clean OT Probe Round 3

Date: 2026-06-17

## Purpose

Round 2 established a clean local optimum for OT structure descriptors:

- retain `self_affinity_gw`
- retire `lowedge_self_affinity_gw`

The next 616 OT question is not "which structure descriptor?" but "should OT still see pointwise appearance distance at all?"

This comes directly from the 616 design diagnosis:

- current pointwise latent `pairwise_cost` mixes semantics and photometry
- that metric mismatch encourages hub targets and mean-like matches
- the stronger OT hypothesis is to let matching happen purely in retained topology space

## New OT mechanism

New candidate: `structure_only` cost composition over retained `self_affinity_gw`

Control:

- current retained mixed OT cost
- normalized appearance term + normalized structure term

Candidate:

- keep the same retained `self_affinity_gw` descriptor
- delete the appearance term from coupling
- keep unbalanced Sinkhorn fixed
- keep `pure_vertical_flow` fixed

This isolates one question:

1. if we remove pointwise appearance cost completely, do hubness and leakage improve for the right reason
2. or does transfer quality collapse because the OT match becomes too coarse

## New probes

This round also upgrades OT observability for unbalanced matching:

- `ot_raw_total_mass`
- `ot_source_mass_mean/min/max`
- `ot_source_mass_entropy`
- `ot_source_marginal_l1`
- `ot_source_truncation`
- `ot_target_marginal_l1`
- `ot_target_truncation`

These are meant to separate "cleaner coupling geometry" from "the solver simply dropped mass and looked cleaner."

## Matched configs

- control: [phase616_clean_ot_probe_selfaffgw_mix_faststep60_e1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_probe_selfaffgw_mix_faststep60_e1.json)
- candidate: [phase616_clean_ot_probe_selfaffgw_structureonly_faststep60_e1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_probe_selfaffgw_structureonly_faststep60_e1.json)
- launcher: [run_phase616_clean_ot_probe_round3.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase616_clean_ot_probe_round3.sh)

## Speed contract

- `batch_size = 16`
- `accumulation_steps = 1`
- `virtual_length_multiplier = 0.05`
- `stop_after_global_steps = 60`
- one epoch only
- one lightweight transfer eval at epoch end

## Decision rule

- keep the candidate only if transfer does not materially regress and the new truncation probes do not show that cleanliness came from mass dropping
- if structure-only improves hubness but transfer degrades sharply, record it as diagnostic-only and do not promote it
- if both transfer and white-box probes regress, retire it immediately

## Results

Round completed on 2026-06-17. Both short probes ran to closure.

### Control: `self_affinity_gw` mixed cost

Train closure:

- epoch wall: `88.23 s`
- `avg_optimizer_step_time_sec = 1.495`
- `ot_cost = 2.7054`
- `ot_target_gini = 0.0594`
- `ot_target_max_mass = 0.3531`
- `ot_raw_total_mass = 0.3528`
- `ot_source_truncation = 0.6472`
- `ot_target_truncation = 0.6472`
- `fiber_energy_ratio = 0.4274`
- `low_freq_leak = 3.5281`

Transfer eval:

- `CLIP-S = 0.6751`
- `LPIPS = 0.7145`
- generated effective rank `= 1.0313`
- generated offdiag cosine `= 0.9800`

### Candidate: `self_affinity_gw` structure-only

Train closure:

- epoch wall: `86.90 s`
- `avg_optimizer_step_time_sec = 1.473`
- `ot_cost = 2.0086`
- `ot_target_gini = 0.1140`
- `ot_target_max_mass = 0.3881`
- `ot_raw_total_mass = 0.5179`
- `ot_source_truncation = 0.4908`
- `ot_target_truncation = 0.4854`
- `fiber_energy_ratio = 0.4198`
- `low_freq_leak = 3.5900`

Transfer eval:

- `CLIP-S = 0.6596`
- `LPIPS = 0.7236`
- generated effective rank `= 1.0207`
- generated offdiag cosine `= 0.9863`

### Matched delta: candidate minus control

White-box train probes:

- `ot_cost`: `-0.6969`
- `ot_target_gini`: `+0.0547`
- `ot_target_max_mass`: `+0.0350`
- `ot_raw_total_mass`: `+0.1651`
- `ot_source_truncation`: `-0.1564`
- `ot_target_truncation`: `-0.1618`
- `fiber_energy_ratio`: `-0.0076`
- `low_freq_leak`: `+0.0619`

Transfer eval:

- `CLIP-S`: `-0.0155`
- `LPIPS`: `+0.0091`
- generated effective rank: `-0.0105`
- generated offdiag cosine: `+0.0063`

## Decision

Decision: `negative_for_promotion`

Interpretation:

- removing appearance cost entirely does lower the normalized OT objective, but that is not a meaningful win
- the actual coupling gets more hub-like, not less: `ot_target_gini` and `ot_target_max_mass` both worsen
- the truncation probes do shrink, but they do not buy cleaner transfer; instead transfer regresses on both style and LPIPS
- eval-side delta diversity also narrows again, which is exactly the failure mode the 616 metric-mismatch diagnosis was meant to avoid

Follow-up implication:

- retain the mixed `appearance_plus_structure` self-affinity OT as the current best OT repair
- retire `structure_only` coupling from the active path
- the next 616 step should move to the next clean mechanism or a probe refinement, not keep pushing pure structure-only OT
