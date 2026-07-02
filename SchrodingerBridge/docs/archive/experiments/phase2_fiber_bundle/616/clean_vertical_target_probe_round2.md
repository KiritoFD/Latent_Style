# Clean Vertical Target Probe Round 2

Date: 2026-06-17

## Purpose

After the fast OT round:

- `self_affinity_gw` is retained
- `lowedge_self_affinity_gw` is retired

The next clean variable is now the base/fiber split operator itself.

Everything else stays fixed:

- `contract_family = phase616`
- `solver_family = solver_i2sb`
- endpoint I2SB objective
- `coupling_solver = sinkhorn_unbalanced`
- `coupling_structure_cost_mode = self_affinity_gw`
- fast one-epoch / 60-step probe contract

## Matched configs

- control: [phase616_clean_vertical_target_selfaffgw_faststep60_e1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_vertical_target_selfaffgw_faststep60_e1.json)
- candidate: [phase616_clean_vertical_target_selfaffgw_wavelet_faststep60_e1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_vertical_target_selfaffgw_wavelet_faststep60_e1.json)
- launcher: [run_phase616_clean_vertical_target_probe_round2.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase616_clean_vertical_target_probe_round2.sh)

## Hypothesis

`docs/616/design.md` explicitly argues that avgpool `5x5` is too crude a separator and that a wavelet-like downsample/upsample lowpass should give a cleaner base/fiber split.

This round therefore asks one narrow question:

- with OT fixed, does `pure_vertical_flow_wavelet` reduce structure leakage and/or improve transfer over the current `pure_vertical_flow` split?

## Decision rule

- if wavelet lowers `base_structural_drift` and `low_freq_leak` while matching or improving transfer, retain it
- if wavelet only improves white-box probes but transfer regresses, treat it as diagnostic-only
- if wavelet loses on both probes and transfer, retire it immediately

## Status

This lane was stopped on 2026-06-17 before candidate eval closure.

Reason:

- the user explicitly redirected the queue back to OT repair and probe-first work
- this vertical split comparison was slower than needed for the current decision point
- OT still had a cleaner unresolved single-variable hypothesis: remove pointwise appearance cost from coupling

Interpretation:

- keep the control-side numbers already collected as diagnostic context
- do not promote or retire the wavelet split from this stopped lane alone
- resume vertical-target work only after the new OT composition probe closes
