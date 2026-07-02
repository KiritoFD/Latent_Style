# Clean OT Probe Round 5: Tokenizer Entropy + Affinity

Date: 2026-06-17

## Purpose

The earlier OT audit answered one narrow question:

- replacing the retained latent-proxy `self_affinity_gw` with plain encoder or
  tokenizer feature-map affinity did **not** produce a better retained trade-off

But `design.md` asks for a slightly different structure signal:

- let OT see tokenizer-side **routing complexity**
- not only raw feature-map similarity

This round therefore keeps the retained 616 control fixed and changes only the
structure descriptor source again, this time to a tokenizer-native descriptor
built from:

- routing entropy
- routing sharpness
- routing entropy edge / high-pass structure
- routing self-affinity

That is closer to the intended “复杂图匹配复杂画，平坦图匹配平坦画” mechanism than the previous plain `aux_map` affinity probe.

## Matched contract

Fixed:

- `contract_family = phase616`
- `coupling_solver = sinkhorn_unbalanced`
- `coupling_cost_composition = appearance_plus_structure`
- retained target geometry:
  - `training_target_projection_mode = pure_vertical_flow`
- one epoch
- `stop_after_global_steps = 60`
- same lightweight transfer-only eval contract

Changed only:

- `bridge.coupling_structure_cost_mode`

## Candidates

Control:

- current retained latent-proxy `self_affinity_gw`

Candidate:

- `tokenizer_entropy_affinity_gw`

Interpretation:

- control asks OT to match on a hand-built latent topology proxy
- candidate asks OT to match on tokenizer routing complexity plus routing
  topology, which is the closer implementation of the 616 OT theory

## Hypothesis

If the earlier tokenizer feature-map probe was too weak because it only saw
`aux_map` affinity and ignored routing complexity, then this candidate should:

- reduce hubness or at least keep it flat
- keep or improve `low_freq_leak`
- improve transfer style relative to the retained proxy control

If it regresses again, the stronger conclusion becomes:

- the current model/tokenizer does not yet expose a tokenizer-native OT surface
  that beats the retained latent proxy

## Configs and launcher

- control:
  [phase616_clean_ot_probe_selfaffgw_mix_faststep60_e1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_probe_selfaffgw_mix_faststep60_e1.json)
- candidate:
  [phase616_clean_ot_probe_tokenentropy_selfaffgw_mix_faststep60_e1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_probe_tokenentropy_selfaffgw_mix_faststep60_e1.json)
- local runner:
  [run_phase616_clean_ot_probe_round5_tokenentropy.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase616_clean_ot_probe_round5_tokenentropy.sh)
- remote launcher:
  [launch_phase616_clean_ot_probe_round5_tokenentropy_remote.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_phase616_clean_ot_probe_round5_tokenentropy_remote.sh)

## Decision rule

- promote only if transfer improves without worsening `ot_target_gini`,
  `ot_target_max_mass`, `base_structural_drift`, or `low_freq_leak`
- if hubness improves but transfer stays flat or regresses, record the mechanism
  as `diagnostic_only`
- if it regresses cleanly against the matched control, treat that as stronger
  evidence that tokenizer-native OT is still not the active bottleneck

## Status

- `2026-06-17`: local implementation compiled and passed a routing-descriptor
  smoke test.
  - `losses.py` `py_compile` passed
  - `tokenizer_entropy_affinity_gw` parsed from config successfully
  - random-tensor smoke on `_routing_entropy_affinity_descriptor()` returned a
    finite descriptor tensor
- `2026-06-17 06:34` remote WSL launch started successfully on
  `100.115.18.62 / Ubuntu-26.04`.
  - launcher:
    [launch_phase616_clean_ot_probe_round5_tokenentropy_remote.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_phase616_clean_ot_probe_round5_tokenentropy_remote.sh)
  - active run root:
    `/home/xy/Latent_Style/SchrodingerBridge_phase616/SchrodingerBridge/exp/aaai2027_phase616_clean_ot_probe_tokenentropy_selfaffgw_mix_faststep60_e1`
  - sampled early remote health read:
    - GPU memory about `5.9 GiB`
    - GPU util about `96%`
    - GPU power about `128 W`
  - interpretation:
    - lane is compute-active, not stalled
    - current throughput is still below the preferred `9-11 GiB` occupancy band,
      so follow-up throughput tuning may still be needed if this mechanism shows
      positive signal
- `2026-06-17`: matched control/candidate run closed with authoritative train
  CSV + transfer summary + GPU monitor output.

## Results

### Control: `self_affinity_gw` mixed OT

- train closure:
  - `loss = 13.8841`
  - `flow = 0.4348`
  - `kinetic_energy = 12.9977`
  - `ot_cost = 2.7054`
  - `ot_target_gini = 0.05935`
  - `ot_target_max_mass = 0.35310`
  - `base_structural_drift = 0.20462`
  - `fiber_energy_ratio = 0.43874`
  - `low_freq_leak = 3.41906`
  - `target_base_shift = 0.01258`
  - `epoch_time_sec = 100.04`
  - `avg_optimizer_step_time_sec = 1.6956`
- transfer eval:
  - `transfer_clip_style = 0.667998`
  - `transfer_content_lpips = 0.763046`
  - `eval_wall_total_sec = 235.40`
  - `generation_sec = 127.93`
  - `vae_decode_sec = 60.82`
- generated-delta observability:
  - `effective_rank_mean = 1.0354`
  - `offdiag_cosine_mean = 0.9770`

### Candidate: `tokenizer_entropy_affinity_gw`

- train closure:
  - `loss = 14.1951`
  - `flow = 0.4552`
  - `kinetic_energy = 13.2713`
  - `ot_cost = 2.6621`
  - `ot_target_gini = 0.05692`
  - `ot_target_max_mass = 0.35305`
  - `base_structural_drift = 0.21521`
  - `fiber_energy_ratio = 0.43055`
  - `low_freq_leak = 3.51281`
  - `target_base_shift = 0.01282`
  - `epoch_time_sec = 121.58`
  - `avg_optimizer_step_time_sec = 2.0607`
- transfer eval:
  - `transfer_clip_style = 0.667469`
  - `transfer_content_lpips = 0.744723`
  - `eval_wall_total_sec = 239.93`
  - `generation_sec = 130.28`
  - `vae_decode_sec = 62.53`
- generated-delta observability:
  - `effective_rank_mean = 1.0658`
  - `offdiag_cosine_mean = 0.9572`

### Matched delta: candidate minus control

- transfer:
  - `clip_style = -0.000529`
  - `content_lpips = -0.018323`
- white-box OT probes:
  - `ot_target_gini = -0.002433`
  - `ot_target_max_mass = -0.000052`
- geometry probes:
  - `base_structural_drift = +0.010590`
  - `fiber_energy_ratio = -0.008192`
  - `low_freq_leak = +0.093746`
- throughput:
  - `epoch_time_sec = +21.54`
  - `avg_optimizer_step_time_sec = +0.3651`
  - `generation_sec = +2.35`
  - `vae_decode_sec = +1.71`

## Decision

Decision: `negative_for_promotion`

Reason:

- the candidate did validate that tokenizer routing complexity is a real
  structure signal
- it slightly reduced hubness and improved transfer LPIPS
- but it did **not** improve transfer style, which remains the priority metric
- it also worsened train-side structural drift / low-frequency leakage and made
  the lane materially slower

So the updated reading is:

- earlier 616 OT was **partly narrower than intended**
- correcting that narrowness toward tokenizer-native routing complexity did
  **not** uncover a hidden positive OT lane
- the current tokenizer-native OT surface is therefore best treated as
  `diagnostic_only`, not as the active control

## Closure note

What this round changes in our understanding:

1. The old retained `self_affinity_gw` control really is only a proxy
   descriptor, so the earlier implementation was not a perfect realization of
   the 616 OT theory.
2. However, the weak 616 style ceiling is not explained by that proxy mismatch
   alone, because a more faithful tokenizer-native descriptor still failed to
   beat the retained proxy under matched conditions.
3. The next bottleneck is more likely target geometry / stats-track /
   solver-track interaction than another small OT descriptor rewrite.
