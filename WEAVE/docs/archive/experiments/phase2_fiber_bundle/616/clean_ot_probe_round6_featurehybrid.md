# Clean OT Probe Round 6: Internal Feature Hybrid Geometry

Date: 2026-06-17

## Purpose

Round 4 already answered one narrow audit question:

- raw-latent proxy `self_affinity_gw` was not a fake implementation
- pure affinity on internal feature maps did not beat the retained control

But that still leaves a cleaner 616 question open:

- was the failure caused by using the wrong surface
- or by using an internal surface that was too narrow

The prior internal-feature candidates used only self-affinity vectors:

- `encoder_self_affinity_gw`
- `tokenizer_aux_self_affinity_gw`

That made them closer to the theory than the raw-latent proxy, but it also
deleted the low/edge/high magnitude cues that made the retained latent proxy
stable.

Round 6 adds a reusable middle ground:

- keep the descriptor source inside model-produced feature maps
- but restore the retained hybrid recipe of statistics plus affinity

This is the smallest next OT step that is both more faithful to the 616
metric-space diagnosis and less likely to fail just because the descriptor is
too thin.

## Matched contract

Fixed:

- `contract_family = phase616`
- `coupling_solver = sinkhorn_unbalanced`
- `coupling_cost_composition = appearance_plus_structure`
- `training_target_projection_mode = pure_vertical_flow`
- one epoch
- `stop_after_global_steps = 60`
- same transfer-only eval contract

## Candidates

- control: retained latent-proxy `self_affinity_gw`
- candidate A: `encoder_hybrid_affinity_gw`
- candidate B: `tokenizer_aux_hybrid_affinity_gw`

Interpretation:

- `encoder_hybrid_affinity_gw` asks whether encoder/down feature maps become
  useful once OT sees both internal feature-map statistics and self-affinity
- `tokenizer_aux_hybrid_affinity_gw` asks the same question on the content-side
  tokenizer routing surface

## Why this probe exists

The 616 design note diagnoses a metric-space mismatch:

- raw latent appearance is too Euclidean and too content-tied
- pure internal affinity may be too abstract and too underconstrained

So the natural next repair is not another brand-new mechanism. It is a
descriptor cleanup:

- internal feature source
- hybrid structure summary
- same OT solver
- same target geometry

That keeps the mechanism delta interpretable.

## Matched configs

- control:
  [phase616_clean_ot_probe_selfaffgw_mix_faststep60_e1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_probe_selfaffgw_mix_faststep60_e1.json)
- candidate A:
  [phase616_clean_ot_probe_encoder_hybridaffgw_mix_faststep60_e1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_probe_encoder_hybridaffgw_mix_faststep60_e1.json)
- candidate B:
  [phase616_clean_ot_probe_tokenaux_hybridaffgw_mix_faststep60_e1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_probe_tokenaux_hybridaffgw_mix_faststep60_e1.json)
- launcher:
  [run_phase616_clean_ot_probe_round6_featurehybrid.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase616_clean_ot_probe_round6_featurehybrid.sh)

## Closure rule

- retain a hybrid internal-feature candidate only if it improves transfer over
  the retained control without worsening the white-box leakage and hubness
  probes materially
- if a candidate improves style but blows up LPIPS again, keep it as negative
  evidence about the feature surface rather than as OT progress
- if both candidates are flat or worse, keep the retained proxy control and
  stop pushing OT descriptor complexity further until another mechanism changes
  the backbone geometry

## Status

Launched on the remote 616 WSL lane on 2026-06-17.

Current known state at launch:

- prelaunch guard passed with `prelaunch_gpu_memory_used_mib = 345`
- first health check passed with `health_gpu_memory_used_mib = 669`
- dataset, pairing cache, and model initialization completed cleanly

### Control progress: `self_affinity_gw_mix`

The reused control lane completed first and the launcher advanced to
`encoder_hybrid_affinity_gw`.

Control train closure:

- epoch wall: `103.8 s`
- `avg_optimizer_step_time_sec = 1.760 s`
- `ot_cost = 2.7054`
- `ot_target_gini = 0.059`
- `ot_target_max_mass = 0.353`
- `base_structural_drift = 0.2068`
- `fiber_energy_ratio = 0.439`
- `low_freq_leak = 3.4645`
- mean GPU util / peak GPU util: `53.2% / 100.0%`
- mean VRAM / peak VRAM: `4.86 GB / 6.46 GB`
- mean power / peak power: `84.3 W / 134.4 W`

Control transfer eval:

- `CLIP-S = 0.6680`
- `LPIPS = 0.7630`
- eval wall `= 235.40 s`

Important hygiene note:

- this first round-6 control uses the retained control `save_dir` instead of a
  fresh isolated root
- that does not invalidate the runtime or train closure, but it weakens the
  artifact boundary for the eval summary
- because of that, the next bridge-noise stage has already been prepared in an
  authoritative isolated form with fresh save roots

Remaining candidate status:

- `encoder_hybrid_affinity_gw`: train closure complete, eval running
- `tokenizer_aux_hybrid_affinity_gw`: queued behind it

### Candidate A progress: `encoder_hybrid_affinity_gw`

Train closure:

- epoch wall: `106.5 s`
- `avg_optimizer_step_time_sec = 1.806 s`
- `ot_cost = 2.5134`
- `ot_target_gini = 0.064`
- `ot_target_max_mass = 0.358`
- `base_structural_drift = 0.2146`
- `fiber_energy_ratio = 0.450`
- `low_freq_leak = 3.4602`
- mean GPU util / peak GPU util: `54.6% / 96.0%`
- mean VRAM / peak VRAM: `5.07 GB / 6.46 GB`
- mean power / peak power: `93.2 W / 137.1 W`

Train-side delta vs. the current control:

- `ot_cost`: better
- `fiber_energy_ratio`: slightly better
- `ot_target_gini`: worse
- `ot_target_max_mass`: worse
- `base_structural_drift`: worse
- wall time: slightly worse

Interpretation before eval:

- the hybrid encoder surface is stronger than the retained proxy in the narrow
  sense of reducing OT cost
- but it is not cleaner on the white-box structural probes
- if the eval side also regresses LPIPS, that will be strong evidence that OT
  descriptor complexity is no longer the main blocker

Transfer eval:

- `CLIP-S = 0.6755`
- `LPIPS = 0.7406`
- eval wall `= 242.95 s`

Matched delta vs. current control:

- `CLIP-S`: `+0.0075`
- `LPIPS`: `-0.0224`
- `ot_cost`: better
- `fiber_energy_ratio`: slightly better
- `ot_target_gini`: worse
- `ot_target_max_mass`: worse
- `target_base_shift`: worse
- epoch wall: `+2.7 s`

Current interpretation:

- this is the first internal-feature OT candidate that is positive on transfer
  as well as train OT cost
- but it is not yet a promotion-quality result because the control side of
  round 6 reused an old save root
- the right next step is not to over-interpret it; it is to run an isolated
  authoritative control-vs-candidate rerun
