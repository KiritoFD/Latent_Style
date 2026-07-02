# DualPath Spatial Sinkhorn Epoch1 Local Non-CLIP Read

Date: 2026-06-10

This note records the first complete local image-backed non-CLIP read for:

- `aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_spatial_sinkhorn_seed42_b8a2`

Scope:

- reviewed point:
  - `epoch_0001`
- local review axes:
  - `IntroStyle`
  - `DINO`

Artifacts:

- image-backed handoff:
  - [full_eval_imagebacked_bestfew_handoff_epoch1_only.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_sinkhorn_bestfew_localreview_20260609/full_eval_imagebacked_bestfew_handoff_epoch1_only.csv)
- local `IntroStyle`:
  - [full_eval_imagebacked_bestfew_introstyle_epoch1_only.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_sinkhorn_bestfew_localreview_20260609/full_eval_imagebacked_bestfew_introstyle_epoch1_only.csv)
  - [full_eval_imagebacked_bestfew_introstyle_epoch1_only.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_sinkhorn_bestfew_localreview_20260609/full_eval_imagebacked_bestfew_introstyle_epoch1_only.json)
  - [full_eval_imagebacked_bestfew_introstyle_epoch1_only.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_sinkhorn_bestfew_localreview_20260609/full_eval_imagebacked_bestfew_introstyle_epoch1_only.md)
- local `DINO`:
  - [full_eval_imagebacked_bestfew_dino_epoch1_only.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_sinkhorn_bestfew_localreview_20260609/full_eval_imagebacked_bestfew_dino_epoch1_only.csv)

## Results

`epoch_0001`:

- `IntroStyle target = 0.1104`
- `IntroStyle identity-target = 0.1538`
- `IntroStyle delta-IDT = -0.0434`
- `IntroStyle margin = -0.0483`
- `DINO = 0.02617`

Reference anchors:

- `LBM-Knee`
  - `IntroStyle target = 0.1073`
  - `IntroStyle delta-IDT = +0.0080`
  - `IntroStyle margin = -0.0373`
  - `DINO = 0.0217`
- `dualpath_spatialtexture epoch_0001`
  - `IntroStyle target = 0.1120`
  - `IntroStyle margin = -0.0503`
  - implied `delta-IDT` about `-0.0407`
- `Seedream`
  - `IntroStyle target = 0.1201`
  - `IntroStyle delta-IDT = +0.0209`
  - `IntroStyle margin = -0.0347`
  - `DINO = 0.0291`

## Interpretation

This first full non-CLIP read does not support a sinkhorn breakthrough.

Why:

1. `IntroStyle delta-IDT` is negative:
   - `0.1104 - 0.1538 = -0.0434`
2. `IntroStyle margin` remains clearly negative:
   - `-0.0483`
3. `DINO` is still weaker than `LBM-Knee`:
   - `0.02617` vs `0.0217`
4. relative to predecessor `dualpath_spatialtexture epoch_0001`, the local
   non-CLIP style read is effectively another near-tie:
   - similar target score
   - similar negative margin
   - still below the `LBM-Knee` `delta-IDT` read

So the current best reading is:

- `sinkhorn` did not collapse
- but it also did not reopen target-specific style
- on the first image-backed non-CLIP point, it still looks like the same
  conservative family with only tiny movement relative to predecessor

## Decision Impact

This strengthens the current theory-facing read:

- changing proximal routing alone is not yet enough
- the bottleneck is likely not just diffuse assignment
- the branch still lacks a mechanism that produces target-specific style motion
  under non-CLIP review

Current status:

- `near-negative / do not promote from epoch_0001 evidence alone`

The branch can still be revisited if later image-backed points such as
`epoch_0009` materially improve the same non-CLIP read.
