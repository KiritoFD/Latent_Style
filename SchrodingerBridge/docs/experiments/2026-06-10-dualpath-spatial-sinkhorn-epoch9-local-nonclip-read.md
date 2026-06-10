# DualPath Spatial Sinkhorn Epoch9 Local Non-CLIP Read

Date: 2026-06-10

This note records the second complete local image-backed non-CLIP read for:

- `aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_spatial_sinkhorn_seed42_b8a2`

Scope:

- reviewed point:
  - `epoch_0009`
- local review axes:
  - `IntroStyle`
  - `DINO`

Artifacts:

- image-backed handoff:
  - [full_eval_imagebacked_bestfew_handoff_epoch9_only.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_sinkhorn_bestfew_localreview_20260609/full_eval_imagebacked_bestfew_handoff_epoch9_only.csv)
- local `IntroStyle`:
  - [full_eval_imagebacked_bestfew_introstyle_epoch9_only.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_sinkhorn_bestfew_localreview_20260609/full_eval_imagebacked_bestfew_introstyle_epoch9_only.csv)
  - [full_eval_imagebacked_bestfew_introstyle_epoch9_only.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_sinkhorn_bestfew_localreview_20260609/full_eval_imagebacked_bestfew_introstyle_epoch9_only.json)
  - [full_eval_imagebacked_bestfew_introstyle_epoch9_only.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_sinkhorn_bestfew_localreview_20260609/full_eval_imagebacked_bestfew_introstyle_epoch9_only.md)
- local `DINO`:
  - [full_eval_imagebacked_bestfew_dino_epoch9_only.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_sinkhorn_bestfew_localreview_20260609/full_eval_imagebacked_bestfew_dino_epoch9_only.csv)

## Results

`epoch_0009`:

- `IntroStyle target = 0.1056`
- `IntroStyle identity-target = 0.1382`
- `IntroStyle delta-IDT = -0.0326`
- `IntroStyle margin = -0.0465`
- `DINO = 0.02741`

Reference anchors:

- `sinkhorn epoch_0001`
  - `IntroStyle target = 0.1104`
  - `IntroStyle delta-IDT = -0.0434`
  - `IntroStyle margin = -0.0483`
  - `DINO = 0.02617`
- `LBM-Knee`
  - `IntroStyle target = 0.1073`
  - `IntroStyle delta-IDT = +0.0080`
  - `IntroStyle margin = -0.0373`
  - `DINO = 0.0217`
- `Seedream`
  - `IntroStyle target = 0.1201`
  - `IntroStyle delta-IDT = +0.0209`
  - `IntroStyle margin = -0.0347`
  - `DINO = 0.0291`

## Interpretation

This second full non-CLIP read still does not rescue the sinkhorn family.

Why:

1. `IntroStyle delta-IDT` remains negative:
   - `0.1056 - 0.1382 = -0.0326`
2. `IntroStyle margin` remains clearly negative:
   - `-0.0465`
3. `DINO` is worse than `epoch_0001`:
   - `0.02741` vs `0.02617`
4. relative to `epoch_0001`, the later point buys:
   - slightly less negative `delta-IDT`
   - slightly less negative margin
   - but lower absolute target score
   - and worse structure

So the later point changes the story only weakly:

- it is not a collapse
- but it is still not a target-specific style recovery
- and it still does not beat the current `LBM-Knee` balanced anchor

## Decision Impact

Combined with `epoch_0001`, the sinkhorn family now has two image-backed
non-CLIP points and still shows the same pattern:

- style motion is weak or negative under `IntroStyle delta-IDT`
- specificity margin stays negative
- structure remains worse than `LBM-Knee`

Current status:

- `negative-to-mixed / do not promote`

This is strong enough to stop treating sinkhorn routing as a likely rescue
family unless a later point outside the current reviewed set shows a genuinely
different non-CLIP read.
