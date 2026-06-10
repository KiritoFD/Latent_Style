# DualPath Spatial BestFew IntroStyle Read

Date: 2026-06-09

This note records the first landed `IntroStyle` bestfew probe for:

- `aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_spatial_seed42_b8a2`

Artifacts:

- [dualpath_spatial_introstyle_bestfew_probe_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_introstyle_bestfew_probe_20260609.csv)
- [dualpath_spatial_introstyle_bestfew_probe_20260609.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_introstyle_bestfew_probe_20260609.json)

## Landed bestfew points

- `epoch_0001`
  - target score `0.11198`
  - best non-target `0.16228`
  - style margin `-0.05031`
- `epoch_0012`
  - target score `0.10755`
  - best non-target `0.15429`
  - style margin `-0.04673`

## Immediate read

This is the first real non-CLIP style read for the spatialtexture branch.

It does **not** show a target-specific style recovery breakthrough.

Why:

1. both bestfew points still have negative `IntroStyle` style margin
2. `epoch_0012` is only slightly less negative than `epoch_0001`
3. the absolute target score is not enough by itself to overturn the current curve read

So the current non-CLIP interpretation is:

- `spatialtexture` may keep a reasonably high absolute target score on some points
- but it is still not target-specific enough
- and it still does not look like a clear style-ceiling rescue

## Combined branch read

Taken together, the current evidence says:

- `CLIP/LPIPS` early curve:
  - conservative basin
- `IntroStyle` bestfew:
  - target specificity still weak
- local blind `VLM`:
  - internal pair still far below `Seedream`

This is enough to state a current working read:

- `dualpath_spatialtexture` remains a negative-to-mixed closure
  for the claim that more late-branch spatial expressivity alone solves the style ceiling
