# Current Round Read: SpatialTexture

Date: 2026-06-09

This note consolidates the current strongest read on the `dualpath_spatialtexture` round using:

- early `CLIP / LPIPS` eval rows
- landed `IntroStyle` bestfew probe
- direct blind local `VLM`
- existing `Knee / Seedream` anchors

## 1. Early curve read

Source:

- [dualpath_spatial_fresh_curve_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_fresh_curve_20260609.csv)

Current curve shape:

- transfer style stays tightly bounded around:
  - `0.6916 to 0.6929`
- `LPIPS` rises from about:
  - `0.401` to `0.440`

Immediate implication:

- the line is stable
- but it does not currently show a style-opening trajectory

## 2. IntroStyle bestfew read

Source:

- [dualpath_spatial_introstyle_bestfew_probe_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_spatial_introstyle_bestfew_probe_20260609.csv)

Current bestfew points:

- `epoch_0001`
  - target `0.11198`
  - best non-target `0.16228`
  - margin `-0.05031`
  - implied delta-idt `-0.04069`
- `epoch_0012`
  - target `0.10755`
  - best non-target `0.15429`
  - margin `-0.04673`
  - implied delta-idt `-0.03441`

Anchor comparison:

- `LBM-Knee e13`
  - target `0.10727`
  - delta-idt `+0.00804`
  - margin `-0.03728`
- `Seedream`
  - target `0.12009`
  - delta-idt `+0.02087`
  - margin `-0.03469`

Immediate implication:

- spatialtexture can keep a decent absolute target score
- but it remains on the wrong side of target-specificity
- it does not beat `Knee` on directional style read

## 3. Direct blind VLM read

Source:

- [vlm_qedgee01_vs_dualpathe01_vs_seedream_20260609.method_summary.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_qedgee01_vs_dualpathe01_vs_seedream_20260609.method_summary.csv)
- [qedge_vs_dualpath_interim_board_20260609.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/qedge_vs_dualpath_interim_board_20260609.md)

Current read:

- completed cases:
  - `621`
- overall wins:
  - `Seedream = 600 / 613`
  - `DualPath e01 = 13 / 613`
  - `QEdgePattn e01 = 0 / 613`

Current mean local scores:

- `DualPath e01`
  - style `2.173`
  - structure `3.515`
  - artifact `2.620`
- `QEdgePattn e01`
  - style `1.980`
  - structure `3.357`
  - artifact `2.378`

Immediate implication:

- `DualPath e01` continues to read cleaner than `QEdgePattn e01` under blind perceptual judgment
- but neither family is remotely close to displacing `Seedream`

## 4. Current combined interpretation

At this point, the current branch-capacity round reads as:

1. `dualpath_texture`
   - safer basin
2. `dualpath_spatialtexture`
   - same basin, traced more fully
3. `IntroStyle`
   - still not target-specific enough
4. blind `VLM`
   - still far below `Seedream`

So the current best working conclusion is:

- `capacity alone is not enough`
- `cleaner` or `more expressive` is not the same as `more target-specific`
- the next branch idea likely needs explicit target-style pressure, not just broader late-branch capacity
