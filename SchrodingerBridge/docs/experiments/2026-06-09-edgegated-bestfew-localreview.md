# EdgeGated BestFew Local Review

Date: 2026-06-09

Scope:

- local low-VRAM review on the current best-few fresh-localreview packet from:
  - `aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_edgegated_seed42_b8a2`
- reviewed points:
  - `epoch_0001`
  - `epoch_0003`
- local review axes:
  - `IntroStyle`
  - `DINO`

Artifacts:

- local `IntroStyle`:
  - [edgegated_bestfew_fresh_introstyle_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/edgegated_bestfew_fresh_introstyle_20260609.csv)
- local `DINO`:
  - [edgegated_bestfew_fresh_dino_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/edgegated_bestfew_fresh_dino_20260609.csv)
- merged review:
  - [edgegated_bestfew_fresh_review_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/edgegated_bestfew_fresh_review_20260609.csv)

## Results

- `epoch_0001`
  - `IntroStyle target = 0.1088`
  - `IntroStyle delta-IDT = +0.0096`
  - `IntroStyle margin = -0.0444`
  - `DINO = 0.0281`
- `epoch_0003`
  - `IntroStyle target = 0.1085`
  - `IntroStyle delta-IDT = +0.0093`
  - `IntroStyle margin = -0.0442`
  - `DINO = 0.0283`

Reference anchors:

- `LBM-Knee`
  - `IntroStyle target = 0.1073`
  - `IntroStyle delta-IDT = +0.0080`
  - `IntroStyle margin = -0.0373`
  - `DINO = 0.0217`
- `Knee + spatial carrier body+decoder`
  - best reviewed point:
    - `IntroStyle target = 0.1108`
    - `IntroStyle delta-IDT = +0.0116`
    - `IntroStyle margin = -0.0399`
    - `DINO = 0.0284`

## Interpretation

- the edge-gated leash does recover a small amount of structure relative to the plain spatial-carrier line
  - `DINO` improves slightly from about `0.0284` to about `0.0281`
- but that small structure recovery is bought by giving back part of the style gain
  - `IntroStyle target` and `delta-IDT` are both lower than the plain spatial-carrier line

So the current read is:

- better than the plain spatial-carrier line as a theory probe
- but still not good enough as a promoted result
- it still sits clearly to the worse side of `LBM-Knee` on `DINO`
- and it still does not create enough target-style lift to justify that structure cost

## Decision

Status:

- `near-negative / theory-positive but not promotable`

Meaning:

- the mechanism taught us something useful:
  - explicit structure leash on top of spatial carrier matters
- but the current implementation does not produce a better paper-facing point

## Next direction

Most likely next step:

- a stronger target-specific spatial branch
- plus a more selective structure leash
- or a late style-recovery head that does not force the carrier branch itself to do all the work
