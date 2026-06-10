# Knee Spatial Carrier BestFew Local Review

Date: 2026-06-09

Scope:

- local low-VRAM review on the three remote handoff points from:
  - `aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_seed42_b8a2`
- reviewed points:
  - `epoch_0003`
  - `epoch_0008`
  - `epoch_0012`
- local review axes:
  - `IntroStyle`
  - `DINO`

Artifacts:

- handoff:
  - [knee_spatial_carriergate_bodydecoder_bestfew_handoff_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/knee_spatial_carriergate_bodydecoder_bestfew_handoff_20260609.csv)
- local `IntroStyle`:
  - [knee_spatial_carriergate_bodydecoder_bestfew_introstyle_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/knee_spatial_carriergate_bodydecoder_bestfew_introstyle_20260609.csv)
- local `DINO`:
  - [knee_spatial_carriergate_bodydecoder_bestfew_dino_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/knee_spatial_carriergate_bodydecoder_bestfew_dino_20260609.csv)
- merged review table:
  - [knee_spatial_carriergate_bodydecoder_bestfew_review_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/knee_spatial_carriergate_bodydecoder_bestfew_review_20260609.csv)

## Local low-VRAM execution note

- local `IntroStyle` was rerun in safe mode:
  - `batch_size = 1`
  - `ensemble_size = 1`
- local GPU memory stayed around:
  - `1.8 to 1.9 / 8.2 GiB`
- the failure seen earlier was not VRAM:
  - it was filename mojibake after remote tar extraction
- local path matching was repaired in:
  - [eval_introstyle_probe.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/eval_introstyle_probe.py)

## Results

Reviewed points:

- `epoch_0003`
  - `IntroStyle target = 0.1098`
  - `IntroStyle delta-IDT = +0.0106`
  - `IntroStyle margin = -0.0434`
  - `DINO = 0.0283`
- `epoch_0008`
  - `IntroStyle target = 0.1108`
  - `IntroStyle delta-IDT = +0.0116`
  - `IntroStyle margin = -0.0399`
  - `DINO = 0.0284`
- `epoch_0012`
  - `IntroStyle target = 0.1100`
  - `IntroStyle delta-IDT = +0.0107`
  - `IntroStyle margin = -0.0431`
  - `DINO = 0.0284`

Reference anchors:

- `LBM-Knee`
  - `IntroStyle target = 0.1073`
  - `IntroStyle delta-IDT = +0.0080`
  - `IntroStyle margin = -0.0373`
  - `DINO = 0.0217`
- `LBM-PS-v2`
  - `IntroStyle target = 0.0993`
  - `IntroStyle delta-IDT = +0.0001`
  - `IntroStyle margin = -0.0326`
  - `DINO = 0.0303`
- `Seedream`
  - `IntroStyle target = 0.1201`
  - `IntroStyle delta-IDT = +0.0209`
  - `IntroStyle margin = -0.0347`
  - `DINO = 0.0291`

## Interpretation

- the spatial carrier line does beat `LBM-Knee` on:
  - absolute `IntroStyle target`
  - and `IntroStyle delta-IDT`
- but it loses clearly on:
  - `DINO`
  - and also weakens the specificity margin relative to `LBM-Knee`

This means:

- the family does create some extra style motion
- but it buys that motion by moving too far right on the structure axis
- and the style it adds is still not specific enough to justify that structure cost

More concretely:

- it moves toward the `Seedream` side of the `DINO` axis
- without moving enough toward `Seedream` on the `IntroStyle` axis

So the line is not a true promoted win.

## Decision

Status:

- `near-negative / do not promote`

Reason:

- not strong enough to replace `LBM-Knee` as the internal balanced point
- not strong enough to become the new style-forward line either
- therefore it should not consume more remote budget as the main family

## Next direction

The next meaningful move should not be another carrier-gate micro-variant.

Prefer:

- a stronger target-specific spatial branch with explicit structure leash
- or a new late style-recovery head
- or a diagnosis-first packet that explains what `Seedream` is still doing better:
  - target-style hierarchy
  - texture specificity
  - clean structure under stronger stylization
