# `Knee e13 + CarrierGate Injection` Fresh `e12` Local Review

Date: 2026-06-09

Scope:

- local full review of the fresh retained point:
  - `epoch_0012`
- review stack:
  - `CLIP/LPIPS`
  - local `IntroStyle`
  - local `DINO`

Artifacts:

- local eval root:
  - [local_eval/knee_carriergate_fresh_e12_localreview](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/local_eval/knee_carriergate_fresh_e12_localreview)
- DINO scalar:
  - [knee_carriergate_fresh_e12_dino_20260609.txt](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/knee_carriergate_fresh_e12_dino_20260609.txt)
- comparison CSV:
  - [local_finalists_introstyle_dino_with_knee_carriergate_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/local_finalists_introstyle_dino_with_knee_carriergate_20260609.csv)
- comparison figure:
  - [local_finalists_introstyle_vs_dino_with_knee_carriergate_20260609.png](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/local_finalists_introstyle_vs_dino_with_knee_carriergate_20260609.png)

Measured read:

- transfer `clip_style = 0.6946`
- transfer `content_lpips = 0.4244`
- `IntroStyle target = 0.1074`
- `IntroStyle delta-IDT = -0.0419`
- `IntroStyle margin = -0.0459`
- `DINO structure = 0.0268`

Comparison against `LBM-Knee e13`:

- `LBM-Knee full750 IntroStyle target = 0.1073`
- `LBM-Knee full750 IntroStyle delta-IDT = +0.0080`
- `LBM-Knee full750 DINO = 0.0217`

Interpretation:

- by `e12`, the packet still does not cross the promotion threshold
- it remains in the same regime as `e2`:
  - slightly higher absolute `IntroStyle target`
  - but still negative `delta-IDT`
  - and still worse `DINO` than `LBM-Knee`

Decision:

- this line does not currently justify replacing `LBM-Knee`
- the evidence now suggests that:
  - `plain carrier_gate` from the `Knee` basin is not enough
  - and the next real improvement likely needs a stronger target-specific spatial mechanism or a different branch family
