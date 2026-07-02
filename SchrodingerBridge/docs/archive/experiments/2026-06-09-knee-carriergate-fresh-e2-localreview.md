# `Knee e13 + CarrierGate Injection` Fresh `e2` Local Review

Date: 2026-06-09

Scope:

- local full review of the fresh retained point:
  - `epoch_0002`
- review stack:
  - `CLIP/LPIPS`
  - local `IntroStyle`
  - local `DINO`

Artifacts:

- local eval root:
  - [local_eval/knee_carriergate_fresh_e2_localreview](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/local_eval/knee_carriergate_fresh_e2_localreview)
- DINO scalar:
  - [knee_carriergate_fresh_e2_dino_20260609.txt](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/knee_carriergate_fresh_e2_dino_20260609.txt)
- comparison CSV:
  - [local_finalists_introstyle_dino_with_knee_carriergate_e2_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/local_finalists_introstyle_dino_with_knee_carriergate_e2_20260609.csv)
- comparison figure:
  - [local_finalists_introstyle_vs_dino_with_knee_carriergate_e2_20260609.png](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/local_finalists_introstyle_vs_dino_with_knee_carriergate_e2_20260609.png)

Measured read:

- transfer `clip_style = 0.6975`
- transfer `content_lpips = 0.4253`
- `IntroStyle target = 0.1092`
- `IntroStyle delta-IDT = -0.0393`
- `IntroStyle margin = -0.0449`
- `DINO structure = 0.0269`

Comparison against `LBM-Knee e13`:

- `LBM-Knee full750 IntroStyle target = 0.1073`
- `LBM-Knee full750 IntroStyle delta-IDT = +0.0080`
- `LBM-Knee full750 DINO = 0.0217`

Interpretation:

- this packet does raise absolute `IntroStyle target` slightly above `LBM-Knee`
- but the directionality read is still bad:
  - `delta-IDT` remains clearly negative
- structure also worsens:
  - `DINO` moves rightward from `0.0217` to `0.0269`

Decision:

- this is not yet a clear successor to `LBM-Knee`
- it is more promising than the `Hold4Mid`-anchored carrier-gate reopen
- but under the current evidence it is still below the threshold needed to replace `LBM-Knee` as the internal balanced point
