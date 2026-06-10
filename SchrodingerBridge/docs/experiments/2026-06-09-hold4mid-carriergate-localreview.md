# `Hold4Mid e8 + Carrier-Gate Injection` Local Review

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
  - [local_eval/carriergate_fresh_e2_localreview](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/local_eval/carriergate_fresh_e2_localreview)
- DINO scalar:
  - [carriergate_fresh_e2_dino_20260609.txt](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/carriergate_fresh_e2_dino_20260609.txt)

Measured read:

- transfer `clip_style = 0.6910`
- transfer `content_lpips = 0.5153`
- `IntroStyle target = 0.1037`
- `IntroStyle delta-IDT = -0.0396`
- `IntroStyle margin = -0.0466`
- `DINO structure = 0.0363`

Comparison against `LBM-Knee e13`:

- `LBM-Knee full750 IntroStyle target = 0.1073`
- `LBM-Knee full750 IntroStyle delta-IDT = +0.0080`
- `LBM-Knee full750 DINO = 0.0217`

Interpretation:

- this packet is not competitive with `LBM-Knee`
- style does not reopen enough:
  - `IntroStyle target` is lower than `LBM-Knee`
  - `delta-IDT` is actually negative
- structure is also worse:
  - `DINO` is materially larger than `LBM-Knee`

Decision:

- negative closure for the `Hold4Mid e8 + Carrier-Gate Injection` family
- do not spend more local or remote budget on this anchor family
- shift the same injection idea to the stronger `LBM-Knee e13` anchor instead
