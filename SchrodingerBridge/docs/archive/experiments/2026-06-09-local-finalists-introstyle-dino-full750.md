# Local Finalists IntroStyle-DINO Full750

Date: 2026-06-09

Scope:

- local full750 reevaluation on the current finalists
- style axis:
  - `IntroStyle target score`
- structure axis:
  - `DINO structure distance`

Artifacts:

- merged CSV:
  - [local_finalists_introstyle_dino_full750_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/local_finalists_introstyle_dino_full750_20260609.csv)
- figure:
  - [local_finalists_introstyle_vs_dino_full750_20260609.png](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/local_finalists_introstyle_vs_dino_full750_20260609.png)
- source tables:
  - [local_finalists_introstyle_full750_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/local_finalists_introstyle_full750_20260609.csv)
  - [local_finalists_dino_full750_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/local_finalists_dino_full750_20260609.csv)

Points:

- `IDT`
- `LBM-K e1`
- `LBM-Knee e13`
- `LBM-PS-v2 e13`
- `SaMST e15`
- `Seedream-4.5`

Key read:

- `Seedream-4.5` is still the strongest current style point on the full750 `IntroStyle` target score:
  - `0.1201`
- `LBM-K` and `LBM-Knee` are almost tied on full750 `IntroStyle`:
  - `LBM-K = 0.1077`
  - `LBM-Knee = 0.1073`
- `LBM-Knee` is the cleaner structure point than `LBM-K`:
  - `LBM-Knee DINO = 0.0217`
  - `LBM-K DINO = 0.0251`
- `LBM-PS-v2` remains weaker on both axes than the desired tradeoff:
  - `IntroStyle = 0.0993`
  - `DINO = 0.0303`
- `SaMST e15` is still unusually structure-faithful:
  - `DINO = 0.0172`
  - but its full750 `IntroStyle` target score remains below the best current LBM and below `Seedream`

Interpretation:

- the full750 read strengthens the earlier smoke conclusion:
  - `LBM-Knee` is still the most credible balanced internal point
  - `LBM-PS-v2` looks more like a style-over-structure tradeoff than a cleaner target-style attribution point
  - `Seedream` remains the strongest current external style ceiling

Paper-facing implication:

- if we need one internal point to compare against `Seedream` under `IntroStyle + DINO`, the current answer is still `LBM-Knee`
- the new mechanism burden is therefore:
  - beat `LBM-Knee` on `IntroStyle`
  - without drifting rightward on the DINO axis into `LBM-PS-v2` territory
