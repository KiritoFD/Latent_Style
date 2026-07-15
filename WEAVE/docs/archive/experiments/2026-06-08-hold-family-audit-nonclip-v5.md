# Hold-Family Audit Non-CLIP Probe v5

Date: 2026-06-08

Scope:

- run a Distinct5 non-CLIP style-classifier audit on the current hold-family audit points
- compare them against the existing paper-facing reference rows:
  - `IDT`
  - `LBM-Knee e13`
  - `LBM-PS-v2 e13`
  - `Seedream-4.5`

Audited points:

- `Hold4Mid e8`
- `Hold4SlowMid e12`
- `Hold4TwoStage e2`

Method:

- images were regenerated with `save_generated_images=true` using a direct Python wrapper around [run_evaluation.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/utils/run_evaluation.py)
- the existing Distinct5 ConvNeXt style classifier was then run locally on those generated images

## Key results

Reference non-CLIP transfer results:

| point | target acc | target prob | source prob | target-source margin |
| --- | ---: | ---: | ---: | ---: |
| `IDT` | `0.0100` | `0.0168` | `0.9329` | `-0.9161` |
| `LBM-Knee e13` | `0.2367` | `0.2123` | `0.5633` | `-0.3511` |
| `LBM-PS-v2 e13` | `0.2717` | `0.2696` | `0.3064` | `-0.0368` |
| `Seedream-4.5` | `0.3783` | `0.3758` | `0.4774` | `-0.1016` |

Hold-family audit results:

| point | target acc | target prob | source prob | target-source margin |
| --- | ---: | ---: | ---: | ---: |
| `Hold4Mid e8` | `0.3040` | `0.2804` | `0.4594` | `-0.1790` |
| `Hold4SlowMid e12` | `0.3053` | `0.2809` | `0.5108` | `-0.2299` |
| `Hold4TwoStage e2` | `0.3067` | `0.2805` | `0.4078` | `-0.1272` |

## Interpretation

This is the most important evaluation surprise so far.

- all three hold-family points beat `LBM-Knee e13` on the non-CLIP style classifier
- `Hold4TwoStage e2` is the strongest of the three on:
  - target accuracy
  - target-source margin
- `Hold4Mid e8` remains the strongest pure LPIPS geometry anchor

What this means:

1. the hold family is not merely “style-dead geometry preservation”
2. `raw CLIP-S` is materially underestimating the hold-family style signal
3. the science question has changed:
   - we are no longer asking only “how do we preserve content while raising style”
   - we are also asking “which style components are being missed by CLIP but seen by a style classifier?”

## Practical reading

- `LBM-Knee` remains the current paper-facing balanced point because it still has a stronger mixed profile across the already-landed diagnostics
- but the hold family is now a much more serious candidate family than its raw CLIP-S numbers suggested
- especially:
  - `Hold4Mid e8` should not be discussed as a trivial no-style point anymore
  - `Hold4TwoStage` deserves continued attention, because even an early retained point already has stronger non-CLIP style evidence than `Knee`

## Next actions

1. add visual comparison against `Seedream` for:
   - `Hold4Mid e8`
   - `Hold4TwoStage best`
   - `LBM-Knee e13`
2. add `DINO structure` on the same points
3. treat the hold family as:
   - `geometry-anchor + classifier-supported style family`
   - not just a low-LPIPS curiosity
