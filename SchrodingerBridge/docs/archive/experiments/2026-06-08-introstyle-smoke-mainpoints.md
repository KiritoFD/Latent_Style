# IntroStyle Smoke On Main Paper Points

Date: 2026-06-08

Scope:

- first runnable `IntroStyle` landing for the current project
- smoke scale only:
  - `20` transfer rows per point
  - `4` held-out bank images per target style
  - `ensemble_size = 1`
  - `t = 25`
  - `up_ft_index = 1`
- model source:
  - local `ModelScope` snapshot of `stabilityai/stable-diffusion-2-1-base`

Points included:

- `IDT`
- `LBM-Knee e13`
- `LBM-PS-v2 e13`
- `Hold4Mid e8`
- `Seedream-4.5`

Outputs:

- temporary smoke CSV:
  - `G:\GitHub\Latent_Style\tmp\introstyle_mainpoints_smoke.csv`
- temporary smoke JSON:
  - `G:\GitHub\Latent_Style\tmp\introstyle_mainpoints_smoke.json`

## Smoke results

| point | transfer target score | transfer source score | best non-target | style margin | identity target score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `IDT` | `0.0645` | `0.2920` | `0.2920` | `-0.2274` | `0.2867` |
| `LBM-Knee e13` | `0.0765` | `0.0642` | `0.0906` | `-0.0141` | `0.0744` |
| `LBM-PS-v2 e13` | `0.0685` | `0.0561` | `0.0806` | `-0.0121` | `0.0620` |
| `Hold4Mid e8` | `0.0709` | `0.0647` | `0.0918` | `-0.0209` | `0.0607` |
| `Seedream-4.5` | `0.0849` | `0.1180` | `0.1397` | `-0.0548` | `0.1411` |

## Immediate reading

- the pipeline is now genuinely runnable, which is the main landing milestone
- even this tiny smoke already gives a different ordering than raw CLIP-S
- `LBM-Knee e13` remains the best current internal point on the smoke `style_margin`
- `Hold4Mid e8` is not catastrophically worse than `Knee`
- `Seedream` has the highest target-style score, but also a worse margin than `Knee / PS-v2`, so the smoke suggests stronger target pull is not the same as cleaner style specificity

## Limits of this smoke

- this is not yet paper-safe
- reasons:
  - only `20` transfer rows per point
  - only `4` style-bank images per class
  - only one `IntroStyle` feature setting
  - no full held-out style-bank protocol yet

## What landed

- runnable `IntroStyle` feature extractor:
  - [introstyle_eval.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/utils/introstyle_eval.py)
- runnable `IntroStyle` probe script:
  - [eval_introstyle_probe.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/eval_introstyle_probe.py)

## Next step

- expand from smoke to a paper-facing shortlist protocol:
  - larger held-out style bank
  - at least the current main points
  - and all abnormal low-LPIPS points that might influence theory claims
