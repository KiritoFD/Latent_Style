# IntroStyle Page-1 Baselines Smoke20

Date: 2026-06-08

Scope:

- extend the existing `IntroStyle` page-1 smoke shortlist
- explicitly include:
  - `SaMAM` pixel
  - `SaMST` pixel
  - `SaMAM-latent`
  - `SaMST-latent`
- keep the same smoke protocol:
  - `20` rows per point
  - `4` held-out bank images per target style
  - `ensemble_size = 1`
  - `t = 25`
  - `up_ft_index = 1`

Artifacts:

- shortlist summary:
  - [introstyle_page1_summary.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/introstyle_page1/introstyle_page1_summary.csv)
  - [introstyle_page1_summary.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/introstyle_page1/introstyle_page1_summary.md)
- updated page-1 figure:
  - [fig_distinct5_page1_summary_introstyle_delta_idt.png](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/figures/fig_distinct5_page1_summary_introstyle_delta_idt.png)
  - [fig_distinct5_page1_summary_introstyle_delta_idt.pdf](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/figures/fig_distinct5_page1_summary_introstyle_delta_idt.pdf)

## Smoke read

| point | transfer target | delta-IDT | style margin |
| --- | ---: | ---: | ---: |
| `Seedream-4.5` | `0.0936` | `+0.0242` | `-0.0292` |
| `LBM-Knee` | `0.0837` | `+0.0142` | `-0.0092` |
| `LBM-K` | `0.0774` | `+0.0080` | `-0.0243` |
| `SaMST e15` | `0.0774` | `+0.0079` | `-0.1379` |
| `LBM-PS-v2` | `0.0733` | `+0.0038` | `-0.0007` |
| `SaMAM-2250` | `0.0720` | `+0.0025` | `-0.1072` |
| `Lat SaMAM` | `0.0694` | `-0.0001` | `-0.1310` |
| `Lat SaMST` | `0.0422` | `-0.0272` | `-0.0194` |

## Immediate reading

- `Seedream` remains strongest on raw `IntroStyle target score`.
- `LBM-Knee` is still the strongest internal point on `delta-IDT`.
- `SaMST e15` beats `SaMAM-2250` on this smoke `IntroStyle` axis, but with much worse specificity margin.
- `Lat SaMAM` is effectively at the `IDT` floor on this smoke protocol.
- `Lat SaMST` is below the `IDT` floor on `delta-IDT`, which keeps it as a negative latent baseline.

## Figure implication

- the page-1 left panel can now be switched from `transfer CLIP-S delta-IDT` to `IntroStyle delta-IDT`
- the current script support is:
  - [scripts_gen_distinct5_page1_summary.py](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/scripts_gen_distinct5_page1_summary.py)
  - `--y-metric introstyle_delta_idt`
