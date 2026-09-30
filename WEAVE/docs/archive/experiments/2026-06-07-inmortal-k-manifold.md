# `K_manifold` Remote Packet

Date: 2026-06-07

Scope:

- dataset: `Distinct5-512`
- surface: `H-family` remote `3060 WSL`
- config:
  - [inmortal_k_manifold_seed42_b16.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_k_manifold_seed42_b16.json)

Intent:

- test the highest-ceiling kinetic family from `inmortal.md`
- allow high-frequency motion in flat regions while heavily taxing edge-breaking motion

Reflection template:

- did manifold-adaptive kinetic beat `K_spatial` on style growth?
- did LPIPS improve relative to pure spatial split?
- did edge protection help without throttling texture too early?

## Full readout

| epoch | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `e1` | `0.6599` | `0.3346` |
| `e2` | `0.6569` | `0.3264` |
| `e3` | `0.6614` | `0.3315` |
| `e4` | `0.6597` | `0.3493` |
| `e5` | `0.6611` | `0.3602` |
| `e6` | `0.6629` | `0.3349` |
| `e7` | `0.6622` | `0.3373` |
| `e8` | `0.6618` | `0.3492` |

Best retained point:

- `e6`
  - transfer `clip_style = 0.6629`
  - transfer `content_lpips = 0.3349`
  - full `clip_style = 0.6949`
  - full `content_lpips = 0.3260`

Interpretation:

- this packet is a mild improvement over `K_spatial`
- it does not materially raise the style ceiling
- but it does keep a comparatively good LPIPS region while staying stable

Mechanism conclusion:

- `K_manifold` is not a standalone answer to the ceiling problem
- but it is a credible content-preserving repair candidate for the `XPred_Barycenter` family
