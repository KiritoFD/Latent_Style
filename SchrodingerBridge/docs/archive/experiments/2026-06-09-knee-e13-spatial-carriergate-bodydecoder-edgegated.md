# `LBM-Knee e13 + Spatial Carrier-Gate Body+Decoder + Edge-Gated Structure Leash`

Date: 2026-06-09

Scope:

- dataset: `Distinct5-512`
- surface: remote `3060 WSL`
- config:
  - [inmortal_knee_e13_spatial_carriergate_bodydecoder_edgegated_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_knee_e13_spatial_carriergate_bodydecoder_edgegated_seed42_b8a2.json)

Why this packet exists:

- the plain `Knee + spatial carrier body+decoder` line was locally reviewed as:
  - slightly better on `IntroStyle target`
  - slightly better on `IntroStyle delta-IDT`
  - but clearly worse on `DINO`
- that suggests the family may need:
  - more explicit structure leash
  - not just another carrier-gate micro-variant

Mechanism:

- keep:
  - `Knee e13` anchor
  - `spatial_carrier_gate`
  - `body_decoder` injection
- add:
  - `edge_gated_anisotropic_plus_stokes`
  - `anisotropic_edge_gate_gamma = 12.0`

Hypothesis:

- the previous spatial carrier line may have failed because it introduced extra target-style motion
  without enough structure-aware resistance
- edge-gating the structure penalty should:
  - keep pressure high around strong content edges
  - while allowing flatter regions to carry more style texture

Success condition:

- preserve the small `IntroStyle` gain already seen in the spatial carrier line
- recover part of the `DINO` loss relative to `LBM-Knee`
- stay meaningfully cleaner than `LBM-PS-v2`

Failure condition:

- style gain disappears once structure leash is added
- or structure still drifts enough that the packet remains a `near-negative`

## Runtime update

Latest checked runtime state:

- the training phase has completed
- because the launcher ran from the workspace root, the actual run directory is:
  - `/mnt/i/Github/Latent_Style/exp/inmortal-exp/aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_edgegated_seed42_b8a2`
- observed checkpoint progress has already reached:
  - `epoch_0012.pt`
- early `full_eval` summaries are already landed through:
  - `epoch_0008`
- the corrected fresh-eval watcher is now attached to this true run directory and is observing the live train process
- remote `full_eval` has continued progressing into later retained points and is already processing:
  - `epoch_0012`
- the watcher has now observed:
  - `train_alive=False`
  - and switched into the post-train `fresh_localreview` image-backed rerun stage
- image-backed `fresh_localreview` summaries have already started landing:
  - through `epoch_0004`
- the latest local pulled early curve is still only stable through:
  - `epoch_0008`

Current interpretation:

- the remote lane is not idle
- and this is now the primary training-side bet after the spatial-carrier line and the schedule-only `Hold4TwoStage` family both failed local non-CLIP review

Current early eval artifacts:

- [knee_spatial_carriergate_bodydecoder_edgegated_fast_eval_curve_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/knee_spatial_carriergate_bodydecoder_edgegated_fast_eval_curve_20260609.csv)
- [knee_spatial_carriergate_bodydecoder_edgegated_bestfew_handoff_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/knee_spatial_carriergate_bodydecoder_edgegated_bestfew_handoff_20260609.csv)

Current early read:

- `epoch_0001`
  - transfer `CLIP-style / LPIPS = 0.7040 / 0.4397`
- `epoch_0002`
  - transfer `CLIP-style / LPIPS = 0.7038 / 0.4398`
- `epoch_0003`
  - transfer `CLIP-style / LPIPS = 0.7036 / 0.4391`
- `epoch_0004`
  - transfer `CLIP-style / LPIPS = 0.7038 / 0.4396`
- `epoch_0005`
  - transfer `CLIP-style / LPIPS = 0.7037 / 0.4400`
- `epoch_0006`
  - transfer `CLIP-style / LPIPS = 0.7036 / 0.4401`
- `epoch_0007`
  - transfer `CLIP-style / LPIPS = 0.7037 / 0.4394`
- `epoch_0008`
  - transfer `CLIP-style / LPIPS = 0.7038 / 0.4401`

Interpretation update:

- the line is machine-safe and operationally stable
- but the first eight retained points are still only a weak early signal
- by `CLIP/LPIPS` alone, this still has not separated cleanly from the earlier spatial-carrier line
- the refreshed best-few handoff still stays on:
  - `epoch_0001` for early best `CLIP-style`
  - `epoch_0003` for early best `LPIPS`
- so `epoch_0004..0008` still do not materially change the early read
- even though the remote eval is already advancing beyond that window, there is still no evidence yet that the line has broken out of the same early plateau
- the real decision still requires later retained points plus the same local non-CLIP review stack

Local non-CLIP update:

- the first local best-few review is now available for:
  - `epoch_0001`
  - `epoch_0003`
- result:
  - `IntroStyle target` is only around `0.1085 to 0.1088`
  - `DINO` improves only slightly versus the plain spatial-carrier line:
    - from about `0.0284` to about `0.0281`
- interpretation:
  - this is useful theory evidence that explicit structure leash matters
  - but it is still not a promoted paper-facing point
