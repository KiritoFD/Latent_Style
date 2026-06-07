# `XPred + Kmanifold + Pattn + AnisoStokes + Queue + ClampRelease + OptimizerReset` Remote Packet

Date: 2026-06-08

Scope:

- dataset: `Distinct5-512`
- surface: remote `3060 WSL`
- config:
  - [inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamprelease_reseed_from_e13_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamprelease_reseed_from_e13_seed42_b8a2.json)

Intent:

- build directly on the positive hard-clamp signal
- keep the e13 weights and optimizer reset
- replace fixed clamp with a release schedule

Why this candidate exists:

- fixed hard clamp clearly solved proximal takeover
- but it likely capped style too aggressively after the first few epochs
- the next natural mechanism is:
  - clamp very hard in the earliest epochs
  - then gradually relax once the run is safely inside the good basin

Mechanism:

- start clamp ratio at `1.10`
- linearly relax to `1.45`
- release over the first `6` epochs

Success condition:

- preserve the stable low-proximal regime from the hard-clamp run
- beat the fixed-clamp `e3` tradeoff on transfer style without giving back all of its LPIPS gain

Failure condition:

- style still stays capped at the same low ceiling
- or the released clamp simply reopens the old proximal-takeover failure mode

Early read:

- this line is not keeping a fixed `1.25` clamp forever
- the logged `proximal_to_transport_ratio` is actually moving as intended:
  - `e1 ≈ 1.03`
  - `e2 ≈ 1.16`
  - `e3 ≈ 1.22`
  - `e4 ≈ 1.28`
  - later retained epochs sit near the released target band

Current retained points:

| epoch | transfer CLIP-style | transfer LPIPS | all-pairs CLIP-style | all-pairs LPIPS |
| --- | ---: | ---: | ---: | ---: |
| `e1` | `0.7064` | `0.5149` | `0.7207` | `0.5086` |
| `e2` | `0.7011` | `0.5453` | `0.7128` | `0.5381` |
| `e3` | `0.7007` | `0.4754` | `0.7183` | `0.4701` |
| `e4` | `0.6958` | `0.5232` | `0.7097` | `0.5169` |
| `e5` | `0.6948` | `0.5137` | `0.7095` | `0.5076` |
| `e6` | `0.6901` | `0.5174` | `0.7063` | `0.5091` |
| `e7` | `0.6916` | `0.4864` | `0.7098` | `0.4780` |

Interim interpretation:

- relative to fixed hard clamp:
  - fixed-clamp best LPIPS-under-`transfer>=0.70` point was:
    - `e3 = 0.7022 / 0.4867`
  - clamp-release currently gives:
    - `e3 = 0.7007 / 0.4754`
- this is the exact trade we wanted to test:
  - tiny style giveback
  - noticeable LPIPS recovery

Status:

- positive interim signal
- keep running to full retained closure before final paper-facing promotion
