# `XPred + Kmanifold + Pattn + AnisoStokes + Queue + HardClamp + OptimizerReset` Remote Packet

Date: 2026-06-08

Scope:

- dataset: `Distinct5-512`
- surface: remote `3060 WSL`
- config:
  - [inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamp_reseed_from_e13_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamp_reseed_from_e13_seed42_b8a2.json)

Intent:

- build directly on the `reseed-from-e13` evidence
- keep the same `e13` model weights
- keep optimizer reset
- replace “soft trust only” with a hard proximal cap

Why this candidate exists:

- the trust penalty was active but could not stop proximal takeover
- optimizer reset helped, but still drifted back toward a high-LPIPS regime
- that means the remaining missing piece is a real output-side control, not another weak regularizer

Mechanism:

- after high-pass filtering the proximal residual, compute:
  - proximal RMS
  - transport RMS from `z_base - x0`
- if proximal RMS exceeds `clamp_ratio * transport_rms`, scale the residual down before adding it back

Success condition:

- hold the early low-proximal regime longer than the `reseed` packet
- recover a retained point materially closer to the parent `e13` tradeoff

Failure condition:

- hard clamp just kills style immediately
- or it still cannot prevent drift away from the parent basin

Interim read:

- this is the first recovery line that actually changes the training regime in the intended direction
- unlike the earlier trust-only packets, the proximal branch is being hard-limited instead of merely softly discouraged

Early retained points:

| epoch | transfer CLIP-style | transfer LPIPS | all-pairs CLIP-style | all-pairs LPIPS |
| --- | ---: | ---: | ---: | ---: |
| `e1` | `0.7071` | `0.5272` | `0.7196` | `0.5211` |
| `e2` | `0.6989` | `0.5602` | `0.7095` | `0.5528` |
| `e3` | `0.7022` | `0.4867` | `0.7183` | `0.4814` |

Training-side read:

- `proximal_to_transport_ratio` is no longer drifting upward
- from `epoch_2` onward it is effectively pinned at the configured clamp ceiling:
  - `1.25`
- `proximal_residual_abs` stays around `0.20-0.23`
  - far below the `0.31-0.45` regime seen in the failed trust and trust-reseed continuations

Why this matters:

- `e3` already beats the user-provided target-side reference on transfer LPIPS:
  - current `e3 transfer = 0.7022 / 0.4867`
  - reference `transfer = 0.6920 / 0.4923`
- and `e1` / `e3` both push all-pairs style very close to the external reference band:
  - `e1 all-pairs = 0.7196 / 0.5211`
  - `e3 all-pairs = 0.7183 / 0.4814`

Current interpretation:

- hard proximal clamping is the first mechanism that appears to genuinely hold the family near the desired content/style regime
- this is still an interim reading
- the packet should continue to full retained closure before any final paper-facing promotion decision

Outcome:

- training completed through `epoch_0016`
- retained eval completed through `epoch_0016`

Full readout:

| epoch | transfer CLIP-style | transfer LPIPS | all-pairs CLIP-style | all-pairs LPIPS |
| --- | ---: | ---: | ---: | ---: |
| `e1` | `0.7071` | `0.5272` | `0.7196` | `0.5211` |
| `e2` | `0.6989` | `0.5602` | `0.7095` | `0.5528` |
| `e3` | `0.7022` | `0.4867` | `0.7183` | `0.4814` |
| `e4` | `0.6933` | `0.5373` | `0.7065` | `0.5309` |
| `e5` | `0.6946` | `0.5242` | `0.7083` | `0.5177` |
| `e6` | `0.6861` | `0.5356` | `0.7011` | `0.5269` |
| `e7` | `0.6903` | `0.5076` | `0.7074` | `0.4989` |
| `e8` | `0.6859` | `0.5243` | `0.7009` | `0.5155` |
| `e9` | `0.6888` | `0.5086` | `0.7048` | `0.5009` |
| `e10` | `0.6839` | `0.5251` | `0.6998` | `0.5158` |
| `e11` | `0.6825` | `0.5176` | `0.6978` | `0.5091` |
| `e12` | `0.6812` | `0.5207` | `0.6969` | `0.5115` |
| `e13` | `0.6882` | `0.5013` | `0.7047` | `0.4929` |
| `e14` | `0.6821` | `0.5211` | `0.6973` | `0.5124` |
| `e15` | `0.6811` | `0.5159` | `0.6972` | `0.5069` |
| `e16` | `0.6826` | `0.5214` | `0.6981` | `0.5122` |

Best retained points:

- best transfer-style:
  - `e1 = 0.7071 / 0.5272`
  - all-pairs `= 0.7196 / 0.5211`
- best LPIPS under `transfer >= 0.70`:
  - `e3 = 0.7022 / 0.4867`
  - all-pairs `= 0.7183 / 0.4814`

Why this packet matters:

- this is the first recovery line that beats the user-provided transfer reference on both axes:
  - packet `e3 transfer = 0.7022 / 0.4867`
  - reference `transfer = 0.6920 / 0.4923`
- and it nearly matches the external all-pairs anchor while staying much lighter than that reference family:
  - packet `e1 all-pairs = 0.7196 / 0.5211`
  - packet `e3 all-pairs = 0.7183 / 0.4814`
  - reference `all-pairs = 0.7198 / 0.4767`

Mechanism conclusion:

- hard clamping works materially better than:
  - soft trust penalty alone
  - trust + optimizer reset
- the decisive difference is that the proximal branch is no longer allowed to silently take over
- in the training log, `proximal_to_transport_ratio` is held at the configured ceiling for the whole retained window instead of drifting into the `1.7+` regime

Decision:

- positive closure
- not the new global headline frontier
- but the new strongest recovery-family evidence for “proximal takeover control matters”
- retain:
  - `e1` as the strongest style-heavy clamp point
  - `e3` as the best low-LPIPS clamp point
