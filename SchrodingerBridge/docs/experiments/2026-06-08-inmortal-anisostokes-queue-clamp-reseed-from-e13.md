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
