# `XPred + Kmanifold + Pattn + AnisoStokes + Queue + ProximalTrust` Remote Packet

Date: 2026-06-08

Scope:

- dataset: `Distinct5-512`
- surface: remote `3060 WSL`
- config:
  - [inmortal_xpred_kmanifold_pattn_anisostokes_queue_trust_from_e13_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_anisostokes_queue_trust_from_e13_seed42_b8a2.json)

Intent:

- preserve the parent `e13` low-LPIPS anchor:
  - transfer `0.7102 / 0.4603`
- do not change the parent family:
  - `Aniso + weak Stokes + Queue`
- only add a proximal trust-region penalty

Why this candidate exists:

- the parent line does not look like a fake eval spike
- but after `e13`, the run drifts into a proximal-dominant regime
- the clearest symptom is:
  - `proximal_residual_abs` stays small through `e13`
  - then jumps hard at `e14+`
- we therefore want a mechanism that:
  - leaves transport alone
  - leaves the good `e13` basin reachable
  - blocks the later proximal takeover

Mechanism:

- measure proximal RMS from `last_proximal_residual`
- measure transport RMS from detached `last_base_endpoint - content`
- only penalize the excess if:
  - `proximal_rms > trust_ratio * detached_transport_rms`
- penalty acts on proximal only
- transport reference is detached so the model is not encouraged to collapse transport just to satisfy the trust gate

Success condition:

- keep the low-LPIPS behavior of the parent line
- while preventing the catastrophic `e14+` degradation
- ideally recover a stable retained region near or above the parent `e13` point

Failure condition:

- the trust gate over-constrains style immediately
- or it fails to prevent the same post-`e13` collapse

Primary readout:

- transfer `CLIP-style / LPIPS`
- all-pairs `CLIP-style / LPIPS`
- training-side:
  - `proximal_residual_abs`
  - `base_transport_abs`
  - `proximal_to_transport_ratio`
  - `proximal_trust_penalty`

Reflection prompt:

- if this works, the remaining ceiling debt is not transport quality but proximal stability
- if this fails, the family probably needs a stronger architectural split than a soft trust-region can provide

Outcome:

- remote packet completed through `epoch_0024`
- full eval completed for every retained point from `epoch_0014` to `epoch_0024`

Full readout:

| epoch | transfer CLIP-style | transfer LPIPS | all-pairs CLIP-style | all-pairs LPIPS |
| --- | ---: | ---: | ---: | ---: |
| `e14` | `0.6942` | `0.5785` | `0.6996` | `0.5742` |
| `e15` | `0.6934` | `0.5786` | `0.6987` | `0.5740` |
| `e16` | `0.6938` | `0.5956` | `0.6987` | `0.5909` |
| `e17` | `0.7095` | `0.5751` | `0.7168` | `0.5703` |
| `e18` | `0.6995` | `0.6117` | `0.7043` | `0.6049` |
| `e19` | `0.6959` | `0.6367` | `0.6999` | `0.6297` |
| `e20` | `0.6888` | `0.6787` | `0.6924` | `0.6726` |
| `e21` | `0.6959` | `0.6692` | `0.6991` | `0.6617` |
| `e22` | `0.6924` | `0.6852` | `0.6943` | `0.6778` |
| `e23` | `0.6931` | `0.6794` | `0.6960` | `0.6712` |
| `e24` | `0.6953` | `0.6887` | `0.6978` | `0.6807` |

Best retained point:

- `e17`
  - transfer `0.7095 / 0.5751`
  - all-pairs `0.7168 / 0.5703`

Why this is a negative closure:

- the parent anchor stayed:
  - `e13 = 0.7102 / 0.4603`
  - all-pairs `= 0.7303 / 0.4559`
- the trust packet never beat it on either transfer or all-pairs
- the best trust point only nearly ties style, but loses catastrophically on LPIPS

Mechanism read:

- the trust penalty was definitely active
  - `proximal_trust_penalty` rose from about `0.027` at `e14` to about `0.106` by `e24`
- but it failed to keep the proximal branch inside the intended regime
  - `proximal_to_transport_ratio` still grew from about `1.29` to about `1.85`
  - `proximal_residual_abs` still grew from about `0.228` to about `0.438`
- so the problem is not “missing penalty wiring”
- the stronger interpretation is:
  - the resumed optimizer/scheduler state was already carrying the run out of the `e13` basin
  - the soft trust-region penalty was too weak to reverse that inherited drift

Decision:

- negative closure
- do not promote
- keep the parent `AnisoStokesQueue e13` as the paper-facing low-LPIPS anchor
- next recovery line should test:
  - keep the `e13` model weights
  - but reset optimizer and training-state momentum
