# `XPred + EndpointTeacher` Remote Packet

Date: 2026-06-07

Scope:

- dataset: `Distinct5-512`
- surface: remote `3060 WSL`
- config:
  - [inmortal_xpred_teacher_endpoint_seed42_b16.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_teacher_endpoint_seed42_b16.json)

Intent:

- isolate the corrected EMA teacher family from `inmortal.md`
- keep endpoint prediction active
- disable:
  - structure-aware OT
  - barycentric target smoothing
- enable only:
  - `target_teacher_mode = style_endpoint_ema`
  - `target_teacher_weight = 0.1`

Expected upside:

- reduce target variance without forcing the model onto a noisy sampled endpoint every step
- improve stability relative to naive sampled endpoint targets
- potentially recover style without the catastrophic LPIPS band seen in early `XPred_Barycenter`

Expected failure mode:

- the teacher may only smooth the same mediocre target regime
- if so, style may remain below the stronger `Pattn/Stokes` family
- or the line may become stable but too conservative

Reflection template:

- does endpoint EMA teacher improve the `XPred` frontier relative to plain `sample` targets?
- does it help more on LPIPS than on style, or vice versa?
- if negative, does that imply teacher smoothing is weaker than barycentric smoothing?
- if positive, should the next escalation be:
  - `Teacher + Pattn`, or
  - `Teacher + Queue`, or
  - `Teacher + StructOT`

## Closure

Remote run:

- run dir:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/inmortal-exp/aaai2027_inmortal_xpred_teacher_endpoint_seed42_b16`

Execution chain:

- queue runner:
  - `run_inmortal_remaining_queue_v3`

Current read:

- the queue advanced into `EndpointTeacher` automatically after `StructOT`
- training completed its full `8` epochs and produced checkpoints through `epoch_0008.pt`
- the deferred `CLIP-S / LPIPS` readout is now complete for `e1-e8`
- first-health stayed under the machine cap during training
- this family remained comfortably safe on the remote `3060`
  - trainer peaks stayed around `2.84 / 3.03 GB`

## Full readout

| epoch | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `e1` | `0.6524` | `0.7571` |
| `e2` | `0.6984` | `0.6468` |
| `e3` | `0.7158` | `0.6163` |
| `e4` | `0.7098` | `0.5943` |
| `e5` | `0.7159` | `0.5555` |
| `e6` | `0.7124` | `0.5586` |
| `e7` | `0.7115` | `0.5581` |
| `e8` | `0.7103` | `0.5495` |

Best retained point:

- `e5`
  - transfer `clip_style = 0.7159`
  - transfer `content_lpips = 0.5555`
  - full `clip_style = 0.7276`
  - full `content_lpips = 0.5473`

Final epoch:

- `e8`
  - transfer `clip_style = 0.7103`
  - transfer `content_lpips = 0.5495`
  - full `clip_style = 0.7226`
  - full `content_lpips = 0.5408`

Interpretation:

- the endpoint-level EMA teacher is a real positive smoothing signal
- it rapidly escapes the weak early `XPred` regime and reaches a stable lower-LPIPS band by `e5`
- compared against the closest single-mechanism control:
  - `StructOT e5 = 0.7190 / 0.5589`
  - `EndpointTeacher e5 = 0.7159 / 0.5555`
- this means the teacher-only line gives back a small amount of style, but slightly improves LPIPS at the same rough frontier band
- later epochs keep shaving LPIPS, but not enough to offset the style drop under the current promotion rule
- relative to the promoted `Pattn/Stokes` family, this packet is still a secondary tradeoff line rather than a new main frontier

Mechanism conclusion:

- corrected EMA teacher smoothing is useful, but not dominant by itself
- its strongest paper-safe reading is:
  - better target smoothing than naive sampled endpoints
  - a slightly stronger low-LPIPS secondary line than `StructOT`
  - still clearly weaker than the stronger proximal transport family on the actual ceiling
- the next logical use of this mechanism is not another standalone rerun
  - it is to combine teacher smoothing with the stronger queue or `Pattn` families
