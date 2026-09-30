# `XPred + QueueSmoothing` Remote Packet

Date: 2026-06-07

Scope:

- dataset: `Distinct5-512`
- surface: remote `3060 WSL`
- config:
  - [inmortal_xpred_queue_seed42_b16.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_queue_seed42_b16.json)

Intent:

- isolate the fixed queue-side smoothing bundle from `inmortal.md`
- keep endpoint prediction active
- turn off:
  - structure-aware OT
  - barycentric target smoothing
  - EMA teacher smoothing
- enable only:
  - `pairing_cache_dual_target_mix = 0.5`
  - `pairing_cache_dual_target_topk = 1`
  - `pairing_cache_aux_target_topk = 1`
  - `terminal_swd_aux_weight = 0.5`

Expected upside:

- reduce target-variance noise through the existing prototype-aware pairing cache path
- preserve more transfer signal than plain sampled-target `XPred`
- test whether queue-side smoothing alone can improve the low-LPIPS tradeoff before combining it with the stronger `P_attn` family

Expected failure mode:

- queue mixing may smooth targets without adding enough structural bias
- if so, the packet may remain trapped near the weak sampled-endpoint regime
- or it may slightly stabilize training while failing to lift the actual transfer frontier

Reflection template:

- does queue-side target smoothing improve the plain `XPred` family without needing barycentric targets or teacher EMA?
- does it help style, LPIPS, or neither?
- if it is positive but weak, does that imply queue smoothing is only decisive on top of the stronger `P_attn` family?
- if it is negative, does that imply the queue bundle needs either:
  - stronger proximal texture generation, or
  - stronger target geometry such as `StructOT`

## Live status

Remote run:

- run dir:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/inmortal-exp/aaai2027_inmortal_xpred_queue_seed42_b16`

Execution chain:

- queue runner:
  - `run_inmortal_remaining_queue_v3`

Current read:

- this packet started automatically after `EndpointTeacher` closed
- a stale remote `latent SaMST` baseline process was briefly overlapping this lane and pushing total GPU memory above the intended single-lane band
- that stale baseline lane was manually killed, after which `QueueSmoothing` returned to the intended single-run state
- current training remains machine-safe:
  - recent remote `nvidia-smi` reads are around `3.6 / 12.3 GiB`
- checkpoints currently present through at least `epoch_0006.pt`

## Mid-run training-only read

Latest training log rows currently available:

| epoch | loss | flow | kinetic | terminal SWD |
| --- | ---: | ---: | ---: | ---: |
| `e1` | `14.4786` | `1.2425` | `0.7739` | `13.2500` |
| `e2` | `12.9046` | `1.2134` | `0.5771` | `12.2500` |
| `e3` | `12.6573` | `1.2156` | `0.4436` | `12.0000` |
| `e4` | `12.2513` | `1.2378` | `0.3982` | `11.5625` |
| `e5` | `12.2666` | `1.2364` | `0.3845` | `11.6875` |
| `e6` | `12.3310` | `1.2459` | `0.3642` | `11.9375` |

Current interim interpretation:

- the queue-side smoothing bundle is clearly changing optimization pressure
- kinetic energy keeps falling through `e6`
- terminal pressure also falls relative to the early epochs
- but this is still only a training-side read
- no transfer `CLIP-S / LPIPS` conclusion is paper-safe until the deferred eval surface lands

## Early eval readout

Available transfer points so far:

| epoch | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `e1` | `0.6564` | `0.7543` |
| `e2` | `0.6955` | `0.6520` |
| `e3` | `0.7083` | `0.6207` |

Current partial interpretation:

- the first retained point is clearly negative
  - it is weaker than the plain `P_attn` family
  - weaker than `EndpointTeacher`
  - and far outside the target LPIPS band
- `e2` recovers style substantially, but still remains well below the stronger mechanism families
- `e3` continues to recover, but still does not catch the relevant comparison lines
- relative to the most relevant baselines:
  - `Queue e2 = 0.6955 / 0.6520`
  - `EndpointTeacher e2 = 0.6984 / 0.6468`
  - `P_attn e2 = 0.7160 / 0.7247`
- and at the next retained point:
  - `Queue e3 = 0.7083 / 0.6207`
  - `StructOT e3 = 0.7057 / 0.6300`
  - `Barycenter e3 = 0.7161 / 0.6559`
  - `EndpointTeacher e3 = 0.7158 / 0.6163`
- this means queue smoothing alone is not showing an obvious standalone frontier gain in the early curve
- if later epochs do not reverse this trend sharply, the likely conclusion is:
  - queue smoothing is not sufficient by itself
  - and should be judged mainly by what it does on top of the stronger `Kmanifold + P_attn` family

Current remote stage-summary state:

- selected point so far:
  - `e5 = 0.7198 / 0.5595`
- remaining deferred eval points still pending:
  - none

## Full readout

| epoch | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `e1` | `0.6564` | `0.7543` |
| `e2` | `0.6955` | `0.6520` |
| `e3` | `0.7083` | `0.6207` |
| `e4` | `0.7085` | `0.5942` |
| `e5` | `0.7198` | `0.5595` |
| `e6` | `0.7133` | `0.5580` |
| `e7` | `0.7161` | `0.5571` |
| `e8` | `0.7145` | `0.5506` |

Best retained point:

- `e5`
  - transfer `clip_style = 0.7198`
  - transfer `content_lpips = 0.5595`
  - full `clip_style = 0.7309`
  - full `content_lpips = 0.5514`

Final epoch:

- `e8`
  - transfer `clip_style = 0.7145`
  - transfer `content_lpips = 0.5506`

Closure interpretation:

- this packet is better than the early standalone queue read initially suggested
- queue-side smoothing alone is not enough to reach the promoted frontier
- but it is also not a throwaway negative:
  - it recovers into the same rough band as the `StructOT` and `EndpointTeacher` secondary lines
  - `Queue e5 = 0.7198 / 0.5595`
  - `StructOT e5 = 0.7190 / 0.5589`
  - `EndpointTeacher e5 = 0.7159 / 0.5555`
- the reading is therefore:
  - positive as a single-mechanism control
  - not promotable over the current balanced frontier
  - still worth carrying forward into the stronger `Kmanifold + P_attn + Queue` family

Mechanism conclusion:

- queue-side smoothing is a real signal, but only a secondary one on its own
- it narrows the gap to the stronger tradeoff lines without replacing them
- the most important result of this packet is not its standalone endpoint
  - it is that the queue bundle is plausible enough to justify the current strong-family escalation

## Next comparison target

Once the full eval surface lands, compare first against:

1. plain sampled-target `XPred` behavior
2. `EndpointTeacher`
3. `StructOT`

If the packet is positive but still below the promoted frontier, the next direct promotion candidate remains:

- [inmortal_xpred_kmanifold_pattn_queue_seed42_b16.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_queue_seed42_b16.json)

because that is the same queue bundle applied to the strongest current `Kmanifold + P_attn` family.
