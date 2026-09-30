# `XPred + Kmanifold + Pattn + AnisoStokes + Queue + ProximalTrust + OptimizerReset` Remote Packet

Date: 2026-06-08

Scope:

- dataset: `Distinct5-512`
- surface: remote `3060 WSL`
- config:
  - [inmortal_xpred_kmanifold_pattn_anisostokes_queue_trust_reseed_from_e13_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_anisostokes_queue_trust_reseed_from_e13_seed42_b8a2.json)

Intent:

- keep the same parent `e13` model weights
- keep the same trust-region mechanism
- explicitly remove inherited optimizer/scheduler/training-state momentum

Why this candidate exists:

- the trust packet proved the leash term was wired correctly
- but it still entered the same bad regime almost immediately
- that points to a continuation-state problem, not just a missing penalty

Hypothesis:

- `e13` is a real good basin
- the parent continuation optimizer state is what pushes the run out of it
- if the model is reseeded from `e13` weights while the optimizer is reset, the trust leash may now be strong enough to keep the basin

Success condition:

- beat the trust packet clearly
- and ideally recover or exceed the parent `e13` low-LPIPS tradeoff

Failure condition:

- it still leaves the basin quickly
- or it never re-enters the parent `e13` quality band even without inherited optimizer momentum

Outcome:

- training completed through `epoch_0016`
- remote eval closure currently confirms `epoch_0001` through `epoch_0016`

Observed readout:

| epoch | transfer CLIP-style | transfer LPIPS | all-pairs CLIP-style | all-pairs LPIPS |
| --- | ---: | ---: | ---: | ---: |
| `e1` | `0.6977` | `0.5691` | `0.7040` | `0.5657` |
| `e2` | `0.6958` | `0.5933` | `0.7004` | `0.5887` |
| `e3` | `0.7021` | `0.5654` | `0.7085` | `0.5621` |
| `e4` | `0.6927` | `0.5975` | `0.6975` | `0.5930` |
| `e5` | `0.6873` | `0.6269` | `0.6910` | `0.6219` |
| `e6` | `0.6941` | `0.6404` | `0.6967` | `0.6325` |
| `e7` | `0.6984` | `0.6776` | `0.7004` | `0.6718` |
| `e8` | `0.6993` | `0.6789` | `0.7019` | `0.6721` |
| `e9` | `0.7031` | `0.6697` | `0.7048` | `0.6630` |
| `e10` | `0.6975` | `0.6850` | `0.6985` | `0.6771` |
| `e11` | `0.6968` | `0.6845` | `0.6986` | `0.6768` |
| `e12` | `0.6995` | `0.6997` | `0.7001` | `0.6923` |
| `e13` | `0.7021` | `0.6924` | `0.7040` | `0.6842` |
| `e14` | `0.6995` | `0.6991` | `0.7010` | `0.6910` |
| `e15` | `0.7017` | `0.6990` | `0.7034` | `0.6908` |
| `e16` | `0.7006` | `0.6976` | `0.7019` | `0.6892` |

Best retained point:

- `e9`
  - transfer `0.7031 / 0.6697`
  - all-pairs `0.7048 / 0.6630`

Interpretation:

- resetting optimizer and training state did help relative to the direct continuation packet
  - early epochs are clearly better than the previous trust continuation
  - the line no longer immediately collapses to the `0.69 / 0.58+` band
- but this still does not recover the parent low-LPIPS basin
  - parent anchor stayed:
    - `e13 = 0.7102 / 0.4603`
    - all-pairs `= 0.7303 / 0.4559`
- the best reseeded point remains far away on LPIPS

Mechanism conclusion:

- inherited optimizer momentum was part of the problem
- but not the whole problem
- even with optimizer reset, the family still drifts back toward a proximal-dominant high-LPIPS regime
- the next mechanism should therefore be stronger than:
  - soft trust penalty
  - or optimizer reset alone

Decision:

- negative closure
- do not promote
- keep this packet as evidence that optimizer-state reset helps, but is insufficient
