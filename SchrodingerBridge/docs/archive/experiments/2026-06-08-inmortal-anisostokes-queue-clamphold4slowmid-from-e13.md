# `XPred + Kmanifold + Pattn + AnisoStokes + Queue + Hold-Then-SlowMid ClampRelease + OptimizerReset` Remote Packet

Date: 2026-06-08

Scope:

- dataset: `Distinct5-512`
- surface: remote `3060 WSL`
- config:
  - [inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamphold4slowmid_reseed_from_e13_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamphold4slowmid_reseed_from_e13_seed42_b8a2.json)

Intent:

- preserve the two ingredients that currently look least wrong:
  - explicit `4`-epoch early hold
  - narrower `1.45` release endpoint
- isolate a new hypothesis:
  - the current degradation may come from release onset being too abrupt, not just from the release endpoint

Why this candidate exists:

- `hold4wide` suggests the explicit hold is useful, but its later `1.60` release is too permissive
- `hold4mid` removes the `1.60` endpoint, but its early training read still shows rebound once the release starts:
  - `e1-e4` steadily improve
  - `e5-e6` lose that monotonic trend after release begins
- that makes the next most coherent probe:
  - keep the same hold
  - keep the same endpoint
  - only slow the release

Mechanism:

- hold clamp ratio at `1.10` for the first `4` epochs
- then linearly relax only to `1.45`
- release over `12` epochs instead of `8`

Success condition:

- preserve the stronger early basin from the hold family
- avoid the immediate post-hold rebound seen in `hold4mid`
- keep later epochs closer to the selected `e3/e4` tradeoff instead of reopening drift

Failure condition:

- the packet still degrades as soon as release begins
- or the longer release simply reproduces the old `hold4mid` curve more slowly without a better retained point

Early health:

- remote launch passed first health under the paper-facing machine contract
- observed runtime memory stayed around `4.42 / 12.29 GiB` during the first health window
- the post-train eval watcher is now also running through the host-owned remote command launcher:
  - task: `hold4slowmid-posteval-watch`
  - log: `.../logs/posteval_watch_launcher_20260608.log`
- the run resumed correctly from the same parent checkpoint as the earlier hold family:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/inmortal-exp/aaai2027_inmortal_xpred_kmanifold_pattn_anisostokes_queue_from_pattn_seed42_b8a2/epoch_0013.pt`
- early epoch-1 read:
  - `loss = 9.3085`
  - `terminal_swd = 5.5312`
  - `epoch_time = 172.57s`
  - `samples_per_sec = 28.97`

Early interpretation:

- the packet is machine-safe
- training speed is in the same band as the earlier hold packets
- the substantive question remains whether slowing release over `12` epochs removes the transition shock that appeared in `hold4mid`

Mid-train read:

- the first four epochs still look like the same good basin as `hold4mid`:
  - `e1 = 9.3085 / 5.5312`
  - `e2 = 9.2705 / 5.5938`
  - `e3 = 9.0954 / 5.4688`
  - `e4 = 8.9713 / 5.0312`
- the first two post-hold epochs still show a release shock:
  - `e5 = 9.1065 / 5.3750`
  - `e6 = 9.1120 / 5.4375`

Interim interpretation:

- simply stretching the release horizon does not eliminate the immediate post-hold shock by `e6`
- relative to `hold4mid`, this line is not yet clearly better in the earliest release window
- the meaningful read now depends on whether later epochs recover more cleanly than `hold4mid` did

Later train read:

- by `e7-e10`, the line still looks like a near-tie rather than a clear win:
  - `e7 = 9.0654 / 5.4062`
  - `e8 = 8.8734 / 5.2188`
  - `e9 = 8.8769 / 5.1562`
  - `e10 = 8.9494 / 5.5312`
- relative to `hold4mid`:
  - `e8` is very slightly stronger
  - `e9` is essentially tied
  - `e10` still rebounds upward

Interpretation update:

- slowing the release over `12` epochs helps only marginally in the training curve
- the family still behaves much more like a geometry-control line than a style-recovery line
- the decisive read should come from the completed eval curve, but by training metrics alone this is not a clear successor to `hold4mid`

Outcome:

- training completed through `epoch_0016`
- full eval completed through `epoch_0016`
- the run produced a complete `clip_lpips_curve.csv` without manual repair

Full readout:

| epoch | transfer CLIP-style | transfer LPIPS | all-pairs CLIP-style | all-pairs LPIPS |
| --- | ---: | ---: | ---: | ---: |
| `e1` | `0.6672` | `0.2953` | `0.7004` | `0.2953` |
| `e2` | `0.6675` | `0.2911` | `0.7009` | `0.2911` |
| `e3` | `0.6671` | `0.2945` | `0.7003` | `0.2946` |
| `e4` | `0.6673` | `0.2921` | `0.7007` | `0.2921` |
| `e5` | `0.6672` | `0.2935` | `0.7005` | `0.2936` |
| `e6` | `0.6671` | `0.2920` | `0.7005` | `0.2921` |
| `e7` | `0.6670` | `0.2947` | `0.7001` | `0.2947` |
| `e8` | `0.6678` | `0.2904` | `0.7012` | `0.2904` |
| `e9` | `0.6670` | `0.2966` | `0.7001` | `0.2966` |
| `e10` | `0.6671` | `0.2941` | `0.7004` | `0.2939` |
| `e11` | `0.6671` | `0.2927` | `0.7005` | `0.2927` |
| `e12` | `0.6673` | `0.2898` | `0.7009` | `0.2898` |
| `e13` | `0.6674` | `0.2940` | `0.7006` | `0.2939` |
| `e14` | `0.6671` | `0.2915` | `0.7005` | `0.2915` |
| `e15` | `0.6670` | `0.2935` | `0.7004` | `0.2935` |
| `e16` | `0.6672` | `0.2922` | `0.7007` | `0.2922` |

Best retained points:

- best transfer-style:
  - `e8 = 0.6678 / 0.2904`
  - all-pairs `= 0.7012 / 0.2904`
- best LPIPS within the same style band:
  - `e12 = 0.6673 / 0.2898`
  - all-pairs `= 0.7009 / 0.2898`

Why this packet matters:

- it closes the last coherent single-stage release-smoothing question in the current clamp family
- relative to `hold4mid`, it confirms that simply stretching the same release shape is not enough

Mechanism conclusion:

- `hold4slowmid` is a near-tie negative relative to `hold4mid`
- compared with `hold4mid e8 = 0.6679 / 0.2877`:
  - `hold4slowmid e12 = 0.6673 / 0.2898`
- style is slightly lower and LPIPS is slightly worse
- therefore:
  - slower single-stage release does not reopen enough style
  - and it does not preserve the geometry anchor better either

Decision:

- near-tie negative closure
- do not promote over `hold4mid`
- retire single-stage release smoothing as the next-round headline direction
- next round should use `hold4mid` as the geometry anchor and add a genuinely different late style-reopening mechanism

Outcome:

- training completed through `epoch_0016`
- full eval completed through `epoch_0016`
- the packet produced a complete `clip_lpips_curve.csv` without needing a manual rerun

Full readout:

| epoch | transfer CLIP-style | transfer LPIPS | all-pairs CLIP-style | all-pairs LPIPS |
| --- | ---: | ---: | ---: | ---: |
| `e1` | `0.6672` | `0.2953` | `0.7004` | `0.2953` |
| `e2` | `0.6675` | `0.2911` | `0.7009` | `0.2911` |
| `e3` | `0.6671` | `0.2945` | `0.7003` | `0.2946` |
| `e4` | `0.6673` | `0.2921` | `0.7007` | `0.2921` |
| `e5` | `0.6672` | `0.2935` | `0.7005` | `0.2936` |
| `e6` | `0.6671` | `0.2920` | `0.7005` | `0.2921` |
| `e7` | `0.6670` | `0.2947` | `0.7001` | `0.2947` |
| `e8` | `0.6678` | `0.2904` | `0.7012` | `0.2904` |
| `e9` | `0.6670` | `0.2966` | `0.7001` | `0.2966` |
| `e10` | `0.6671` | `0.2941` | `0.7004` | `0.2939` |
| `e11` | `0.6671` | `0.2927` | `0.7005` | `0.2927` |
| `e12` | `0.6673` | `0.2898` | `0.7009` | `0.2898` |
| `e13` | `0.6674` | `0.2940` | `0.7006` | `0.2939` |
| `e14` | `0.6671` | `0.2915` | `0.7005` | `0.2915` |
| `e15` | `0.6670` | `0.2935` | `0.7004` | `0.2935` |
| `e16` | `0.6672` | `0.2922` | `0.7007` | `0.2922` |

Best retained points:

- best transfer-style:
  - `e8 = 0.6678 / 0.2904`
  - all-pairs `= 0.7012 / 0.2904`
- best LPIPS within the same style band:
  - `e12 = 0.6673 / 0.2898`
  - all-pairs `= 0.7009 / 0.2898`

Why this packet matters:

- it tests the cleanest remaining same-family question after `hold4mid`:
  - can a slower single-stage release recover style without losing the geometry anchor?
- the answer is effectively no

Mechanism conclusion:

- `hold4slowmid` is a near-tie negative relative to `hold4mid`
- it preserves the same low-LPIPS regime, but does not produce a stronger retained point
- compared with `hold4mid e8 = 0.6679 / 0.2877`:
  - `hold4slowmid e12 = 0.6673 / 0.2898`
- this means:
  - slowing the same one-stage release is not enough
  - the next round should stop tuning single-stage schedules and move to a genuinely different late style-reopening mechanism

Decision:

- near-tie negative closure
- do not promote over `hold4mid`
- retain this packet only as evidence that single-stage release smoothing is insufficient
