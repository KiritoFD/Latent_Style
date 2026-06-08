# `XPred + Kmanifold + Pattn + AnisoStokes + Queue + Hold-Then-TwoStage ClampRelease + OptimizerReset` Remote Packet

Date: 2026-06-08

Scope:

- dataset: `Distinct5-512`
- surface: remote `3060 WSL`
- config:
  - [inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamphold4twostage_reseed_from_e13_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamphold4twostage_reseed_from_e13_seed42_b8a2.json)

Intent:

- stop spending more GPU on single-stage release smoothing
- preserve what the hold family clearly does well:
  - geometry stabilization
  - low-LPIPS basin formation
- reopen style only after a mid-band geometry basin has already been established

Why this candidate exists:

- `hold4mid` proved the family can lock geometry into an extreme low-LPIPS basin
- `hold4slowmid` showed that simply slowing the same one-stage release does not improve on that anchor
- the next coherent mechanism change is therefore structural:
  - first release into a controlled middle band
  - pause there
  - then reopen later toward a wider style regime

Mechanism:

- hold clamp ratio at `1.10` for the first `4` epochs
- release to `1.30` over the next `4` epochs
- hold the `1.30` band for another `4` epochs
- then reopen late toward `1.60` over the final `8` epochs
- total budget is extended to `20` epochs

Success condition:

- preserve the `hold4mid` geometry anchor through the middle of training
- then recover a meaningful amount of style in the late second release
- beat either:
  - `hold4mid` on style at comparable LPIPS
  - or the current recovery-family `e3` points on LPIPS at comparable style

Failure condition:

- the packet just reproduces the geometry anchor with no late style recovery
- or the late second release reintroduces the old wide-release instability

Early health:

- remote launch passed first health under the paper-facing machine contract
- observed runtime memory stayed around `4.58 / 12.29 GiB` during the first health window
- the host-owned post-train eval watcher is also armed:
  - task: `hold4twostage-posteval-watch`
  - log: `.../logs/posteval_watch_launcher_20260608.log`
- the run resumed correctly from the same parent checkpoint as the other recovery-family packets:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/inmortal-exp/aaai2027_inmortal_xpred_kmanifold_pattn_anisostokes_queue_from_pattn_seed42_b8a2/epoch_0013.pt`

Early train read:

- the first `4` epochs match the known good hold-family basin:
  - `e1 = 9.2995 / 5.5625`
  - `e2 = 9.2608 / 5.5312`
  - `e3 = 9.0997 / 5.4688`
  - `e4 = 8.9704 / 5.1875`
- the first transition into the mid band still shows some release shock:
  - `e5 = 9.0940 / 5.2500`
  - `e6 = 9.1177 / 5.5312`

Early interpretation:

- the packet is machine-safe and operationally stable
- the early hold regime still behaves as expected
- by `e6`, the first release into the mid band has not yet proved itself cleaner than the earlier one-stage schedules
- the real test for this family will be:
  - whether the middle-band hold stabilizes better than `hold4mid`
  - and whether the late reopening can recover style without reopening geometric drift

Mid-band read:

- by `e7-e11`, the packet is starting to look more informative than the earlier one-stage schedules:
  - `e7 = 9.0819 / 5.4375`
  - `e8 = 8.8987 / 5.3438`
  - `e9 = 8.8992 / 5.1875`
  - `e10 = 8.9727 / 5.5312`
  - `e11 = 8.7770 / 5.1562`

Interim interpretation:

- this is still not a clean monotone improvement through the first release window
- but it is slightly more stable than the one-stage lines:
  - `e8/e9` are in the same good basin as the earlier hold family
  - `e11` recovers cleanly after the `e10` rebound
- the family is therefore still live
- the decisive question remains the late reopening phase:
  - if style can recover after the mid-band hold without destroying LPIPS, this will be the first real positive beyond the geometry-anchor family
