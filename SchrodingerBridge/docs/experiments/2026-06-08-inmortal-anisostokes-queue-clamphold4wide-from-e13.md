# `XPred + Kmanifold + Pattn + AnisoStokes + Queue + Hold-Then-Wide ClampRelease + OptimizerReset` Remote Packet

Date: 2026-06-08

Scope:

- dataset: `Distinct5-512`
- surface: remote `3060 WSL`
- config:
  - [inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamphold4wide_reseed_from_e13_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamphold4wide_reseed_from_e13_seed42_b8a2.json)

Intent:

- preserve the known-good early `1.10` clamp exactly
- stop approximating the early hold with a slow linear schedule
- test whether the missing ingredient is an explicit hold window before wider release

Why this candidate exists:

- the first release packet suggests the early low-LPIPS basin depends on a genuinely tight proximal cap
- the negative `wide release` packet shows that loosening the early basin is harmful
- the current `late-wide linear` packet is still only an approximation of "hold, then release"
- this packet turns that hypothesis into an explicit mechanism

Mechanism:

- hold clamp ratio at `1.10` for the first `4` epochs
- then linearly relax to `1.60`
- release over the next `8` epochs

Success condition:

- keep the early packet in the same low-LPIPS basin as:
  - first release `e3 = 0.7007 / 0.4754`
- while giving the model a later style-recovery window that can exceed that style level without reopening proximal takeover

Failure condition:

- even with an explicit hold window, later wider release still cannot improve the frontier
- or the release phase simply reintroduces late proximal domination

Outcome:

- training completed through `epoch_0015`
- full eval completed through `epoch_0015`
- the final eval closure had to be resumed with smaller eval batches to stay inside the remote memory contract:
  - first pass completed `e1..e12`
  - resumed `e13` at smaller batch
  - finished `e14..e15` at `batch_size=5`, `vae_decode_batch_size=10`

Full readout:

| epoch | transfer CLIP-style | transfer LPIPS | all-pairs CLIP-style | all-pairs LPIPS |
| --- | ---: | ---: | ---: | ---: |
| `e1` | `0.7067` | `0.5137` | `0.7209` | `0.5075` |
| `e2` | `0.7015` | `0.5462` | `0.7133` | `0.5390` |
| `e3` | `0.7009` | `0.4726` | `0.7189` | `0.4671` |
| `e4` | `0.6968` | `0.5318` | `0.7099` | `0.5255` |
| `e5` | `0.6965` | `0.5141` | `0.7113` | `0.5075` |
| `e6` | `0.6886` | `0.5226` | `0.7041` | `0.5145` |
| `e7` | `0.6914` | `0.5004` | `0.7087` | `0.4923` |
| `e8` | `0.6899` | `0.5092` | `0.7059` | `0.5010` |
| `e9` | `0.6896` | `0.4937` | `0.7068` | `0.4870` |
| `e10` | `0.6865` | `0.5068` | `0.7030` | `0.4991` |
| `e11` | `0.6850` | `0.4950` | `0.7014` | `0.4882` |
| `e12` | `0.6850` | `0.4951` | `0.7021` | `0.4880` |
| `e13` | `0.6901` | `0.4846` | `0.7076` | `0.4775` |
| `e14` | `0.6850` | `0.4968` | `0.7022` | `0.4893` |
| `e15` | `0.6846` | `0.4938` | `0.7023` | `0.4862` |

Best retained points:

- best transfer-style:
  - `e1 = 0.7067 / 0.5137`
  - all-pairs `= 0.7209 / 0.5075`
- best LPIPS under `transfer >= 0.70`:
  - `e3 = 0.7009 / 0.4726`
  - all-pairs `= 0.7189 / 0.4671`

Why this packet matters:

- it is the first explicit `hold then release` packet that actually beats the earlier release family's selected low-LPIPS point on both transfer axes:
  - first release `e3 = 0.7007 / 0.4754`
  - hold-then-wide `e3 = 0.7009 / 0.4726`
- the gain is small, but it is real and directionally clean:
  - explicit early hold helps the packet settle into a slightly better `e3` basin

Mechanism conclusion:

- the explicit `4`-epoch hold is useful
- the later drift still shows that widening all the way to `1.60` is too aggressive
- the best point remains early, before the broader release has time to undo the low-LPIPS basin

Decision:

- positive incremental closure
- keep `e3` as the best low-LPIPS point in the clamp-release recovery family
- do not promote this packet over the paper-facing `AnisoStokesQueue e13` anchor
- next justified move is:
  - preserve the explicit hold
  - but reduce the release endpoint back toward the successful `1.45` family instead of `1.60`
