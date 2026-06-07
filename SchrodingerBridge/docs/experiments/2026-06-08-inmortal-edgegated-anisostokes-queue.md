# `XPred + Kmanifold + Pattn + EdgeGatedAnisoStokes + Queue` Remote Packet

Date: 2026-06-08

Scope:

- dataset: `Distinct5-512`
- surface: remote `3060 WSL`
- config:
  - [inmortal_xpred_kmanifold_pattn_edgegated_anisostokes_queue_from_e13_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_edgegated_anisostokes_queue_from_e13_seed42_b8a2.json)

Intent:

- resume from the `e13` low-LPIPS anchor:
  - `0.7102 / 0.4603`
- keep:
  - weak Stokes smoothing
  - queue-side target smoothing
- replace uniform anisotropic normal pressure with edge-gated anisotropic pressure

Why this candidate exists:

- the current low-LPIPS anchor likely over-regularizes style away in flat regions
- content edges deserve strong normal suppression
- flat regions should be freer to restore style

Success condition:

- recover transfer style toward `0.72+`
- while preserving a clear LPIPS win over the current balanced frontier

Failure condition:

- style still does not recover
- or LPIPS immediately collapses back toward the `0.55+` band

Current expectation:

- if the main bottleneck was uniform structure over-penalization, this should be the cleanest next mechanism-level rescue

Outcome:

- training completed cleanly through `epoch_0020`
- remote train-side VRAM stayed comfortably under the paper cap:
  - epoch-20 train log reports `peak=3.44/4.17GB`
- deferred eval completed for `epoch_0014` through `epoch_0020`

Repro / frozen baseline:

- confirmed pre-continue anchor packet:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/inmortal-exp/aaai2027_inmortal_xpred_kmanifold_pattn_anisostokes_queue_from_pattn_seed42_b8a2/repro_packet_best_e13_source_config_frozen.zip`
- anchor being protected before this continuation:
  - `epoch_0013 = 0.7102 / 0.4603` on the `transfer` surface
  - full/all-pairs `= 0.7303 / 0.4559`

Observed continuation results:

- `epoch_0014`: transfer `0.6945 / 0.5780`, full `0.6997 / 0.5736`
- `epoch_0015`: transfer `0.6923 / 0.5810`, full `0.6975 / 0.5762`
- `epoch_0016`: transfer `0.6933 / 0.5942`, full `0.6987 / 0.5894`
- `epoch_0017`: transfer `0.7100 / 0.5746`, full `0.7173 / 0.5699`
- `epoch_0018`: transfer `0.6956 / 0.6041`, full `0.7006 / 0.5978`
- `epoch_0019`: transfer `0.6999 / 0.6140`, full `0.7050 / 0.6079`
- `epoch_0020`: transfer `0.6936 / 0.6590`, full `0.6971 / 0.6515`

Reading:

- edge-gated anisotropic pressure did recover a little style around `epoch_0017`
- but it did not preserve the low-LPIPS frontier of the parent `AnisoStokesQueue` anchor
- best continuation point was `epoch_0017`, and even that is still worse than the protected anchor:
  - style is effectively tied (`0.7100` vs `0.7102`)
  - LPIPS is dramatically worse (`0.5746` vs `0.4603`)

Decision:

- negative closure
- do not promote
- keep `epoch_0013` from the parent `AnisoStokesQueue` line as the paper-facing low-LPIPS point
