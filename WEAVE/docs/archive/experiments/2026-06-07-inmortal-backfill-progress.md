# inmortal Backfill Progress

Date: 2026-06-07

Scope:

- supplement historical `inmortal` runs with per-checkpoint `CLIP-S + LPIPS`
- keep a stage summary CSV and a missing-checkpoint CSV for ongoing audit
- use the remote `3060 WSL` artifact surface only

Current authoritative CSVs:

- [2026-06-07-inmortal-stage-summary.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-07-inmortal-stage-summary.csv)
- [2026-06-07-inmortal-missing-fast-eval.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-07-inmortal-missing-fast-eval.csv)
- [inmortal-exp-manifest.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/inmortal-exp-manifest.csv)

Bundle-root convention:

- remote aggregate root:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/inmortal-exp`
- legacy run dirs are linked into that bundle root for audit and summary generation

Progress snapshot at the current stop point:

- `aaai2027_inmortal_k_manifold_seed42_b16`
  - fast eval backfill complete: `8 / 8`
  - current selected point in the stage summary:
    - `epoch_0006`
    - `clip_style = 0.6789`
    - `content_lpips = 0.3349`
- `aaai2027_inmortal_k_spatial_seed42_b16`
  - fast eval backfill partially complete: `6 / 8`
  - current selected point in the stage summary:
    - `epoch_0006`
    - `clip_style = 0.6787`
    - `content_lpips = 0.3505`
- remaining missing checkpoints when this note was written:
  - `K_spatial b16`: `2`
  - `XPred_Bary b16`: `8`
  - `XPred_Bary b40`: `8`
  - `late-Stokes 0.05`: `1`
  - total remaining: `19`

Important interpretation boundary:

- some older run-local snapshots are incomplete and fail under `mainline-on-run-local`
  because they are missing `utils.diffeomorphic`
- for those historical runs, the backfill workflow falls back to `mainline`
- therefore these new per-checkpoint curves are valid for training-curve analysis and
  gap detection, but they should not silently overwrite the existing paper-safe
  promoted headline points without an explicit audit

Operational stop condition that interrupted continuation:

- after stopping the WSL backfill processes, remote `nvidia-smi` still showed:
  - about `11127 MiB / 12288 MiB`
  - about `98%` utilization
- the remaining load was on Windows-side `C+G` processes rather than WSL Python
- under that host-GUI state, continuing the backfill would violate the current
  `11.5 GiB` machine-safety rule

Next resume action:

1. wait until remote total GPU memory falls back into a safe idle band
2. relaunch the remaining `19` missing checkpoints with a smaller eval batch
3. rebuild the stage summary CSV and missing CSV again after the resumed pass

## Automated recovery lane

An automated remote recovery lane is now armed so this work can continue without
manual babysitting once the host-side GUI load drops.

Remote launcher:

- task name:
  - `SB-Resume-Inmortal-Backfill-When-Idle`
- remote log:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/_codex_tmp/resume_inmortal_backfill_when_idle.log`

Behavior:

- poll `nvidia-smi` inside WSL
- wait until total GPU memory falls to `<= 2000 MiB`
- then resume the remaining missing fast-eval checkpoints with a safer profile:
  - `batch_size = 4`
  - `vae_decode_batch_size = 12`
  - `target_chunk_size = 1`
- target runs:
  - `aaai2027_inmortal_k_spatial_seed42_b16`
  - `aaai2027_inmortal_xpred_bary_seed42_b16`
- `aaai2027_inmortal_xpred_bary_seed42_b40`
- `aaai2027_inmortal_xpred_kmanifold_pattn_stokes_from_pattn_seed42_b16`

## Automated next-train queue

A second host-owned remote runner is also armed so that the experiment program
can continue automatically after the remaining backfill gaps close.

Remote launcher:

- task name:
  - `SB-Run-Inmortal-Queue-When-Ready`
- remote log:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/_codex_tmp/run_inmortal_packet_queue_when_ready.log`

Queue behavior:

- wait until:
  - remote GPU memory falls to `<= 2000 MiB`
  - `2026-06-07-inmortal-missing-fast-eval.csv` is empty
  - no WSL backfill/eval process is alive
- then start the queued training packets in order:
  1. `inmortal_k_spectral_seed42_b16.json`
  2. `inmortal_xpred_structot_seed42_b16.json`
  3. `inmortal_xpred_teacher_endpoint_seed42_b16.json`
  4. `inmortal_xpred_queue_seed42_b16.json`
  5. `inmortal_xpred_kmanifold_pattn_queue_seed42_b16.json`
  6. `inmortal_xpred_kmanifold_pattn_anisostokes_queue_from_pattn_seed42_b8a2.json`

## State after backfill closure

The historical fast-eval backfill has now closed:

- `2026-06-07-inmortal-missing-fast-eval.csv` is empty
- `run_inmortal_packet_queue_when_ready.py` has already advanced from wait mode into
  the first formal queued packet

Current active formal packet:

- safety-corrected `K_spectral` rerun:
  - [inmortal_k_spectral_seed42_b12.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_k_spectral_seed42_b12.json)

Meaning:

- the project is no longer in the legacy backfill closure phase
- GPU time has returned to the next `inmortal` mechanism family
