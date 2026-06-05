# Weekly Model Improvement Tracker

Date: 2026-06-05

Scope:

- one-week autonomous execution plan for AAAI 2027 paper hardening
- remote `RTX 3060` WSL only
- priority order:
  - improve our model
  - repeat and validate our model
  - add cheap reviewer-directed controls
  - keep latent/baseline work bounded

Primary plan:

- `docs/plans/2026-06-05-one-week-model-improvement-plan.md`

## Current keep / stop policy

Keep spending formal remote budget on:

1. executor-side model improvement
2. narrow softening / stability sweeps on the current paper-facing family
3. repeatability packets for any promoted improved point
4. cheap reviewer controls tied to concrete questions

Stop spending formal remote budget on:

- more endpoint-only reruns
- more semantic-vs-random reruns
- broad speed rhetoric experiments
- turning `latent SaMam` into the main lane

## Day 1 packet

### A1. Executor-side promotion

Intent:

- port the landed `executor-only` localization signal onto the paper-facing
  `H` family instead of opening a new tokenizer family

Config:

- `configs/aaai2027/executor_promotion_h_e1_seed42_b44.json`

Checkpoint source:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote/epoch_0001.pt`

Training rule:

- load only:
  - `style_tokenizer.*`
  - `style_spatial_id_16`
- train only the executor side

Status:

- `prepared`

### A2. Narrow mainline softening sweep

Reasoning:

- the current same-family path packet says weakening `w_kinetic` hurts quality,
  so this sweep does **not** reduce kinetic
- the softening lane therefore changes only:
  - terminal endpoint pressure
  - semantic routing temperature

Configs:

- `configs/aaai2027/mainline_h_softterm18_sem010_seed42_b44.json`
- `configs/aaai2027/mainline_h_softterm18_sem012_seed42_b44.json`
- `configs/aaai2027/mainline_h_softterm16_sem012_seed42_b44.json`

Shared policy:

- keep `w_kinetic = 1.0`
- use the reviewed Distinct5 full-eval contract
- compare against current `H e1/e2` and `F e1`

Status:

- `prepared`

## Near-term comparison anchors

Primary anchors already in the ledger:

- `LBM-F e1`
- `LBM-H e1`
- `LBM-H e2`
- `LBM-K e1`

Use these as the first comparison surface before touching broader baseline work.

## Side quest budget

`latent SaMam`:

- allowed only as a short smoke lane later in the week
- not allowed to block the main improvement queue

## Next log entries to add

When the next packet launches, append:

1. exact remote launcher / command
2. save dir
3. GPU memory band
4. first-health heartbeat
5. completion or failure note

## 2026-06-05 preflight snapshot

What is already confirmed:

- remote host `100.115.18.62:2222` is reachable
- remote WSL project root resolves cleanly to:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge`
- Windows-side GPU snapshot before launch attempt:
  - `memory.used = 991 MiB / 12288 MiB`
  - `utilization.gpu = 18%`
  - `power.draw = 15.16 W`

What is not yet trusted:

- a shell-safe one-line launch contract for this host
- path-truth for the exact reusable `H e1` checkpoint under the current remote
  workspace surface

Immediate next ops task:

- add a small remote launch / preflight wrapper that avoids the current
  PowerShell + SSH + WSL quoting ambiguity before the first formal `A1` launch

## 2026-06-06 live update

Remote latent side quest currently occupying the only allowed GPU lane:

- active run:
  - `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_latent_legacy256_probe4`
- current GPU usage snapshot:
  - `7460 MiB / 12288 MiB`
- latest observed training progress:
  - around `Epoch 0 step 863`
- retained checkpoint status:
  - none yet
  - first save still waits for `step 5000`
- state persistence status:
  - `step_checkpoints/last.ckpt` now exists
  - retained numbered checkpoints still have not reached the first `5000-step`
    boundary

Interpretation:

- `latent SaMam` is healthy enough to keep running as a bounded side quest
- but it is currently blocking the main `A1/A2` queue because the remote `3060`
  must stay below the hard single-run budget

Ops progress landed during this update:

- added reviewed remote launcher helper:
  - [launch_remote_aaai2027_packet.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/launch_remote_aaai2027_packet.py)
- launcher purpose:
  - sync the current `src` plus `configs/aaai2027`
  - write a remote `_codex_tmp/*.sh` launch script
  - run targeted remote `py_compile`
  - start the packet via remote `schtasks`

What this unblocks:

- once the current `SaMam` lane yields a checkpoint or is explicitly paused,
  `A1` can be launched without reopening the earlier SSH + WSL quoting failure
  class

Additional review-risk reduction landed during this update:

- implementation clarity packet:
  - [2026-06-06-distinct5-implementation-clarity.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-06-distinct5-implementation-clarity.md)
- this note records the current Distinct5-512 train root, eval root, VAE
  preset, latent-scale contract, style list, and prototype-pairing cache
  contract for the active paper-facing `H` family
