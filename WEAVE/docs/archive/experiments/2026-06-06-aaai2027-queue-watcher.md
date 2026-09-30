# AAAI2027 Queue Watcher

Date: 2026-06-06

Purpose:

- remove the next manual gap after the latent `SaMam -> A1` handoff
- keep the remote `3060` on a strict single-lane schedule while continuing the
  mainline improvement queue
- make the reviewed `A1 -> A2` order executable from local CPU only

## Helper

- [watch_remote_aaai2027_queue.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_remote_aaai2027_queue.py)

## Default queue

The default queue starts from `A1` and then continues:

1. `A2_softterm18_sem010`
2. `A2_softterm18_sem012`
3. `A2_softterm16_sem012`

This keeps the queue bounded to the narrow same-family improvement sweep first.

Optional controls can be appended later with:

- `--include-controls`

which adds:

1. `C1` pairing-cache random-pairing control
2. `C2` projection-count sensitivity

## What it watches

For `A1` and every later queued config, the watcher checks:

1. process exists for the exact `src/run.py --config ...` launch
2. remote log exists and is non-empty
3. first health heartbeat after `30s`
4. process exit before launching the next packet
5. remote total `memory.used <= 1500 MiB` before the next launch

Current note:

- the first live queue watcher instance was started before the later
  WSL-based `tail` patch landed
- that old instance still advances the queue correctly
- only its stdout log tails are noisy because they still use the old non-WSL
  tail path
- restart the watcher after the current live lane finishes if clean log tails
  are needed for the next packet

## Boundary

- this helper does not overlap runs
- it only continues the queue after the current packet exits
- it is local CPU orchestration only and does not use the local GPU
- the remote formal cap remains:
  - `< 11.0 GiB`

## Suggested detached launch

```powershell
$outLog = "G:\GitHub\Latent_Style\SchrodingerBridge\_codex_tmp\watch_remote_aaai2027_queue.out.log"
$errLog = "G:\GitHub\Latent_Style\SchrodingerBridge\_codex_tmp\watch_remote_aaai2027_queue.err.log"
$pidFile = "G:\GitHub\Latent_Style\SchrodingerBridge\_codex_tmp\watch_remote_aaai2027_queue.pid"
$proc = Start-Process -FilePath python -ArgumentList "-u","SchrodingerBridge\tools\experiments\watch_remote_aaai2027_queue.py" -WorkingDirectory "G:\GitHub\Latent_Style" -RedirectStandardOutput $outLog -RedirectStandardError $errLog -WindowStyle Hidden -PassThru
Set-Content -Path $pidFile -Value $proc.Id
```

## Current intended use

Use this watcher together with:

- [watch_remote_latent_samam_handoff.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/watch_remote_latent_samam_handoff.py)

The first watcher gets us from latent `SaMam` into `A1`.
The second watcher then keeps the mainline queue moving after `A1` starts and
finishes.
