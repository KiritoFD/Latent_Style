# attn_pnp_selfinject Relaunch Prep

Date: 2026-06-11

Purpose:

- keep the next `attn_pnp_selfinject` decision path explicit
- keep the successful relaunch recipe and the reasons it mattered after the family has now moved to `reviewing`

## Current Read

- current manifest status:
  - `reviewing`
- why this family is still worth keeping around:
  - it has a real canonical curve with image-backed deep-review candidates
  - it showed a recoverable style/LPIPS tradeoff rather than immediate collapse
- why this note still matters:
  - this family only became a real formal lane after a multi-step calibration path
  - future reopen decisions should not forget that:
    - `batch=20` strict stayed just under the floor
    - `batch=20` warn-path recovered the Pareto surface
    - `batch=21` strict finally produced the formal in-band converged lane

## Historical Outcome

- `attn_pnp_selfinject` no longer needs an immediate relaunch decision:
  - the family is already closed for round-1 training
  - current status is `reviewing`
- the useful preserved lesson is the recipe:
  - segmented non-concurrent train/eval
  - detached local controller
  - final formal batch `= 21`

## Prelaunch Checks

1. Confirm no family remains `running`:

```powershell
python SchrodingerBridge\tools\experiments\audit_round1_queue_state.py
```

2. Confirm `attn_pnp_selfinject` now appears in:

- `reviewing`

3. If the family ever needs to be reopened in a later round:

```powershell
python SchrodingerBridge\tools\experiments\retag_round1_manifest_family.py `
  --family-id attn_pnp_selfinject `
  --decision-status planned `
  --if-current-status reviewing
```

## Launch Caution

- do not treat this as a normal direct relaunch candidate by default
- if reopened, prefer the segmented orchestration entrypoint and keep train/eval non-concurrent:
- historical successful formal attempt:
  - canonical batch `= 21`
  - strict `runtime_guard_min_mode=stop`
  - `health_wait_seconds = 60`

```powershell
python SchrodingerBridge\tools\experiments\run_remote_round1_family_segmented.py `
  --family-id attn_pnp_selfinject `
  --segment-epochs 1 `
  --health-wait-seconds 60 `
  --min-runtime-slack-mib 512
```

## Guardrails

- keep the one-lane rule:
  - no concurrent remote train with remote fast-eval for the same family
- keep strict paper-facing thresholds for any new formal claim:
  - preferred band `9.0-10.8 GiB`
  - hard cap `11.3 GiB`
- if a future reopen ever again needs `runtime_guard_min_mode=warn` to survive, keep that path as calibration evidence only rather than silently promoting it as a formal lane
