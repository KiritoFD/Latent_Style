# attn_pnp_selfinject Relaunch Prep

Date: 2026-06-11

Purpose:

- keep the next `attn_pnp_selfinject` decision path explicit
- make clear that this family is now the immediate next non-DINO relaunch candidate after `attn_gated_spade` closure

## Current Read

- current manifest status:
  - `planned`
- why this family is still worth keeping around:
  - it has a real canonical curve with image-backed deep-review candidates
  - it showed a recoverable style/LPIPS tradeoff rather than immediate collapse
- why it is now the immediate next relaunch:
  - `attn_gated_spade` is formally converged and moved to `reviewing`
  - `attn_gw_ot` remains a weak strict-band fit under the current host state
  - `attn_pnp_selfinject` now has:
    - explicit `switch_smoke_status=ok`
    - a real canonical calibration curve
    - a clear next segmented launch recipe

## Recommended Order

1. `attn_pnp_selfinject`
2. if it fails to become a clean formal lane again, reconsider `attn_gw_ot`
3. keep tokenizer `DINO` tail blocked until the remaining non-DINO structure families are intentionally exhausted

Rationale:

- `attn_pnp_selfinject` still requires segmented non-concurrent train/eval orchestration.
- But it now has better empirical upside than restarting `attn_gw_ot` blindly under the same strict contract.
- The remaining risk is operational rather than conceptual, so it is the best next family to push.

## Prelaunch Checks

1. Confirm no family remains `running`:

```powershell
python SchrodingerBridge\tools\experiments\audit_round1_queue_state.py
```

2. Confirm `attn_pnp_selfinject` now appears in:

- `planned`

3. If the manifest ever drifts away from that state and we intentionally want to reopen it again:

```powershell
python SchrodingerBridge\tools\experiments\retag_round1_manifest_family.py `
  --family-id attn_pnp_selfinject `
  --decision-status planned `
  --if-current-status recalibration_needed
```

## Launch Caution

- do not treat this as a normal direct relaunch candidate by default
- if reopened, prefer the segmented orchestration entrypoint and keep train/eval non-concurrent:

```powershell
python SchrodingerBridge\tools\experiments\run_remote_round1_family_segmented.py `
  --family-id attn_pnp_selfinject `
  --segment-epochs 1
```

## Guardrails

- keep the one-lane rule:
  - no concurrent remote train with remote fast-eval for the same family
- keep strict paper-facing thresholds for any new formal claim:
  - preferred band `9.0-10.8 GiB`
  - hard cap `11.3 GiB`
- if the next segmented retry still needs `runtime_guard_min_mode=warn` to survive, keep it as calibration evidence only rather than silently promoting it as a formal lane
