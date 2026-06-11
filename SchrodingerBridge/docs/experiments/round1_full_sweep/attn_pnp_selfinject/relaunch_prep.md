# attn_pnp_selfinject Relaunch Prep

Date: 2026-06-11

Purpose:

- keep the next `attn_pnp_selfinject` decision path explicit
- make clear that this family is currently a later non-DINO candidate, not the immediate next relaunch

## Current Read

- current manifest status:
  - `recalibration_needed`
- why this family is still worth keeping around:
  - it has a real canonical curve with image-backed deep-review candidates
  - it showed a recoverable style/LPIPS tradeoff rather than immediate collapse
- why it is not the immediate next relaunch:
  - its earlier formal path depended on segmented non-concurrent train/eval orchestration
  - it is operationally messier than `attn_gw_ot` or `attn_gated_spade`

## Recommended Order

1. `attn_gw_ot`
2. `attn_gated_spade`
3. only then consider reopening `attn_pnp_selfinject`

Rationale:

- `attn_pnp_selfinject` still reads as a segmented calibration family rather than the cleanest next formal lane.
- The previous evidence is useful, but the relaunch path is more complex and easier to mis-handle than the other two non-DINO candidates.

## Prelaunch Checks

1. Confirm no family remains `running`:

```powershell
python SchrodingerBridge\tools\experiments\audit_round1_queue_state.py
```

2. Confirm `attn_pnp_selfinject` still appears in:

- `recalibration_needed`

3. If we intentionally want to reopen it anyway:

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
  --family-id attn_pnp_selfinject
```

## Guardrails

- do not re-promote while `solver_unsb_cycle` still has `decision_status=running`
- do not reopen it as the first fallback just because tokenizer `DINO` tail is blocked
- prefer it only after the cleaner non-DINO relaunch paths have been tried or explicitly rejected
