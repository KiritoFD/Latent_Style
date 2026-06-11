# Post UNSB Queue Handoff Plan

Date: 2026-06-11

Purpose:

- make the post-`solver_unsb_cycle` queue behavior explicit
- prevent the generic queue from silently falling into the tokenizer `DINO` tail
- keep the next round-1 launch aligned with the current paper-facing non-DINO-first rule

## Suspension Note

- this handoff plan was written under the assumption that `solver_unsb_cycle`
  would actually close
- the later interruption audit showed the train log ending at:
  - `2026-06-11 13:33:06 +08:00`
  - `rc=143`
- with the latest retained checkpoint still at `epoch_0014`
- until `solver_unsb_cycle` is either resumed to closure or explicitly dropped,
  this file should be read as queued contingency only, not as the current next
  action

## Current Queue Fact Pattern

- active formal lane:
  - `solver_unsb_cycle`
- current queue helper behavior:
  - `run_round1_family_queue.py` now blocks DINO-tail families by default
  - if no non-DINO family is restored to `planned`, the queue will stop with a non-launchable notice after `solver_unsb_cycle` closes
- current non-DINO candidates that can be explicitly re-promoted:
  - `attn_gw_ot`
  - `attn_gated_spade`
- current non-DINO family that still needs more calibration before any re-promotion:
  - `attn_pnp_selfinject`

## Recommended Order

1. `attn_gw_ot`
2. `attn_gated_spade`
3. only after those are closed or explicitly dropped, consider reopening tokenizer `DINO` tail families

Rationale:

- `attn_gw_ot` already has real round-1 evidence and stays closer to the main backbone/solver sweep than reopening tokenizer `DINO` work immediately.
- `attn_gated_spade` is also a non-DINO architecture family with existing formal evidence, but its previous lane was less stable than `attn_gw_ot`.
- `attn_pnp_selfinject` still reads like a segmented recalibration line rather than the cleanest next formal lane.
- tokenizer `DINO` families remain valuable, but they are still the least elegant next step while the mainline non-DINO sweep has unfinished branches.

## Safe Commands

Inspect current queue state:

```powershell
python SchrodingerBridge\tools\experiments\audit_round1_queue_state.py
```

Re-promote `attn_gw_ot` into `planned` after `solver_unsb_cycle` closure:

```powershell
python SchrodingerBridge\tools\experiments\retag_round1_manifest_family.py `
  --family-id attn_gw_ot `
  --decision-status planned `
  --if-current-status recalibration_needed
```

Re-promote `attn_gated_spade` instead:

```powershell
python SchrodingerBridge\tools\experiments\retag_round1_manifest_family.py `
  --family-id attn_gated_spade `
  --decision-status planned `
  --if-current-status recalibration_needed
```

If we just want the first smoke-ok non-DINO relaunch candidate promoted automatically:

```powershell
python SchrodingerBridge\tools\experiments\promote_next_round1_non_dino_candidate.py
```

Only if we intentionally want tokenizer `DINO` tail through the generic queue:

```powershell
python SchrodingerBridge\tools\experiments\run_round1_family_queue.py --allow-dino-tail
```

## Guardrails

- Do not re-promote any family while `solver_unsb_cycle` still has `decision_status=running`.
- Do not rely on queue defaults alone to preserve non-DINO-first once the manifest contains only tokenizer `planned` rows.
- After any retag, re-run the queue audit helper before actual launch.
- Keep `attn_pnp_selfinject` out of the immediate next slot unless its calibration state is explicitly re-evaluated and promoted on purpose.
