# attn_gated_spade Relaunch Prep

Date: 2026-06-11

Purpose:

- make the next formal `attn_gated_spade` attempt executable without re-reading old launch history
- keep it positioned as the second non-DINO relaunch candidate after `attn_gw_ot`

## Current Read

- current manifest status:
  - `recalibration_needed`
- why this family is still in play:
  - it already owns a meaningful directional fast-eval curve through `epoch_0022`
  - it is a smoke-ok non-DINO backbone family
- why it is not formal evidence yet:
  - prior training lived too close to or below the requested formal VRAM floor
  - the live train pid disappeared before a clean formal closure packet was complete

## Recommended Order

1. first reopen `attn_gw_ot`
2. if `attn_gw_ot` is explicitly dropped or fails another formal attempt, reopen `attn_gated_spade`
3. only after those options are resolved, consider tokenizer `DINO` tail through explicit override

## Prelaunch Checks

1. Confirm no family remains `running`:

```powershell
python SchrodingerBridge\tools\experiments\audit_round1_queue_state.py
```

2. Confirm `attn_gated_spade` still appears in:

- `recalibration_needed`
- `relaunchable_non_dino`

3. Re-open it in the manifest:

```powershell
python SchrodingerBridge\tools\experiments\retag_round1_manifest_family.py `
  --family-id attn_gated_spade `
  --decision-status planned `
  --if-current-status recalibration_needed
```

4. Re-audit queue state:

```powershell
python SchrodingerBridge\tools\experiments\audit_round1_queue_state.py
```

Expected read:

- `planned_non_dino` should include `attn_gated_spade`
- `next_queue_candidate_if_running_clears` should become `attn_gated_spade` if `attn_gw_ot` is not also restored to `planned`

## Launch Options

Generic queue path:

```powershell
python SchrodingerBridge\tools\experiments\run_round1_family_queue.py
```

Direct launch path:

```powershell
python SchrodingerBridge\tools\experiments\launch_remote_round1_family_train.py `
  --config SchrodingerBridge\configs\aaai2027\round1_full_sweep\aaai2027_round1_attn_gated_spade_seed42_b8a2.json
```

## Guardrails

- do not re-promote this family while `solver_unsb_cycle` still has `decision_status=running`
- treat the next launch as a fresh formal attempt, not a continuation of the earlier nonformal line
- require the same paper-facing memory rules:
  - preferred `9.0-10.8 GiB`
  - hard stop above `11.3 GiB`
- if the next opening is still under-band or unstable, drop it again instead of letting the queue drift into an accidental tokenizer lane
