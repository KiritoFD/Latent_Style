# attn_gw_ot Relaunch Prep

Date: 2026-06-11

Purpose:

- make the next formal `attn_gw_ot` attempt executable without re-reading old chat or log context
- keep the post-`solver_unsb_cycle` non-DINO-first path explicit

## Current Read

- current manifest status:
  - `recalibration_needed`
- why this family is still eligible:
  - it is a non-DINO backbone family with existing round-1 evidence
  - queue audit currently lists it as a relaunchable non-DINO candidate
- why it is not already formal:
  - earlier attempts crossed the hard cap or drifted under the formal floor
  - the current retained line is directional evidence, not closure-grade paper evidence

## Recommended Use

- if `solver_unsb_cycle` closes without promotion, and we still want to preserve the current non-DINO-first paper-facing order, `attn_gw_ot` is the first relaunch candidate.
- `attn_gated_spade` is the second fallback if we explicitly decide not to reopen `attn_gw_ot`.

## Prelaunch Checks

1. Confirm no family remains `running`:

```powershell
python SchrodingerBridge\tools\experiments\audit_round1_queue_state.py
```

2. Confirm `attn_gw_ot` still appears in:

- `recalibration_needed`
- `relaunchable_non_dino`

3. Re-open it in the manifest:

```powershell
python SchrodingerBridge\tools\experiments\retag_round1_manifest_family.py `
  --family-id attn_gw_ot `
  --decision-status planned `
  --if-current-status recalibration_needed
```

4. Re-audit queue state:

```powershell
python SchrodingerBridge\tools\experiments\audit_round1_queue_state.py
```

Expected read:

- `planned_non_dino` should include `attn_gw_ot`
- `next_queue_candidate_if_running_clears` should become `attn_gw_ot`

## Launch Options

Generic queue path:

```powershell
python SchrodingerBridge\tools\experiments\run_round1_family_queue.py
```

Direct launch path:

```powershell
python SchrodingerBridge\tools\experiments\launch_remote_round1_family_train.py `
  --config SchrodingerBridge\configs\aaai2027\round1_full_sweep\aaai2027_round1_attn_gw_ot_seed42_b8a2.json
```

## Guardrails

- do not re-promote `attn_gw_ot` while `solver_unsb_cycle` is still `running`
- treat the next `attn_gw_ot` launch as a fresh formal attempt, not a continuation of the earlier nonformal line
- require the same remote band discipline as every other formal family:
  - preferred `9.0-10.8 GiB`
  - hard stop above `11.3 GiB`
- if the next opening still reads under-band, demote again and move to `attn_gated_spade` instead of forcing tokenizer `DINO` tail prematurely
