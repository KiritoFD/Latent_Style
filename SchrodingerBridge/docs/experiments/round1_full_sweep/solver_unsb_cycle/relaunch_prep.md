# solver_unsb_cycle Relaunch Prep

Date: `2026-06-11`

## Why This Exists

- `solver_unsb_cycle` stopped before convergence.
- the latest retained checkpoint is `epoch_0014`
- the train log ended at:
  - `2026-06-11 13:33:06 +08:00`
  - `rc=143`
- because the family already has a real partial curve, the next move should be
  continuation from `epoch_0014`, not a fresh from-parent restart

## Recommended Resume Path

- use segmented remote continuation:
  - it auto-discovers the latest retained remote checkpoint
  - it writes a temporary launch config with:
    - `resume_checkpoint = latest_ckpt_remote`
    - `resume_training_state = true`
    - `resume_optimizer = false`
  - it keeps train and fast-eval non-concurrent when the controller is used in
    the intended bounded way

## Safe First Resume

Preconditions:

- no foreign GPU-heavy job is occupying the remote 3060 lane
- the family manifest row is not left at stale `running`
- remote fast-eval packet root still resolves to the same run

Suggested first bounded resume:

```powershell
python SchrodingerBridge\tools\experiments\run_remote_round1_family_segmented.py `
  --family-id solver_unsb_cycle `
  --segment-epochs 2 `
  --manifest-decision-status-on-start running
```

Why `segment-epochs 2` first:

- the family is already beyond its first meaningful frontier point
- the next decision hinge is whether the post-`epoch_0014` continuation can
  move back toward `epoch_0009`
- a short bounded continuation gives two more retained checkpoints before
  reopening a longer lane commitment

## After Resume

1. refresh the remote fast-eval packet
2. verify all new retained checkpoints received `CLIP-S + LPIPS`
3. re-check:
   - `since_last_pareto`
   - `tail_flat`
   - whether a new Pareto point appears after `epoch_0009`
4. only reopen post-UNSB queue handoff logic if this family truly closes
