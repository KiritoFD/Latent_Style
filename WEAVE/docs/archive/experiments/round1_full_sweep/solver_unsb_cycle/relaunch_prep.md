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

## 2026-06-11 Recalibration Update

- the first segmented resume attempt from `epoch_0014` proved that resume
  wiring itself is healthy:
  - model resumed from `epoch_0014.pt`
  - `epoch=15`, `global_step=4410`
- but the old formal `batch=15` setting no longer satisfies the current remote
  band gate
  - 30-second health check observed only `7734 MiB`
  - launcher rejected it as under-band before allowing the lane to continue
- practical consequence:
  - `batch=15` is no longer the authoritative formal UNSB setting
  - the first upward retry at `batch=19` also failed:
    - runtime guard observed `11811 MiB`
    - this crossed the hard cap `11571 MiB`
    - the process was terminated before writing a new retained checkpoint
- the calibration bracket is therefore now:
  - `batch=15`: too low / not formally trustworthy
  - `batch=19`: too high / hard-cap failure
- the next retry should use the updated canonical batch:
  - `batch_size = 17`
- this is the current best paper-safe midpoint candidate

## Current Outcome

- `batch=17` has now been validated in real bounded continuation
- successful retained/eval extension:
  - from `epoch_0014` to `epoch_0018`
- current meaning of this file:
  - it remains the restart contract if the lane stops again
  - but it is no longer only a hypothetical resume plan

## After Resume

1. refresh the remote fast-eval packet
2. verify all new retained checkpoints received `CLIP-S + LPIPS`
3. re-check:
   - `since_last_pareto`
   - `tail_flat`
   - whether a new Pareto point appears after `epoch_0009`
4. only reopen post-UNSB queue handoff logic if this family truly closes
