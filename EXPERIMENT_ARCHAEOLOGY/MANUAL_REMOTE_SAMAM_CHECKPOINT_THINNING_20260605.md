# Remote SaMAM Checkpoint Thinning Audit - 2026-06-05

Scope:

- Remote host: `administrator@100.115.18.62:2222`
- Remote run: `I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag`
- Target directory: `step_checkpoints`

This pass manually opened the central SaMAM Distinct5 curve directory and checked whether any checkpoints could be safely deleted. No remote deletion was performed in this pass.

## Directory State

Opened top-level evidence:

- `segmented.log`
- `eval_curve`
- `step_checkpoints`
- `run_segmented.sh`
- direct continue logs for steps 2000 and 2250
- wrapper logs for resume/3000 attempts

`step_checkpoints` contains 19 checkpoint files:

- 7 alias files: `last.ckpt`, `last-v1.ckpt` ... `last-v6.ckpt`
- 12 step files: `step-step=000250.ckpt` through `step-step=003000.ckpt`

Each file is about 275.9MB, so the directory is about 5.24GB.

## Evidence Opened

`eval_curve` contains:

- per-step image/metrics directories for steps 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, and 3000;
- `convergence_recovered.md`, which records the recovered curve for 250 through 2000;
- current `curve_metrics.csv`, now containing step 3000;
- ArtFID reuse directories for 2500, 2750, and 3000.

Important opened facts:

- `convergence_recovered.md` reports step 2000 as both best style and best LPIPS in the recovered 250-2000 curve.
- step 2250 exists as a later comparison point and is cited in existing comparison tables.
- step 3000 is the closed/last point with current `curve_metrics.csv`, `step_003000_artfid_reuse/summary.json`, and timing lines in `segmented.log`.
- `segmented.log` records `TRAIN_STEP_3000_WALL_SECONDS=3156.25` and `EVAL_STEP_3000_WALL_SECONDS=289.31`.
- `step_002500_artfid_reuse/summary.json` is zero bytes, so the 2500 checkpoint may still be needed if that ArtFID reuse has to be repaired.
- `step_002750_artfid_reuse` contains images only, so the 2750 checkpoint may still be needed if that reuse has to be completed.

## Alias Hash Check

I did not assume `last*.ckpt` files are duplicates. I compared SHA256 hashes for the natural pairs:

- `last.ckpt` vs `step-step=000250.ckpt`
- `last-v1.ckpt` vs `step-step=000500.ckpt`
- `last-v2.ckpt` vs `step-step=000750.ckpt`
- `last-v3.ckpt` vs `step-step=001000.ckpt`
- `last-v4.ckpt` vs `step-step=001250.ckpt`
- `last-v5.ckpt` vs `step-step=001500.ckpt`
- `last-v6.ckpt` vs `step-step=001750.ckpt`

All seven pairs have different SHA256 hashes. Therefore these alias files cannot be deleted as duplicate copies.

Detailed hashes are in:

- `manual_remote_samam_hash_pairs_20260605.csv`

## Decision

No SaMAM central checkpoint was deleted in this pass.

Reason:

- early and mid step checkpoints preserve the recovered convergence curve;
- step 2000 is the recovered best point;
- step 2250 is a cited comparison point;
- step 2500 and 2750 have incomplete/partial ArtFID reuse evidence, so checkpoints may be needed for repair;
- step 3000 is the closed/last point with timing and ArtFID reuse summary;
- `last*` alias files are not byte-identical duplicates.

## What Would Be Needed Before Thinning

A destructive thinning policy would need an explicit decision to sacrifice full curve rerun capability. A possible aggressive policy would keep only:

- step 2000;
- step 2250;
- step 3000;
- maybe step 2500/2750 until ArtFID reuse is repaired;
- alias files only if their role is proven unnecessary.

That would free several GB, but it would no longer preserve the complete SaMAM curve as retrainable checkpoint evidence. I did not apply that policy.
