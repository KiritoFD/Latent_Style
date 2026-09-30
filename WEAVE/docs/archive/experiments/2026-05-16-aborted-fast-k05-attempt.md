# 2026-05-16 Aborted Fast K0.5 Attempt

## Status

Aborted before any training batch completed. No experimental result should be recorded from this attempt.

## Intended Experiment

Resume from:

`S-add__K-1_C-0_W-20_Col-0/epoch_0007.pt`

with:

- `w_kinetic = 0.5`
- `swd_num_projections = 32`
- `semantic_swd_num_projections = 32`

## What Actually Happened

### Attempt 1

The run was launched from the repository root, which broke relative data paths. The dataset lookup failed before training started.

### Attempt 2

The run was relaunched from the correct `SchrodingerBridge/` working directory. Dataset loading succeeded, checkpoint resume succeeded, but Windows DataLoader worker processes failed before batch execution because of environment-level DLL / pagefile issues.

### Attempt 3

The config was adjusted locally to `num_workers = 0` so the environment issue could be bypassed. Before the resumed attempt produced any meaningful result, the user redirected work back to cleanup and documentation. The run was therefore intentionally stopped.

## Interpretation

- no style/content/timing conclusion should be drawn from this attempt
- the theory path remains only a documented hypothesis
- the next valid run should start from a clean, documented config after cleanup work is complete
