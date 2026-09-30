# WEAVE

This directory is the active, portable implementation used for submission
reproduction and architecture experiments. The method trains from scratch;
the canonical pipeline does not use a frozen adapter or image/latent
post-processing.

## Reproduce

Both local and remote machines use the same project-relative layout:

```text
data/train/          packed training latents and pairing cache
data/test/           150-image evaluation board
runs/cache/hf/       untracked model cache
runs/submission/     untracked checkpoints, images, metrics, and logs
```

From this directory:

```powershell
# Train 15 epochs from a fresh initialization, then evaluate every checkpoint.
powershell -ExecutionPolicy Bypass -File scripts/run_submission_repro.ps1

# Reuse existing checkpoints and evaluate missing epochs only.
powershell -ExecutionPolicy Bypass -File scripts/run_submission_repro.ps1 -EvalOnly
```

The canonical files are:

- `config.json`: model, objective, optimizer, data, and checkpoint settings.
- `inference.json`: fixed 8-step evaluation protocol.
- `run.py`: root training entry point.
- `scripts/batch_eval_all.py`: per-epoch DINO-S, CLIP-S, DINO-C, and LPIPS evaluation.

## Current Baseline

> This section describes the **base model** (873,680 parameters, `config.json` alone).
> The paper model adds the oriented target-HF route via
> `experiments/architecture/hf_oriented_internal_early_stop.json` (1,037,087 parameters, internal
> stop at epoch 4: DINO-S 0.4918 / CLIP-S 0.7128 / LPIPS 0.2595 / DINO-C 0.8102). Provenance of
> every paper number: `../docs/experiments/02_results_of_record.md`.

The clean 15-epoch base-model reproduction selected epoch 6:

| DINO-S | CLIP-S | LPIPS | DINO-C |
|---:|---:|---:|---:|
| 0.4867 | 0.7074 | 0.2508 | 0.8280 |

DINO-S is the primary style metric. CLIP-S is secondary; DINO-C and LPIPS
reject style gains caused by content collapse. No mixed selection score is
used.

## Documentation

- `../docs/experiments/README.md`: repository-wide experiment index and results of record (start here).
- `icme2027/`: current ICME paper and supplement; `aaai2027_v4/`: frozen AAAI 2027 submission.
- `docs/713/SUBMISSION_HANDOFF_2026-07-15.md`: current repository, remote, method, and experiment handoff.
- `docs/reproduction/baseline_reproduction.md`: baseline protocol and full per-epoch provenance.
- `docs/reproduction/root_layout_equivalence.md`: old-to-root implementation equivalence.
- `docs/reproduction/hf_oriented_nohh_result.md`: latest from-scratch architecture result.
- `archives/README.md`: historical material retained for provenance only.

The active Python package lives at the project root. The former `src/` tree,
legacy launchers/configs/tests, and rejected post-processing experiments are
archived and must not be used for new submission runs.
