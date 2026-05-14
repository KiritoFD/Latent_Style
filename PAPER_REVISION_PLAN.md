# AAAI 2026 Paper Revision Plan

Last updated: 2026-05-13
Target paper: `SchrodingerBridge/paper_orchestra_workspace/aaai_submission/paper_aaai2026.tex`

## Goal

Upgrade the current AAAI draft from a coherent internal draft to a submission-ready paper with:

- tighter claims
- stronger experimental evidence
- transparent baseline coverage
- reproducible figures/tables/captions

## Status Snapshot

- `completed`: repository audit and paper-source audit
- `in_progress`: revision plan + evidence mapping
- `pending`: paper rewrites, figure/table upgrades, protocol-aligned justification text, compile pass

## Completed

### Repo and paper audit

- Located the active AAAI submission source at `SchrodingerBridge/paper_orchestra_workspace/aaai_submission/`.
- Confirmed the compiled PDF and `.tex` source are in sync enough for direct editing.
- Identified the current hard blockers in the draft:
  - unbounded `Fast` wording
  - over-strong `Bridge` naming/theory framing
  - incomplete baseline table wording
  - missing failure-case section
  - incomplete efficiency table
  - artifact metrics not yet validated with a controlled diagnostic experiment

### Existing evidence already available in the repo

- Main strict-750 baseline results already exist for:
  - Ours
  - SaMST
  - StyleID
  - S2WAT
  - AdaIN variants
- Destructive ablations already exist and support:
  - terminal SWD drives style
  - kinetic regularization preserves content
  - strong color matching is harmful
- Weight/sweep assets already exist:
  - `weight_sweep_40`
  - `next_round_80`
  - `full_dimensional_orthogonal_sweep_20`
  - `step_size_sweep_epoch7`

### Baseline coverage audit

- Confirmed that `StyTr2`, `CAST`, `AesFA`, and `AesPA-Net` are not yet present as matched main-table strict-750 baselines.
- Confirmed the repo contains partial reproduction traces for missing baselines:
  - smoke runs
  - timing probes
  - partial output folders
- Confirmed current paper text still overstates their role and needs to be downgraded to transparent status reporting unless full matched evidence is recovered.

### Baseline review suite

- Added `Related_Works/run_511/launchers/run_review_baseline_suite.py`.
- This script sequentially:
  - runs each baseline launcher
  - records train/infer wall-clock time from launcher summaries
  - profiles params / FLOPs / peak VRAM on the inference path when the model can be instantiated
  - records blocking reasons when a baseline still cannot be reproduced cleanly
- Fixed `AesPA-Net` preflight so it accepts the actually present VGG checkpoint formats (`.pth`, `.t7`, `.pkl`) instead of only `.t7`.
- Iterated baseline launcher fixes after live smoke runs:
  - `AesFA` now uses a legal smoke/train batch size (`>=2`) and maps the newest training checkpoint to `main.pth` for test-time loading.
  - `AesFA` test-time `thop` import is shimmed so inference does not fail when `thop` is absent.
  - `AesPA-Net` launcher now disables `wandb`, passes the training result directory/comment back into test mode, uses `num_workers=0` for test, and collects both `.jpg` and `.png` outputs.
  - The unified suite now runs launcher reproduction first and profiling second, avoiding cross-method import/GPU contamination.
- Smoke validation status:
  - `StyTR-2`: train + infer + profile pass
  - `CAST`: train + infer + profile pass
  - `AesFA`: train + infer + profile pass
  - `AesPA-Net`: train + infer + profile pass
- Full-run progress on `review_baseline_suite_full4g`:
  - `StyTR-2` full `train + infer` completed and summary has been written.
  - `CAST` full `train + infer` completed and summary has been written.
  - `AesFA` full `train + infer` completed after adding periodic checkpoint saving.
  - `AesPA-Net` full `train + infer` completed.
- Additional launcher hardening after the first full-run attempt:
  - `AesFA` now saves checkpoints periodically (`save_interval`) and can auto-resume from the latest `model_iter_*` checkpoint.
  - Baseline launchers now prefer `UV_PYTHON` over the shell-default `python`, preventing nested subprocesses from silently switching to an interpreter without `torch`.
  - Rechecked `AesFA` and `AesPA-Net` with fresh smoke runs after the interpreter fix; both still pass end to end.
  - Added a fallback profiler backend based on `torch.profiler(with_flops=True)` so the unified suite now records parameter count and FLOPs even when `thop` cannot trace a baseline cleanly.
  - Final unified review baseline table has been materialized at:
    - `Related_Works/run_511/outputs/review_baseline_suite_full4g/summary.csv`
    - `Related_Works/run_511/outputs/review_baseline_suite_full4g/summary.json`

### Server-ready review experiment runner

- Added `SchrodingerBridge/run_review_experiments.py`.
- This is now the main server script for the new review-driven experiments on our own model.
- It runs sequentially in one pass:
  - inference-step sweep
  - `lambda_kin / lambda_swd` grid training + evaluation
  - efficiency profiling
- Output root defaults to `SchrodingerBridge/review_additional_experiments/`.
- Local smoke check still requires clearing `PYTHONHOME` first if Python startup is abnormal.
- Dry-run command expansion has been validated end-to-end:
  - step sweep commands are emitted first
  - then the full lambda-grid train/eval queue
  - then the efficiency stage summary
- Review-added experiment status for our own model:
  - The nested result root `SchrodingerBridge/review_additional_experiments/review_additional_experiments/` contains a complete 5-point step sweep (`1/4/8/12/16`) and a complete `3 x 3 x 8-epoch` `lambda_kin / lambda_swd` grid.
  - Added `SchrodingerBridge/summarize_review_additional_experiments.py` to aggregate these assets without rerunning the large jobs.
  - Because the original result directory is partially write-restricted, the aggregate CSVs are written to:
    - `review_additional_experiments_aggregates/step_sweep_pareto.csv`
    - `review_additional_experiments_aggregates/lambda_grid_final_epoch.csv`
    - `review_additional_experiments_aggregates/lambda_grid_best_transfer_ec.csv`
    - `review_additional_experiments_aggregates/efficiency_profile.csv`
  - This aggregation also backfilled our model's efficiency profile for the review-added package.

### Paper source revision progress

- Updated `SchrodingerBridge/paper_orchestra_workspace/aaai_submission/paper_aaai2026.tex` to reflect the newly completed evidence:
  - tightened the title and abstract from unconditional `Fast` wording to efficiency-focused wording,
  - replaced the old baseline-status table entries for `StyTr2`, `CAST`, `AesFA`, and `AesPA-Net` with completed strict-750 core-metric/efficiency status,
  - expanded the main comparison table to include the newly completed baselines,
  - added a new `Step-count and lambda sensitivity` subsection using the aggregated review-added results,
  - replaced the old runtime table with a multi-dimensional efficiency table (Params / FLOPs / Peak VRAM / profile throughput / end-to-end throughput),
  - rewrote the discussion/limitations section to remove the now-obsolete "baselines unavailable" limitation.
- Compilation verification is still pending on this machine because no TeX toolchain (`latexmk`, `pdflatex`, `xelatex`, `tectonic`, `bibtex`) is currently installed in the shell environment.

### Historical internal runner

- `SchrodingerBridge/run_paper_internal_suite.py` remains available for older internal sweeps and ablations.
- It is no longer the preferred entry point for the new review-added experiment package.

## In Progress

### Revision strategy

- Map each required paper fix to one of three buckets:
  - `directly supported by existing evidence`
  - `supported by partial assets and needs a summary figure/table`
  - `not fully supported and must be transparently downgraded`
- Build a single server-launchable script for the new review experiments so the added evidence can be reproduced in one ordered run.

### Current decisions

- Tighten the framing from unconditional `Fast` to efficiency relative to heavier diffusion or training-free methods.
- Keep the method bridge-inspired unless a stronger theoretical appendix is added later.
- Move `EC` to an explicitly internal diagnostic role rather than a claimed community metric.
- Replace vague baseline-status wording with protocol-aligned inclusion/exclusion rules.

## Remaining

### Paper structure and writing

- Rewrite title, abstract, and introduction to constrain `Fast` and soften `Bridge`.
- Rewrite experiment setup so baseline inclusion rules are explicit and reproducible.
- Add a `Failure Cases / Boundary Analysis` subsection.
- Rewrite `Discussion and Limitations` with concrete, review-facing limitations.
- Expand figure captions with sampling, crop, resolution, and post-processing protocol details.

### Experiment packaging

- Build a clean sensitivity story from existing assets:
  - step-size / integration stability evidence
  - `lambda_kin` / `lambda_swd` grid evidence
  - temperature / patch-scale evidence
- Upgrade efficiency reporting:
  - params
  - FLOPs or MACs if available
  - peak VRAM
  - throughput assumptions and hardware notes
- Add a transparent baseline-status table with exclusion reasons for incomplete baselines.

### Artifact metric validation

- Check whether existing high-frequency ablation assets can support a controlled artifact-validity section.
- If not, add a lightweight synthetic-artifact validation script and summarize:
  - Gaussian noise sensitivity
  - checkerboard / dithering sensitivity
  - FFT slope / HF-Patch-KID response
  - weak correlation with LPIPS / CLIP

### Verification

- Recompile the AAAI paper after edits.
- Check for broken references, overfull boxes, and figure/table fit.
- Write a final remaining-risk note after the compile pass.

## Working Rules

- This file is the live root-level progress log for the revision work.
- Every meaningful completed step should be appended here or reflected by moving items across status sections.
