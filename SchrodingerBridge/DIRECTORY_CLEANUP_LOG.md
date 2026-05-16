# Directory Cleanup Log

## 2026-05-16: Pass 1

### Objective

Make the active paper, active theory, and active experimental record obvious before any model-side work resumes.

### Confirmed Canonical Paths

- Current paper PDF:
  `paper_orchestra_workspace/aaai_submission/paper_aaai2026.pdf`
- Current paper source:
  `paper_orchestra_workspace/aaai_submission/paper_aaai2026.tex`
- Theory:
  `maths/`
- Experiment narrative:
  `EXPERIMENT_LOG.md`

### Findings

1. `PROJECT_OVERVIEW.md` was stale and still implied experiment staging folders were the current main line.
2. `paper_refine_v2/` contains useful history, but it duplicates paper assets and is not the canonical manuscript workspace.
3. The root contains archive-like payloads and packaging outputs that do not need to stay in the top-level view.

### Keep In Root

- `src/`
- `maths/`
- `paper_orchestra_workspace/`
- `ablation_destructive_7epoch/`
- `S-add__K-1_C-0_W-20_Col-0/`
- `EXPERIMENT_LOG.md`
- `EXPERIMENT_PLAN.md`
- `PROJECT_OVERVIEW.md`
- `DIRECTORY_CLEANUP_LOG.md`

### Treat As Historical / Secondary

- `paper_refine_v2/`
- `paper_aaai_draft/`
- `paper_cn/`
- `kinetic_sweep/`
- `omf_sweep/`
- `omf_sweep2/`
- `omf_sweep3/`
- `weight_sweep_40/`
- `screening_grid_3epoch_108/`
- `review_additional_experiments/`

### Planned Cleanup Actions

- Move large packaged snapshots out of the root into an archive folder.
- Remove clear cache/byproduct files only when they are not part of user-edited work.
- Keep experiment evidence directories intact.

### Principle

The immediate need is clarity, not aggressiveness. Archive first, delete later only when there is no ambiguity.
