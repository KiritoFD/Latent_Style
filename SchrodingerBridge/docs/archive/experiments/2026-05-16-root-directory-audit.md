# 2026-05-16 Root Directory Audit

## Purpose

Create one written inventory of the entire `SchrodingerBridge/` root before further theoretical or experimental work.

## High-Level Classification

The directory now has four functional layers:

1. Active core:
   `src/`, `maths/`, `paper_orchestra_workspace/`, `docs/`
2. Evidence:
   `S-add__K-1_C-0_W-20_Col-0/`, `ablation_destructive_7epoch/`, `weight_sweep_40/`, `theory_switch_validation/`
3. Legacy / staging:
   `paper_refine_v2/`, `paper_aaai_draft/`, `paper_cn/`, `omf_sweep*/`, `next_round_80/`, `path_kinetic/`
4. Archive / cache:
   `archives/`, `eval_cache/`, `eval_results/`

## Main Observations

### 1. Root clarity is much better, but still not finished

The canonical paper, theory, and experiment narrative are now documented. However, several staging directories remain at the root because they still have references in scripts or reports.

### 2. Large evidence directories are not the first cleanup target

Folders like `weight_sweep_40/` and `grid_search_3epoch/` are very large, but they still function as evidence stores. Moving them now would create a large reference-repair task.

### 3. `path_kinetic/` deserves a dedicated cleanup pass

It contains nested duplication (`path_kinetic/path_kinetic/`) and branch-specific remnants. This is now the clearest candidate for a focused structural cleanup.

### 4. `paper_refine_v2/` remains intentionally untouched beyond log cleanup

It still contains active-looking human edits, generated figures, and tracked outputs. Until those are consolidated or explicitly retired, it should be treated as legacy but not aggressively pruned.

## Immediate Outcome

The repository now has enough written structure that later cleanup can be selective instead of exploratory.

## Recommended Next Cleanup Targets

1. `path_kinetic/`
2. root-level historical helper outputs that are already fully represented elsewhere
3. duplication between `paper_refine_v2/` and `paper_orchestra_workspace/`
