# Project Overview: SchrodingerBridge

## Canonical Entry Points

| Purpose | Path | Note |
|---|---|---|
| Current AAAI manuscript PDF | `paper_orchestra_workspace/aaai_submission/paper_aaai2026.pdf` | Canonical submission snapshot |
| Current AAAI manuscript source | `paper_orchestra_workspace/aaai_submission/paper_aaai2026.tex` | Canonical manuscript source |
| Mathematical model | `maths/MODEL.md` | Active |
| Theoretical analysis | `maths/ANALYSIS.md` | Active |
| Failure reflections | `maths/REFLECTIONS.md` | Active |
| Experiment history | `EXPERIMENT_LOG.md` | Empirical source of truth |
| Cleanup record | `DIRECTORY_CLEANUP_LOG.md` | Directory hygiene decisions |
| Main code | `src/` | Active implementation |

## What This Folder Contains

`SchrodingerBridge/` currently mixes three classes of material:

1. Active assets:
   `src/`, `maths/`, `paper_orchestra_workspace/`, `EXPERIMENT_LOG.md`
2. Evidence worth preserving:
   `ablation_destructive_7epoch/`, `S-add__K-1_C-0_W-20_Col-0/`, `weight_sweep_40/`, `theory_switch_validation/`
3. Historical or staging workspaces:
   `paper_refine_v2/`, `paper_aaai_draft/`, `kinetic_sweep/`, `omf_sweep*/`, older sweeps and root-level packaged snapshots

The cleanup goal is not to erase history. It is to make the active path obvious.

## Current Canonical Status

- Canonical paper workspace:
  `paper_orchestra_workspace/aaai_submission/`
- Canonical theory workspace:
  `maths/`
- Canonical experiment narrative:
  `EXPERIMENT_LOG.md`
- Stable historical baseline run:
  `S-add__K-1_C-0_W-20_Col-0/`
- Trusted ablation control:
  `ablation_destructive_7epoch/configs/D0_full_correct_7ep.json`

## Important Warnings

- Root `config.json` should not be treated as the trusted OMF baseline config.
- `paper_refine_v2/` is a legacy refinement workspace, not the current paper root.
- `omf_sweep/`, `omf_sweep2/`, and `omf_sweep3/` are staging/proposal folders unless their outcomes are written into `EXPERIMENT_LOG.md`.

## Directory Roles

### `paper_orchestra_workspace/`

Primary paper workspace. This is the place to update when the manuscript changes.

### `paper_refine_v2/`

Legacy refinement and figure-generation workspace. Keep it for traceability, but do not cite it as the active paper location.

### `maths/`

Primary location for derivations, corrections, theoretical objections, and future mathematical updates.

### `ablation_destructive_7epoch/`

Key experimental evidence for what components are necessary and what ablations fail.

### `S-add__K-1_C-0_W-20_Col-0/`

Important historical baseline lineage and user-specified restart point.

## Cleanup Policy

- Document first.
- Prefer archive/move over deletion when uncertainty exists.
- Keep evidence directories intact.
- Only remove files that are clearly duplicated, packaged, cached, or build byproducts.

## Last Review

- Reviewed on 2026-05-16 during directory cleanup pass 1.
