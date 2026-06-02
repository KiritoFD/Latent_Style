# Config Index

Updated: 2026-06-03

This directory keeps only the configurations that still matter for current
training, paper-facing reruns, or reusable smoke checks.

## Keep in the main directory

### Current Distinct5-512 paper-facing family

- `distinct5_512_ema_baseline_direct_atom_residual.json`
- `distinct5_512_ema_variant_*.json`

### Current tokenizer exploration family

- `tokenizer_t01_*.json`

### Reusable base / calibration configs

- `wikiart512_ema_direct_atom_residual_calib.json`
- `wikiart512_ema_direct_atom_residual_calib_b80.json`

### General smoke / local sanity

- `exp_sanity.json`

## Archived out of the main view

These remain versioned, but they are no longer first-line experiment entrypoints.

### Local WSL WikiArt512 probes

- `archive/20260603_local_wsl_wikiart512/`

This bundle contains local timing, resume, and short continuation configs used
for 2026-06-01 and 2026-06-02 diagnostics.

### Legacy refactor baseline configs

- `archive/20260603_refactor_legacy/`

This bundle contains older baseline-preservation refactor runs that are still
useful for provenance, but no longer belong in the main config surface.

## Rule of thumb

If a config is only for:

- one-off timing,
- local WSL calibration,
- abandoned refactor scaffolding,
- or a failed direction that is already summarized in docs,

prefer archiving it under `configs/archive/` and linking to it from the
relevant note instead of leaving it in the top-level config list.
