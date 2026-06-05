# SaMST Latent Distinct5-512 Same-Cost Preflight

Date: 2026-06-06

Scope:

- method: latent `SaMST`
- dataset: `Distinct5-512`
- lane: `same-cost`
- machine: remote `RTX 3060 WSL`
- hard runtime cap: `< 11.0 GiB`

## Summary

This note records the current preflight state for latent `SaMST` on
`Distinct5-512`.

Unlike latent `SaMAM`, the current blocker is no longer a large-margin OOM.
After repairing the wrapper contract, the lane now reaches real training and
misses the `11.0 GiB` cap by only about `5 MiB` at the `30s` health gate.

## Preflight progression

### Attempt 1

Run root:

- `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samst_latent_distinct5_512_samecost_20260606_034941`

Observed result:

- the wrapper copied style exemplar inputs from the wrong surface
- `train_latent.py` expected latent `*.pt` style exemplars
- the run failed immediately with:
  - `IndexError: list index out of range`

### Attempt 2

Run root:

- `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samst_latent_distinct5_512_samecost_20260606_035136`

Observed result:

- the wrapper switched to manifest-aware latent exemplar discovery
- however, it still pointed `style_latent_root` at the packed training root
- the run again failed before training because the script could not find
  per-style exemplar files under that root

### Attempt 3

Run root:

- `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samst_latent_distinct5_512_samecost_20260606_035540`

Observed result:

- the wrapper now materializes one style exemplar `.pt` per style from the
  packed latent cache and points both `style_image` and `style_latent_root`
  consistently at that materialized exemplar root
- the lane starts real training
- the `30s` health gate observes about `11005 MiB`
- this is above the hard cap, so the lane was stopped immediately

## Current read

Current read:

- latent `SaMST` same-cost is no longer blocked on packaging mismatch
- it is now blocked on a **very small runtime memory overage**
- the next step should be a narrow low-VRAM adjustment, not a broad wrapper
  rewrite

Suggested next search space:

- reduce a small constant-memory component rather than change the whole method
- keep the same `Distinct5` protocol and same-cost selection rule unchanged
- do not yet write a paper-facing row, because no retained checkpoint or metric
  closure exists
