# Hold4Mid `e8` Geometry Anchor Archive

Date: 2026-06-08

Archived point:

- run:
  - `aaai2027_inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamphold4mid_reseed_from_e13_seed42_b8a2`
- selected checkpoint:
  - `epoch_0008.pt`
- transfer:
  - `0.6679105 / 0.2877402`
- all-pairs:
  - `0.7013853 / 0.2877823`

Why this point is archived:

- this is an unusually strong low-LPIPS geometry/content anchor
- even though it is not the main paper-facing frontier, it is scientifically important:
  - it proves the clamp family can lock geometry extremely hard
  - it sets a concrete target for later style-reopening mechanisms

Archive zip:

- [2026-06-08-hold4mid-e8-geometry-anchor.zip](/G:/GitHub/Latent_Style/SchrodingerBridge/archives/root_level_snapshots/2026-06-08-hold4mid-e8-geometry-anchor.zip)

Expected archive contents:

- config JSON
- checkpoint `epoch_0008.pt`
- `summary.json`
- `metrics.csv`
- `clip_lpips_curve.csv`
- training CSV
- this archive note
- run closure note

Use:

- treat this packet as the reproducibility anchor for the current ultra-low-LPIPS regime
- future late-style-reopening packets should be compared against this archive, not only against the paper-facing style frontier
