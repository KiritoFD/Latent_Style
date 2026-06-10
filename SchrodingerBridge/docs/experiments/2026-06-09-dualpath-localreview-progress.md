# DualPath Local Review Progress

Date: 2026-06-09

This note records the current local-review state for:

- `aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_seed42_b8a2`

It is specifically about what is already available locally, and what is still missing before full non-CLIP review can start.

## Current remote eval state

Current remote training state:

- retained checkpoints have already landed through at least `epoch_0012`
- deferred `full_eval` has started and is actively landing per-epoch summaries
- current local pull shows at least:
  - `epoch_0001`
  - `epoch_0002`
  - `epoch_0003`
  - `epoch_0004`
  - `epoch_0005`
  - `epoch_0006`
- remote `full_eval_fresh_localreview` summaries are now visible through at least:
  - `epoch_0012`

Current remote read:

- the family is still in its early eval-side phase
- current points stay in a low-style, lower-LPIPS basin

## Current local handoff

Local handoff root:

- [dualpath_bestfew_localreview_20260609](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_bestfew_localreview_20260609)

Current best-few handoff CSVs:

- train-side:
  - [full_eval_bestfew_handoff.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_bestfew_localreview_20260609/full_eval_bestfew_handoff.csv)
- image-backed:
  - [full_eval_fresh_localreview_bestfew_handoff.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_bestfew_localreview_20260609/full_eval_fresh_localreview_bestfew_handoff.csv)

Current local handoff points:

- best transfer `LPIPS`
  - `epoch_0001 = 0.6927 / 0.4037`
- best transfer `CLIP-style`
  - `epoch_0009 = 0.6926 / 0.4385`

Current image-backed curve read:

- the image-backed family stays in a narrow low-style band:
  - about `0.6916 -> 0.6926`
- while `LPIPS` stays in a much lower range than `QEdgePattn`:
  - about `0.4036 -> 0.4403`

## Current image-backed state

The first image-backed local packet is now genuinely present:

- `full_eval_fresh_localreview/epoch_0001`
- `full_eval_fresh_localreview/epoch_0004`
- `full_eval_fresh_localreview/epoch_0009`

So dualpath is no longer scalar-only.

It is now valid for:

- local `VLM`
- local `IntroStyle`
- local `DINO`

Current evidence caveat:

- the image-backed best-few packet is already local
- but the best-few local `IntroStyle / DINO` outputs are not yet fully materialized as standalone CSVs
- so the current family judgment still leans primarily on:
  - image-backed `VLM`
  - plus the image-backed scalar `CLIP/LPIPS` curve
- detached local `CPU-only` `DINO` backfill is now running to close the structure-axis gap on the best-few packet

The current image-backed local non-CLIP review has started with:

- `DualPathFresh_e01`
- `DualPathFresh_e09`

Current first CPU-only `VLM` summary:

- [vlm_dualpathfresh_e01_vs_knee_vs_seedream_20260609.method_summary.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_dualpathfresh_e01_vs_knee_vs_seedream_20260609.method_summary.csv)
- current compact staging board:
  - [dualpath_vlm_triplets_compare_20260609.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_vlm_triplets_compare_20260609.md)
- current completed cases:
  - `252`
- current vote read:
  - overall wins `15 / 252`
  - style wins `12 / 252`
  - structure wins `23 / 252`
  - artifact wins `31 / 252`
- mean local scores:
  - style `2.96`
  - structure `3.87`
  - artifact `3.25`

Current second image-backed local CPU-only `VLM` summary:

- [vlm_dualpathfresh_e09_vs_knee_vs_seedream_20260609.method_summary.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_dualpathfresh_e09_vs_knee_vs_seedream_20260609.method_summary.csv)
- current completed cases:
  - `130`
- current vote read:
  - overall wins `6 / 130`
  - style wins `5 / 130`
  - structure wins `2 / 130`
  - artifact wins `4 / 130`
- mean local scores:
  - style `2.90`
  - structure `3.64`
  - artifact `2.75`

Interpretation:

- `DualPathFresh_e01` is already clearly above `LBM-Knee`
- and unlike the earlier scalar-only read, it has already produced sparse real non-CLIP wins
- but it is still well below `Seedream`
- first image-backed `DINO` has now also landed for `DualPathFresh_e01`:
  - `745` pairs
  - `DINO = 0.02902`
- this is worse than:
  - `LBM-Knee full750 DINO = 0.02171`
  - and worse than the first landed `QEdgePattn e01 DINO = 0.02635`
- together with the scalar comparison table, the current read is:
  - lower-style but lower-`LPIPS` basin
  - with enough local quality to beat `LBM-Knee` sometimes
  - but not enough target-style recovery to challenge `Seedream`
- `DualPathFresh_e09` now gives the first look at the higher-style image-backed point
- it remains below `Seedream`
- but it shows slightly higher style-win density than `e01` so far
- while giving weaker structure/artifact breadth and using a much smaller batch
- so dualpath is no longer a single-point local phenomenon

Current family-level read:

- `DualPathFresh_e01` remains the more reliable point in this family
  - broader evidence
  - stronger structure/artifact breadth
- `DualPathFresh_e09` remains the more style-leaning point
  - slightly denser style wins
  - but still much weaker structure/artifact coverage
- this split is now stable enough to treat as the current dualpath family shape:
  - `e01` = safer, stronger all-around local quality point
  - `e09` = higher-style probe that still has not proven it can scale

## Current implication

The dualpath family is now in a different evidence state from `QEdgePattn`.

`QEdgePattn` already has:

- image-backed local packet
- CPU-only `VLM`
- non-CLIP family-level read

`DualPath` now has:

- training-side scalar curve
- train-side `full_eval` scalar curve
- image-backed local packet
- first image-backed local non-CLIP review chain for:
  - `DualPathFresh_e01`

Current direct comparison against the earlier family:

- [dualpath_vs_qedgepattn_early_curve_20260609.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/dualpath_vs_qedgepattn_early_curve_20260609.md)

That comparison currently shows:

- lower cheap style than `QEdgePattn`
- but much lower `LPIPS`

So the current dualpath question is still earlier-stage:

- can it escape the low-style conservative basin in scalar eval

The next gate now becomes:

- whether later image-backed `DualPathFresh` points can keep the lower-`LPIPS` basin
- while matching or exceeding `DualPathFresh_e01` on local non-CLIP style read
