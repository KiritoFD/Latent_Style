# Comparison 2026-06-02

This directory consolidates the current cross-model comparison against `idt`
for three datasets:

- `legacy256_overfit50`
- `wikiart512_5style`
- `distinct5_512`

The comparison covers:

- `clip_style` vs `1 - LPIPS` scatter/curve plots
- target-wise aggregate `ArtFID` bar charts
- representative `LANCET` checkpoints grouped by dataset
- a compact comparison report in both Markdown and PDF

## Report

- Markdown: [comparison_report.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/comparison_report.md)
- PDF: [comparison_report.pdf](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/comparison_report.pdf)

## Datasets

Unified dataset roots:

- `G:\GitHub\Latent_Style\Dataset\legacy256_overfit50`
- `G:\GitHub\Latent_Style\Dataset\wikiart512_5style`
- `G:\GitHub\Latent_Style\Dataset\distinct5_512`

Each test split was normalized to `30` images per style, `150` test images per
dataset total.

## Main tables

- Scatter source: [scatter_points.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/scatter_points.csv)
- LANCET representative points: [lancet_representative_points.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/lancet_representative_points.csv)
- LANCET history registry: [lancet_history_registry.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/lancet_history_registry.csv)
- ArtFID comparison points: [artfid_comparison_points.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/artfid_comparison_points.csv)
- Historical selected style metrics (merged): [selected_style_metrics_historical_merged.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/selected_style_metrics_historical_merged.csv)

## Scatter figures

Notes:

- `LANCET` is rendered as an unconnected scatter cloud plus representative diamonds.
- `SaMST` now includes intermediate `wikiart512_5style` points at `e5`, `e10`, and `e15`.

`full`:

- [legacy256_overfit50 full](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/legacy256_overfit50_clip_style_vs_1lpips_full.png)
- [wikiart512_5style full](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/wikiart512_5style_clip_style_vs_1lpips_full.png)
- [distinct5_512 full](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/distinct5_512_clip_style_vs_1lpips_full.png)

`transfer`:

- [legacy256_overfit50 transfer](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/legacy256_overfit50_clip_style_vs_1lpips_transfer.png)
- [wikiart512_5style transfer](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/wikiart512_5style_clip_style_vs_1lpips_transfer.png)
- [distinct5_512 transfer](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/distinct5_512_clip_style_vs_1lpips_transfer.png)

## ArtFID figures

`full`:

- [legacy256_overfit50 ArtFID full](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/legacy256_overfit50_artfid_full.png)
- [wikiart512_5style ArtFID full](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/wikiart512_5style_artfid_full.png)
- [distinct5_512 ArtFID full](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/distinct5_512_artfid_full.png)

`transfer`:

- [legacy256_overfit50 ArtFID transfer](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/legacy256_overfit50_artfid_transfer.png)
- [wikiart512_5style ArtFID transfer](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/wikiart512_5style_artfid_transfer.png)
- [distinct5_512 ArtFID transfer](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/distinct5_512_artfid_transfer.png)

## Provenance and caveats

1. `idt`, `SaMST`, and most `SaMAM`/`LANCET` points use freshly computed
   target-wise aggregate `ArtFID` JSON files.
2. `distinct5_512` uses remote-computed target-wise aggregates pulled back as
   csv/json bundles, not full downloaded image directories.
3. `legacy256_overfit50` `LANCET` ArtFID bars use archive runs under
   `archives/old_experiment_dirs/grid_search_3epoch/S-none_K-1_C-0_W-20_Col-0`
   because the exact modern representative dirs no longer retain generated
   images locally. These are marked as `targetwise_archive_proxy` in
   [artfid_comparison_points.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/artfid_comparison_points.csv).
4. The scatter plots remain anchored to the current strict-summary
   representative points, including the exact modern `legacy256` `K1 original`
   and `steps_12` records.
5. `selected_style_metrics_historical_merged.csv` consolidates the older
   `Ours/SaMST` style-metric run with newly recovered `StyleID/S2WAT/AdaIN`
   Gram, FID, KID, CLIP-FID, CMMD, and ArtFID values from the preserved
   `complete_750` image folders.

## Related note

Dataset-organized `LANCET` history and representative choices are summarized
in [lancet_history_by_dataset.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/lancet_history_by_dataset.md).
