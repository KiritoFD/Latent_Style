# Comparison Report 2026-06-02

This note compares `LANCET`, `SaMAM`, `SaMST`, and `idt` on three datasets using:

- `clip_style` vs `1 - LPIPS`
- target-wise aggregate `ArtFID`

`LANCET` is plotted as an unconnected scatter cloud. `idt` is a diagnostic baseline, not a transfer model.

## Timing notes

- SaMAM 512: 0->5k about 2h06m; 0->10k about 4h25m. Style peak appears at 5k; LPIPS keeps improving through 10k.
- SaMAM 256: 0->15k about 2h24m30s; 17k->25k about 1h31m52s. Later training mostly buys LPIPS.
- SaMST strict historical reference: 750-image inference about 39.8s. Current wikiart512 e5/e10 generation took about 194s for 750 images, with reuse eval about 23-28s and target-wise ArtFID aggregation about 149-154s.
- LANCET timing is heterogeneous across archived and remote runs, so this report does not place a single normalized training-time number beside every LANCET point.

## Legacy256 / overfit50

- Best non-idt CLIP style comes from a LANCET historical point cloud run (epoch_0020).
- Best non-idt 1-LPIPS and best ArtFID both come from different baselines, not the same model.
- This dataset is saturated by metric hacking pressure; idt remains a strong content extreme.
- `full`: Best style: LANCET / epoch_0020 | clip_style=0.7369, 1-LPIPS=0.1549
- `full`: Best content: LANCET / o20a | clip_style=0.6673, 1-LPIPS=0.7038
- `full`: Best ArtFID: SaMAM / SaMAM best-content (25k) | aggregate ArtFID=248.71
- `transfer`: Best style: LANCET / epoch_0020 | clip_style=0.7363, 1-LPIPS=0.1472
- `transfer`: Best content: LANCET / o20a | clip_style=0.6299, 1-LPIPS=0.7027
- `transfer`: Best ArtFID: SaMAM / SaMAM best-content (25k) | aggregate ArtFID=277.27

![Legacy256 / overfit50 full scatter](G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/legacy256_overfit50_clip_style_vs_1lpips_full.png)

![Legacy256 / overfit50 transfer scatter](G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/legacy256_overfit50_clip_style_vs_1lpips_transfer.png)

![Legacy256 / overfit50 full ArtFID](G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/legacy256_overfit50_artfid_full.png)

![Legacy256 / overfit50 transfer ArtFID](G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/legacy256_overfit50_artfid_transfer.png)

## WikiArt512 / 3600 per style

- LANCET owns the highest CLIP style among evaluated non-idt points.
- SaMAM 10k owns the best non-idt 1-LPIPS and best aggregate ArtFID.
- SaMST e5/e10/e15 moves only slightly; the curve is nearly flat on both axes.
- `full`: Best style: LANCET / local WSL hist b32 e8 | clip_style=0.7923, 1-LPIPS=0.6450
- `full`: Best content: SaMAM / SaMAM 10000 | clip_style=0.7851, 1-LPIPS=0.8357
- `full`: Best ArtFID: SaMAM / SaMAM best-content (10k) | aggregate ArtFID=254.39
- `transfer`: Best style: LANCET / local WSL hist b32 e8 | clip_style=0.7853, 1-LPIPS=0.6443
- `transfer`: Best content: SaMAM / SaMAM 10000 | clip_style=0.7774, 1-LPIPS=0.8356
- `transfer`: Best ArtFID: SaMAM / SaMAM best-content (10k) | aggregate ArtFID=328.58

![WikiArt512 / 3600 per style full scatter](G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/wikiart512_5style_clip_style_vs_1lpips_full.png)

![WikiArt512 / 3600 per style transfer scatter](G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/wikiart512_5style_clip_style_vs_1lpips_transfer.png)

![WikiArt512 / 3600 per style full ArtFID](G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/wikiart512_5style_artfid_full.png)

![WikiArt512 / 3600 per style transfer ArtFID](G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/wikiart512_5style_artfid_transfer.png)

## Distinct5-512 / 1000 per style

- SaMST e15 is the best style point in CLIP style.
- LANCET wins the best non-idt 1-LPIPS and the best aggregate ArtFID.
- This is the one dataset where LANCET clearly beats SaMST on content-side metrics.
- The `idt` floor is explicit here: full-scope `clip_style=0.6801`, transfer-only `clip_style=0.6399`.
- No-op-adjusted style gain separates the methods cleanly:
  - `LANCET F e1`: `+0.0168` full, `+0.0244` transfer-only
  - `LANCET K e1`: `+0.0209` full, `+0.0312` transfer-only
  - `SaMST e15`: `+0.0446` full, `+0.0558` transfer-only, but with severe LPIPS / ArtFID collapse
  - `SaMAM 2250`: `-0.0990` full, `-0.0877` transfer-only
- `full`: Best style: SaMST / SaMST e15 | clip_style=0.7247, 1-LPIPS=0.3745
- `full`: Best content: LANCET / F e1 | clip_style=0.6969, 1-LPIPS=0.6814
- `full`: Best ArtFID: LANCET / LANCET best-lpips (F e1) | aggregate ArtFID=122.63
- `transfer`: Best style: SaMST / SaMST e15 | clip_style=0.6957, 1-LPIPS=0.3681
- `transfer`: Best content: LANCET / F e1 | clip_style=0.6644, 1-LPIPS=0.6755
- `transfer`: Best ArtFID: LANCET / LANCET best-lpips (F e1) | aggregate ArtFID=126.83

![Distinct5-512 / 1000 per style full scatter](G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/distinct5_512_clip_style_vs_1lpips_full.png)

![Distinct5-512 / 1000 per style transfer scatter](G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/distinct5_512_clip_style_vs_1lpips_transfer.png)

![Distinct5-512 / 1000 per style full ArtFID](G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/distinct5_512_artfid_full.png)

![Distinct5-512 / 1000 per style transfer ArtFID](G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/comparison_20260602/figures/distinct5_512_artfid_transfer.png)
