# Distinct5 Seedream 4.5 Repaired 750

Date: 2026-06-07

## Scope

- Dataset: Distinct5 `512x512`, `30` sources per style, `150` sources total.
- Model: `doubao-seedream-4-5-251128`
- Provider: `https://windhub.cc`
- Final evaluated package: [distinct5_512_seedream45_windhub_20260607_repaired750](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/seedream45_api/distinct5_512_seedream45_windhub_20260607_repaired750)

## Why repaired

The first full pass produced `720/750` images. The remaining `30` jobs all failed with provider moderation on six source images, each blocking all five target styles. The blocked-source replacement map is stored in [2026-06-07-distinct5-seedream45-replacements.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-07-distinct5-seedream45-replacements.json).

## Final metrics

- `all_pairs`
  - `clip_style = 0.7198476771513621`
  - `clip_content = 0.8165366362800001`
  - `content_lpips = 0.47671699915999993`
  - `art_fid = 350.9185254316592`
- `style_transfer_ability`
  - `clip_style = 0.6920063383380572`
  - `clip_content = 0.8035140595`
  - `content_lpips = 0.4922635666833334`
  - `art_fid = 386.12684478576335`
- `identity_reconstruction`
  - `clip_style = 0.8312130324045818`
  - `clip_content = 0.8686269434`
  - `content_lpips = 0.41453072906666677`
  - `art_fid = 210.08524801524194`

## Artifacts

- Summary: [summary.json](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/seedream45_api/distinct5_512_seedream45_windhub_20260607_repaired750/summary.json)
- Targetwise ArtFID: [aggregate_targetwise_artfid.json](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/seedream45_api/distinct5_512_seedream45_windhub_20260607_repaired750/aggregate_targetwise_artfid.json)
- Visual grid: [summary_grid.png](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/seedream45_api/distinct5_512_seedream45_windhub_20260607_repaired750/summary_grid.png)
- Assembly manifest: [assembly_manifest.json](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/seedream45_api/distinct5_512_seedream45_windhub_20260607_repaired750/assembly_manifest.json)

## Notes

- The repaired `750` package was assembled from the successful `720`-image base run plus a `30`-image repair run using replacement sources from the same styles.
- Relative to the earlier `720/750` subset, the repaired `750` metrics changed only slightly, so the replacement set did not materially alter the conclusion.
