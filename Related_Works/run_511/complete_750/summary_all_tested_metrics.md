# All Tested Metrics Summary

Source folder: `run_511/complete_750`

This file merges every metric that has actually been run so far.

## Coverage

### Full `complete_750` coverage

These metrics have been run for all 6 strict-750 runs:

- `LPIPS-content (VGG)`
- `CLIP-style`
- `CLIP-content`
- `SSIM-Y`
- `Edge-F1`
- `Edge-IoU`
- `BlurStyleDrop`
- `DownStyleDrop`
- `ExtraEdgeRate`
- `ChromaSpeckleZ`
- `FlatChromaHF-Z`

Runs covered:

- `ours_epoch_0007`
- `samst_strict`
- `styleid_strict`
- `adain_v32k`
- `adain_vgg19`
- `adain_bad`

### Partial strong-diagnostic coverage

These metrics have been run only for:

- `ours_epoch_0007`
- `samst_strict`

Metrics:

- `MUSIQ`
- `MANIQA`
- `DISTS-content`
- `DenoiseStyleDrop`
- `DenoiseChromaDelta`
- `FFT-Radial-KL-style`
- `FFT-Slope-Error`
- `Chroma-ACL-Z`
- `Chroma-Moran-Z`
- `SmallBlobRatio-Z`
- `StructureTensorCoherence-Z`
- `ChromaGrainIndex`
- `HF-Patch-KID`
- `plain KID`

## A. Full Coverage Table

| Method | Run | LPIPS↓ | CLIP-style↑ | CLIP-content↑ | SSIM-Y↑ | Edge-F1↑ | Blur-drop↓ | Down-drop↓ | Extra-edge↓ | Chroma-Z↓ | FlatChroma-Z↓ | Risk flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Ours epoch_0007 | `ours_epoch_0007` | 0.4587 | 0.7041 | 0.8043 | 0.4545 | 0.3110 | 0.0034 | -0.0014 | 0.0764 | -0.4860 | -0.5239 |  |
| SaMST strict | `samst_strict` | 0.4664 | 0.7194 | 0.8193 | 0.6520 | 0.5162 | 0.0025 | -0.0012 | 0.1077 | 0.2636 | -0.3002 | `grainy_but_structured` |
| StyleID strict | `styleid_strict` | 0.7497 | 0.7597 | 0.5519 | 0.1466 | 0.1954 | -0.0005 | -0.0120 | 0.2886 | 0.7909 | 1.9891 | `weak_content,semantic_drift,noisy_vs_style,chroma_speckle` |
| AdaIN v32k | `adain_v32k` | 0.6298 | 0.7130 | 0.6990 | 0.3246 | 0.0167 | 0.0391 | 0.0185 | 0.0071 | 1.1066 | -1.2710 | `washed_structure,blocky,style_not_blur_robust,style_not_scale_robust,chroma_speckle` |
| AdaIN vgg19 | `adain_vgg19` | 0.6870 | 0.6930 | 0.5991 | 0.2897 | 0.0182 | 0.0662 | 0.0361 | 0.0197 | 0.3528 | -1.4401 | `semantic_drift,washed_structure,blocky,style_not_blur_robust,style_not_scale_robust` |
| AdaIN bad | `adain_bad` | 0.8490 | 0.6308 | 0.5297 | 0.2868 | 0.0000 | 0.0070 | 0.0000 | 0.0000 | 410.2368 | -1.8290 | `weak_content,semantic_drift,washed_structure,chroma_speckle` |

## B. Strong Diagnostics Table

Current coverage: `ours_epoch_0007`, `samst_strict` only.

| Method | Run | MUSIQ↑ | MANIQA↑ | DISTS-content↓ | DenoiseStyleDrop↓ | DenoiseChromaDelta↓ | FFT-KL↓ | FFT-Slope-Error↓ | ACL-Z | Moran-Z | Blob-Z | Coherence-Z | ChromaGrainIndex↓ | HF-Patch-KID↓ | plain KID↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours epoch_0007 | `ours_epoch_0007` | 49.2059 | 0.4057 | 0.2477 | 0.0056 | 0.7039 | 0.0853 | 0.5473 | -0.4245 | 0.0905 | 0.1160 | 0.5311 | 0.1391 | 4.169393 | 0.052437 |
| SaMST strict | `samst_strict` | 36.0950 | 0.3139 | 0.2943 | 0.0009 | 0.9413 | 0.2419 | 1.0536 | -0.0915 | 0.6034 | -0.3883 | 0.7819 | -0.0063 | 6.759762 | 0.048909 |

## C. Current Readout

- `Ours` is still the most stable overall run.
- `SaMST` remains strong on raw style/content/structure metrics.
- `plain KID` does **not** penalize `SaMST`; it is slightly better there.
- `HF-Patch-KID`, `MUSIQ`, `MANIQA`, `DISTS-content`, `FFT-KL`, and `FFT-Slope-Error` all move in the expected direction and start exposing SaMST's perceptual / high-frequency realism weakness.
- `StyleID` is clearly `semantic_drift`.
- `AdaIN` is mainly `washed_structure`, with the weaker run also effectively failed.

## D. Source Files

- Full-coverage summary:
  [summary_complete_750.md](</g:/GitHub/Latent_Style/run_511/complete_750/summary_complete_750.md>)
- Related-works-only summary:
  [summary_related_works_750.md](</g:/GitHub/Latent_Style/run_511/complete_750/summary_related_works_750.md>)
- Ours strong diagnostics:
  [eval_artifact_pack750.json](</g:/GitHub/Latent_Style/run_511/complete_750/ours_epoch_0007/eval_artifact_pack750.json>)
  [eval_hf_patch_kid750.json](</g:/GitHub/Latent_Style/run_511/complete_750/ours_epoch_0007/eval_hf_patch_kid750.json>)
  [eval_plain_kid750.json](</g:/GitHub/Latent_Style/run_511/complete_750/ours_epoch_0007/eval_plain_kid750.json>)
- SaMST strong diagnostics:
  [eval_artifact_pack750.json](</g:/GitHub/Latent_Style/run_511/complete_750/samst_strict/eval_artifact_pack750.json>)
  [eval_hf_patch_kid750.json](</g:/GitHub/Latent_Style/run_511/complete_750/samst_strict/eval_hf_patch_kid750.json>)
  [eval_plain_kid750.json](</g:/GitHub/Latent_Style/run_511/complete_750/samst_strict/eval_plain_kid750.json>)
