# All Tested Metrics Summary

Source folder: `Related_Works/run_511/complete_750`

This file summarizes every strict protocol-750 metric that has actually been run so far. The machine-readable version is `summary_all_tested_metrics.csv`; standalone paper/result copies are also written to `Related_Works/results/metrics_summary/`.

## Coverage

Full strict-750 coverage for all six rows:

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

Strong artifact-diagnostic coverage currently exists for `ours_epoch_0007` and `samst_strict`:

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

## Full-Coverage Table

| Method | Run | LPIPS down | CLIP-style up | CLIP-content up | SSIM-Y up | Edge-F1 up | Blur-drop down | Down-drop down | Extra-edge down | Chroma-Z down | FlatChroma-Z down |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours epoch_0007 | `ours_epoch_0007` | 0.4587 | 0.7041 | 0.8043 | 0.4545 | 0.3110 | 0.0034 | -0.0014 | 0.0764 | -0.4860 | -0.5239 |
| SaMST strict | `samst_strict` | 0.4664 | 0.7194 | 0.8193 | 0.6520 | 0.5162 | 0.0025 | -0.0012 | 0.1077 | 0.2636 | -0.3002 |
| StyleID strict | `styleid_strict` | 0.7497 | 0.7597 | 0.5519 | 0.1466 | 0.1954 | -0.0005 | -0.0120 | 0.2886 | 0.7909 | 1.9891 |
| AdaIN v32k | `adain_v32k` | 0.6298 | 0.7130 | 0.6990 | 0.3246 | 0.0167 | 0.0391 | 0.0185 | 0.0071 | 1.1066 | -1.2710 |
| AdaIN vgg19 | `adain_vgg19` | 0.6870 | 0.6930 | 0.5991 | 0.2897 | 0.0182 | 0.0662 | 0.0361 | 0.0197 | 0.3528 | -1.4401 |
| AdaIN bad | `adain_bad` | 0.8490 | 0.6308 | 0.5297 | 0.2868 | 0.0000 | 0.0070 | 0.0000 | 0.0000 | 410.2368 | -1.8290 |

## Strong Diagnostics Table

| Method | Run | MUSIQ up | MANIQA up | DISTS-content down | DenoiseStyleDrop down | DenoiseChromaDelta down | FFT-KL down | FFT-Slope-Error down | ChromaGrainIndex down | HF-Patch-KID down | plain KID down |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours epoch_0007 | `ours_epoch_0007` | 49.2059 | 0.4057 | 0.2477 | 0.0056 | 0.7039 | 0.0853 | 0.5473 | 0.1391 | 4.169393 | 0.052437 |
| SaMST strict | `samst_strict` | 36.0950 | 0.3139 | 0.2943 | 0.0009 | 0.9413 | 0.2419 | 1.0536 | -0.0063 | 6.759762 | 0.048909 |

## Current Readout

- `Ours` remains the most stable overall row under the combined metric stack.
- `SaMST` is strong on raw style/content/structure metrics, so it should not be described as structurally failed.
- `plain KID` does not penalize `SaMST`; use it as a conventional distribution metric, not as an anti-grain detector.
- `HF-Patch-KID`, `MUSIQ`, `MANIQA`, `DISTS-content`, `FFT-KL`, and `FFT-Slope-Error` expose the perceptual/high-frequency realism weakness visible in SaMST outputs.
- `StyleID` is clearly a semantic-drift/content-preservation failure in this protocol.
- `AdaIN` is mainly a washed-structure baseline; `adain_bad` should be treated as invalid.

## Source Files

- Full summary CSV: `Related_Works/run_511/complete_750/summary_all_tested_metrics.csv`
- Standalone metric folder: `Related_Works/results/metrics_summary/`
- Inventory doc: `Related_Works/docs/REPRO_DATA_INDEX.md`
- Toolchain doc: `Related_Works/run_511/docs/ADVANCED_METRICS_TOOLCHAIN.md`
