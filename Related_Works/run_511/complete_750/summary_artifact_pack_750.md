# Complete 750 Artifact-Pack Summary

Source folder: `run_511/complete_750`

| Method | Run | MUSIQ up | MANIQA up | DISTS-content down | DenoiseStyleDrop down | FFT-KL down | ACL-Z | Moran-Z | Blob-Z | GrainIndex down | Risk flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Ours epoch_0007 | `ours_epoch_0007` | 49.2059 | 0.4057 | 0.2477 | 0.0056 | 0.0853 | -0.4245 | 0.0905 | 0.116 | 0.1391 | content_distance_high |
| SaMST strict | `samst_strict` | 36.095 | 0.3139 | 0.2943 | 0.0009 | 0.2419 | -0.0915 | 0.6034 | -0.3883 | -0.0063 | low_nr_quality,low_maniqa,content_distance_high,hf_distribution_shift |
| s2wat_strict | `s2wat_strict` | 36.5256 | 0.1754 | 0.2942 | 0.0192 | 0.1224 | -0.6068 | -0.2747 | 0.3967 | 0.2606 | low_nr_quality,low_maniqa,content_distance_high,denoise_fragile_style |

## Notes

- `MUSIQ` and `MANIQA` are no-reference quality metrics; higher is better.
- `DISTS-content` is computed against the source content image; lower is better.
- `DenoiseStyleDrop` measures how much CLIP-style falls after mild bilateral denoising.
- `GrainIndex` combines short chroma autocorrelation, weak chroma spatial coherence, and excess small chroma blobs.
