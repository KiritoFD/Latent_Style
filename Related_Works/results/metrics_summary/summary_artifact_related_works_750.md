# Related Works 750 Artifact-Pack Summary

Source folder: `run_511/complete_750`

| Method | Run | MUSIQ up | MANIQA up | DISTS-content down | DenoiseStyleDrop down | FFT-KL down | ACL-Z | Moran-Z | Blob-Z | GrainIndex down | Risk flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SaMST strict | `samst_strict` | 36.095 | 0.3139 | 0.2943 | 0.0009 | 0.2419 | -0.0915 | 0.6034 | -0.3883 | -0.0063 | low_nr_quality,low_maniqa,content_distance_high,hf_distribution_shift |

## Notes

- `MUSIQ` and `MANIQA` are no-reference quality metrics; higher is better.
- `DISTS-content` is computed against the source content image; lower is better.
- `DenoiseStyleDrop` measures how much CLIP-style falls after mild bilateral denoising.
- `GrainIndex` combines short chroma autocorrelation, weak chroma spatial coherence, and excess small chroma blobs.
