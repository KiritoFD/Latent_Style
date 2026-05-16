# SaMST Targeted Metric Recommendation

Visual inspection shows that SaMST is not a simple content-collapse baseline. It preserves low-frequency layout and many source-aligned contours, so standard content metrics can favor it. Its visible weakness is perceptual degradation: muddy/blurry local structure, noisy texture edges, color contamination in flat regions, and high-frequency texture distribution shift.

## Recommended Metrics

| Metric | Direction | Ours epoch_0007 | SaMST strict | Why it is valid here |
|---|---:|---:|---:|---|
| MUSIQ | up | 49.2059 | 36.0950 | No-reference perceptual quality; matches visible dirty/blurry degradation. |
| MANIQA | up | 0.4057 | 0.3139 | NR-IQA, often sensitive to generated-image distortions. |
| DISTS-content | down | 0.2477 | 0.2943 | Better perceptual content/structure distance than CLIP-content alone. |
| HF-Patch-KID | down | 4.169393 | 6.759762 | Checks whether high-frequency patches match real style texture statistics. |
| FFT slope error | down | 0.5473 | 1.0536 | Captures abnormal spectrum shape from grain/dithering artifacts. |
| Denoise chroma delta | down | 0.7039 | 0.9413 | SaMST changes more under mild denoising, indicating unstable color residue. |
| Flat chroma residual | down | 0.003358 | 0.004428 | Measures color residual pollution in source-flat regions. |
| Gram micro style loss | down | 0.079837 | 0.094652 | Shallow VGG Gram texture statistics favor ours. |
| LPIPS-content | down | 0.4514 | 0.4664 | Ours has slightly better perceptual content distance. |
| EC product: CLIP-style x (1-LPIPS) | up | 0.3928 | 0.3839 | Effectiveness/coherence trade-off: ours has a better style-content balance. |

## Metrics Not Suitable For Claiming Ours Beats SaMST

| Metric | Why not |
|---|---|
| CLIP-content | SaMST keeps semantic layout, so this does not capture local blur/artifacts. |
| SSIM-Y | SaMST preserves low-frequency luminance and wins, despite visible muddy texture. |
| Edge-F1 / Edge-IoU | SaMST creates many texture edges; recall boosts F1 even when boundaries are visually dirty. |
| FID/KID/CLIP-FID | Current 750-set distribution metrics favor SaMST; do not use them as our winning claim. |
| Gram macro | Deep Gram currently favors SaMST; use Gram micro only if discussing shallow texture statistics. |

## Suggested Paper Framing

SaMST is a strong low-frequency/semantic baseline, but it shows lower perceptual quality and less realistic local texture statistics. Therefore, the fair claim is not "ours has higher CLIP-content or FID." The stronger and more defensible claim is:

> Ours achieves a better perceptual quality and style-content trade-off, with lower artifact-sensitive distances (MUSIQ/MANIQA, DISTS, HF-Patch-KID, FFT slope error, flat-chroma residual), while SaMST preserves low-frequency layout but suffers from muddy, grain-like texture artifacts.

