# Conference Guidelines: AAAI-Style Draft

## Target venue

AAAI Conference on Artificial Intelligence.

## Submission deadline

Assume a planning cutoff date of May 13, 2026 for the current internal draft. Literature review should prioritize papers available before this date. If a final AAAI call-for-papers deadline is later specified, update this file and rerun literature review.

## Page limit

The main paper should target 7 pages of technical content plus references, consistent with a compact AAAI-style submission. Appendices may be prepared separately for additional visuals, user study materials, implementation details, and extended baseline tables.

## Mandatory sections

The paper must contain, in order:

1. Abstract
2. Introduction
3. Related Work
4. Method
5. Experiments
6. Discussion and Limitations
7. Conclusion
8. References

## Formatting rules

- Use an anonymized double-blind style.
- Do not include author names, affiliations, acknowledgements, or self-identifying repository URLs in the main draft.
- Use LaTeX.
- Use numbered citations through BibTeX.
- Use compact tables with `booktabs`.
- Keep the main paper concise; place long implementation and additional result tables into appendix files if necessary.
- All claims must be grounded in `experimental_log.md`.
- Do not claim state-of-the-art superiority unless supported by matched-protocol data.

## Paper positioning

The paper should be positioned as fast latent-space multi-style artistic transfer, not generic arbitrary reference-image style transfer and not unpaired CycleGAN-style domain translation.

The main claim should be conservative:

```text
The method achieves a competitive style-content trade-off and improved artifact-sensitive perceptual quality compared with strong multi-style baselines under a strict local 750-output protocol.
```

The paper should avoid claiming:

```text
The method universally outperforms SaMST, StyleID, CAST, AesFA, or all diffusion-based style-transfer methods.
```

## Main paper tables and figures

Preferred main paper organization:

1. Quality comparison table on the strict-750 protocol.
2. Efficiency and scalability table.
3. Ablation table showing terminal SWD, kinetic, and color loss effects.
4. Style-content trade-off figure or scatter plot.
5. Qualitative comparison grid.
6. Optional user-study table if human preference data is later collected.

## Evaluation requirements

The experiments section should report:

- CLIP-style.
- CLIP-content.
- LPIPS-content.
- EC score.
- KID/FID or distributional style metrics where available.
- Artifact-sensitive diagnostics such as MUSIQ, MANIQA, DISTS-content, HF-Patch-KID, FFT slope error, and Gram micro for the SaMST comparison.
- Training time and inference time when available.

## Reproducibility requirements

The method section should specify:

- Data domains and strict-750 evaluation protocol.
- Latent input shape.
- Model architecture.
- Loss components and weights.
- Training epochs and optimizer settings.
- Evaluation scripts and paths.
- Which baselines are complete strict-750 comparisons and which were only smoke-tested.

