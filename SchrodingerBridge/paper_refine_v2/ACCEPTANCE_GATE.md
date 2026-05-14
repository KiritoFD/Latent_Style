# AAAI Acceptance Gate for the Latent Bridge Matching Paper

This document tracks whether the current paper is ready for a serious AAAI-style submission. It is intentionally stricter than a normal project TODO list.

## Current Status

Current draft:

`final/paper.tex`

Current compiled PDF:

`final/paper.pdf`

AAAI-format draft:

`aaai_submission/paper_aaai2026.pdf`

Current simulated review:

`refinement/iter4/review.json`

Verdict:

The paper is now a coherent internal method draft, but it is not yet real-submission-ready.

## Passed Gates

1. The paper has a concrete title, abstract, contribution list, related work, method, experiments, discussion, and conclusion.
2. The method section gives enough information to reproduce the core architecture and objective.
3. The main table reports only complete strict-750 local baselines.
4. The SaMST comparison is no longer hand-wavy: it is supported by qualitative grids, zoom crops, MUSIQ, MANIQA, DISTS, HF-Patch-KID, FFT slope error, and Gram micro statistics.
5. Destructive ablations support the central mechanism: terminal SWD drives style and kinetic regularization preserves content.
6. Weight-sweep evidence is included and avoids cherry-picking only the final epoch.
7. The paper compiles successfully to PDF.
8. Citation, LaTeX structural, and anti-leakage checks pass.

## Blocking Gates Before Submission

1. Complete at least the strongest missing matched baselines if compute permits: CAST and StyTr2 first, then AesFA/AesPA-Net.
2. Add a small user study or clearly move cleanliness/artifact preference to supplement as diagnostic evidence only.
3. Re-run timing in a controlled setting if efficiency is a main-paper claim.
4. Audit `refs.bib` against official paper metadata.
5. Resolve the remaining minor overfull equation in the AAAI-format draft.
6. Ensure author-side policy compliance for any AI-assisted drafting. The current text should be treated as an internal technical draft that authors verify and rewrite before submission if required by the venue.

## Recommended Main-Paper Shape

Target main paper:

1. Introduction with the claim-evidence map compressed or removed.
2. Related work in three short paragraphs.
3. Method with one objective equation, one architecture figure, and one compact algorithm.
4. Main strict-750 table.
5. One qualitative grid and one artifact diagnostic table.
6. One destructive ablation table.
7. Short limitations.

Move to supplement:

1. Full metric taxonomy.
2. Full baseline status table.
3. Full 40-run sweep.
4. Extra theory-switch experiments.
5. Full timing logs.
6. Extra qualitative grids and zoom crops.

## Minimal Next Experimental Upgrade

If only one more baseline can be completed, run CAST strict-750.

If two can be completed, run CAST and StyTr2 strict-750.

If only one human-facing validation can be completed, run a small pairwise cleanliness/artifact preference study:

Ours vs SaMST, 100 pairs, 3 votes per pair, questions: cleaner texture, better content preservation, better overall stylization.
