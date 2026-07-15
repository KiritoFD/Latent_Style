# AAAI 2027 adversarial review round 6

Date: 2026-06-04

Scope:
- Main draft: `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
- Current PDF: `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`
- Figure script touched: `SchrodingerBridge/aaai_submission/scripts_gen_framework_claimsafe.py`
- Reviewers: Godel/Feynman, Kant, Wegener, Cicero

## Reviewer scores

- Godel/Feynman, area-chair and writing lens: 7.3/10.
- Kant, experiments and statistics lens: 7.0/10.
- Wegener, method and math lens: 7.0/10.
- Cicero, figures and layout lens: 7.3/10.

## Shared diagnosis

- The paper is strongest as an evaluation-contract paper: IDT is the main contribution, and LBM is the compact proof point.
- Claims must stay scoped to CLIP-S-based art-to-art Style-ID evaluation until additional fixed-rule splits and clustered bootstrap packets land.
- SaMAM should remain current point-estimate evidence, not a closed reproduced-baseline conclusion.
- Row-resampled intervals should read as exploratory sign checks, not confidence intervals.
- LBM method language is now mostly safe, but endpoint queues and semantic projection needed explicit implementation boundaries.
- Figure 2 had one technical mismatch: the kinetic budget should attach to endpoint training, not inference Euler steps.

## Applied in this round

- Abstract and conclusion now state the operational standard for CLIP-S-based art-to-art evaluation, not all stylization metrics.
- Introduction now names Distinct5 earlier as a fixed-rule CLIP-separated WikiArt stress split chosen before model-output inspection.
- LBM is framed more explicitly as a proof point; representation language now says variants motivate executable control rather than proving a causal tokenizer theorem.
- Related work now says current SaMAM estimates can show the failure mode.
- Method overview now states that LBM does not estimate a stochastic bridge or supervised probability path.
- Endpoint queue construction is specified:
  - VAE-latent feature construction from channel, low/high-pass, gradient, and high-band FFT statistics;
  - target clustering into eight prototypes;
  - nearest target-prototype routing and feature-space L2 ranking;
  - cross-style top-8 queues;
  - F rank-biased top-2 to top-8 curriculum;
  - H/K fixed top-2 plus 0.15 exploration into top-8.
- Terminal semantic projection now states that TopM and target-derived axes are asymmetric projection heuristics, with gradients through generated endpoint projections.
- Table 1 now uses `row check`, `ret-ckpt min`, and SaMAM `(point)` row labels.
- ArtFID is described as a combined art-domain/content-preservation diagnostic.
- SaMST e5/e15 is described as a limited CLIP-S/LPIPS stability check; ArtFID convergence remains unclosed.
- Figure 2 was regenerated with a shorter title and corrected endpoint-velocity-budget arrow.
- Reproducibility text was moved before references while avoiding forbidden AAAI packages.

## Verification

- Build command: `cmd /c build_paper.bat`
- Output: `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`
- Page count: 11.
- Final log scan: no unresolved citations/references, no fatal errors, no overfull boxes. Remaining warnings are XeLaTeX font-substitution warnings.
- Rendered pages inspected: page 1, page 4, page 6, page 10.

## Remaining risks

- Figure 3 still makes LBM look conservative/pale; better visual examples would improve perceived method strength.
- Distinct5 remains one CLIP-separated stress split; Dalton/Faraday should complete additional fixed-rule splits before making stronger benchmark claims.
- SaMAM still needs a completed aligned packet before being treated as a closed baseline.
- Clustered source/style bootstrap remains needed before row sign intervals can become formal uncertainty.
- Fixed-tokenizer/fixed-executor swaps remain the clean causal test for the representation claim.

## Experiment backlog for Dalton/Faraday

- Dalton: finish SaMAM final/converged aligned packet with full/transfer CLIP-S, LPIPS, targetwise ArtFID, per-image rows, timing, and same IDT slots.
- Dalton: compute clustered bootstrap by source image and source-target direction for IDT deltas and direct method comparisons.
- Dalton/Faraday: complete additional fixed-rule WikiArt stress splits under the same 30 x 5 x 5 IDT protocol.
- Dalton: finish SaMST e5/e10/e15 aligned packet with targetwise ArtFID.
- Faraday: run fixed-tokenizer/fixed-executor swaps.
- Faraday: run semantic-axis vs random-axis terminal matching with ArtFID.
