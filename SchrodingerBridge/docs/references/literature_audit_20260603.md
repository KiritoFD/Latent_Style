# Literature Audit for Intro / Related Work

Date: 2026-06-03

Scope of this pass:

- `aaai_submission/paper_aaai2026.tex`
- `aaai_submission/refs.bib`

Primary-source confirmation used:

- CVPR / ICCV / WACV / ACCV / AAAI / IEEE main pages and OpenAccess entries
- arXiv only where a primary venue page was not the practical citation target

## What was added

The following recent works were added to tighten 2024-2025 coverage and to
separate different style-transfer regimes more clearly:

1. `zheng2024puffnet`
   - `Puff-Net: Efficient Style Transfer with Pure Content and Style Feature Fusion Network`
   - role: efficient reference-guided arbitrary style transfer

2. `deng2024zstar`
   - `Z*: Zero-shot Style Transfer via Attention Reweighting`
   - role: diffusion-based / training-free style transfer at inference time

3. `shang2025scsa`
   - `SCSA: A Plug-and-Play Semantic Continuous-Sparse Attention for Arbitrary Semantic Style Transfer`
   - role: semantic-region-aware reference-guided style injection

4. `jiang2025sms`
   - `Balanced Image Stylization with Style Matching Score`
   - role: recent diffusion-side style-distribution matching / style-content balancing

## Positioning changes made

### 1. Separated reference-guided arbitrary transfer from domain-level style-id transfer

The old narrative mixed together arbitrary reference-image transfer, compact
multi-style transfer, and state-space backbones. The revised text now makes the
deployment assumption explicit:

- reference-guided arbitrary methods consume a style image at inference;
- compact multi-style methods amortize a fixed style family;
- our main protocol is in the second regime.

This matters because several recent strong methods improve exemplar injection,
but do not answer the same question as LBM.

### 2. Fixed misleading state-space grouping

`HSI` was previously grouped too close to the state-space / Mamba line. That is
now corrected:

- `Mamba-ST` and `SaMam` are the state-space line;
- `HSI` and `SCSA` are treated as reference-guided injection / semantic
  correspondence refinements.

### 3. Strengthened the training-free / diffusion lane

The paper now cites both `StyleID` and `Z*`, and adds `SMS` as a recent
diffusion-side style matching paper. This makes the diffusion positioning less
fragile than citing only one or two methods.

### 4. Tightened the metric narrative

The intro / related-work text now stays within the safe theory boundary:

- safe: endpoint-side OT + SA-SWD / `W1`-style matching is the supported story;
- safe: evaluation calibration and unchanged-image prior are real issues;
- not claimed: a closed broad theorem that all latent local `MSE/Huber/L1`
  choices are already settled by current Distinct5 evidence.

## What remains intentionally uncovered

1. **2026 papers**
   - No additional 2026 style-transfer paper was added in this pass.
   - Reason: by 2026-06, many 2026 items are either not yet stable in venue
     form, not clearly central to the current narrative, or would require
     over-expanding related work for marginal gain.

2. **Broad latent-metric correction literature**
   - We did not add citations that would make the paper sound like it has
     already closed a general latent-geometry theorem for local flow losses.
   - Current writing should remain narrower until a matched direct ablation
     exists.

3. **Human-evaluation / VLM-evaluation literature**
   - The current paper already has enough to motivate metric calibration
     caution.
   - A fuller human-study / VLM-judge survey could be added later if the
     evaluation section grows, but it is not necessary for the current intro.

## Current safe summary

After this pass, the intro / related-work story is safest when read as:

1. reference-guided arbitrary stylization has improved injection and semantic
   correspondence;
2. compact multi-style and state-space methods improved deployment efficiency;
3. training-free and diffusion methods improved raw style flexibility;
4. evaluation remains protocol-sensitive and metric-sensitive;
5. LBM is different because it targets domain-level style-id transfer through
   latent transport, endpoint-side OT + SA-SWD matching, and explicit
   evaluation calibration rather than exemplar-conditioned stylization.
