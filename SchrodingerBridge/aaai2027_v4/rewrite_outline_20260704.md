# AAAI 2027 Rewrite Outline

## 1. One-sentence claim

IDT calibration shows that art-to-art style transfer is easy to overclaim because an unchanged artwork can already score highly on the requested target style; WD-VF fixes the underlying geometric conflict by separating low-frequency structure from high-frequency style motion in the Stable Diffusion v1.5 EMA VAE latent, yielding real target-direction transfer at 903K parameters, 3.08 minutes of training, and 83.7 seconds for 750 stylizations on a single RTX 3060.

## 2. Story spine

1. The evaluation bug:
   Distinct5 is built from the five styles with the lowest IDT CLIP-S, so the benchmark explicitly penalizes fake transfer by no-op behavior.
2. The geometric cause:
   Euclidean rectified flow mixes structure and texture in the same coordinates, so optimization is pulled toward source preservation.
3. The fix:
   WD-VF uses wavelet coordinates, LL de-weighting, high-frequency query routing, and endpoint-only WCT.
4. The payoff:
   The method is both small and fast, while staying above the IDT floor and preserving a strong CLIP-S / LPIPS operating point.
5. The implication:
   Affordable real style transfer is a representation-design problem, not a large-prior or large-cluster problem.

## 3. Hard writing rules

- No rebuttal tone.
- No internal lab slang.
- No unverified claims.
- `97.3 s` is evaluation packet time only; the inference headline is `83.7 s / 750` and `111.7 ms / image`.
- The VAE name is fixed as `Stable Diffusion v1.5 EMA VAE`.
- Do not restate full table rows in prose.
- Do not describe successful extension results as failure.
- Do not promise Hessian-spectrum evidence unless we actually run it.

## 4. Section-by-section outline

### Title

Keep the current title unless the user wants one last OT-facing rename:

`Affordable Real Style Transfer: Training Wavelet-Decoupled Rectified Flow on a Single RTX 3060 in Minutes`

### Abstract

Use four compact moves:

1. Art-to-art transfer is mis-evaluated without an unchanged-image baseline.
2. Euclidean latent flow suppresses style motion because low-frequency structure dominates the objective.
3. WD-VF changes coordinates and training geometry through Haar subbands, LL de-weighting, routed style queries, and endpoint WCT.
4. Final result:
   `903K params`, `3.08 min train`, `83.7 s / 750 infer`, `CLIP-S 0.7213`, `LPIPS 0.2868`, `+0.1397 CLIP-S over SaMam`, `141x faster training`.

### 1. Introduction

Five short paragraphs only.

1. Problem setup:
   art-to-art transfer is vulnerable to no-op success because the source is already art.
2. IDT calibration:
   define the no-op floor and explain why Distinct5 uses the five lowest-IDT styles.
3. Empirical punch:
   SaMam has low LPIPS but sits below the IDT floor, so low artifact alone is not real transfer.
4. Theory + method preview:
   Euclidean latents entangle structure and texture; WD-VF fixes this by moving to wavelet coordinates.
5. Cost/result paragraph:
   `903K`, `3.08 min`, `83.7 s / 750`, positive-IDT CLIP-S / LPIPS tradeoff.

Contribution bullets:

1. IDT-calibrated evaluation for art-to-art transfer.
2. A theorem-led explanation of Euclidean collapse.
3. WD-VF as the geometric fix.
4. A consumer-GPU operating point with audited training and inference cost.

### 2. Related Work

Shrink to three short paragraphs.

1. Exemplar-conditioned stylization:
   classical AdaIN/WCT/StyTr2/AesFA class; explain that our task is stricter because inference sees style identity, not a target reference image.
2. Domain-conditional learned baselines and large priors:
   SaMST, SaMam, CUT, StyleID, SDEdit, Seedream.
3. Wavelets and evaluation:
   wavelets as feature modules in prior work; our use is a coordinate change plus IDT calibration.

What to remove:

- paper-by-paper mini summaries
- long metric philosophy digressions
- any old LBM / FC-SB historical narrative

### 3. Method

Make this section theorem-led rather than implementation-led.

1. Setup:
   latent notation, rectified-flow segment, target velocity, style-domain conditioning.
2. Why Euclidean latents collapse:
   define the style-suppressed manifold, state the sufficient local condition, then explain it in plain language.
3. Wavelet coordinates:
   define Haar split into LL/LH/HL/HH and show why this changes the optimization geometry.
4. Training objective:
   show the implemented loss with `(0.3, 1, 1)` weights and no HH head.
5. High-frequency-only routed limit:
   state the decoupling theorem and explain why this predicts faster optimization.
6. Endpoint alignment:
   justify endpoint WCT and explain why per-step injection over-stylizes.

Method writing target:

- fewer but cleaner equations
- each theorem followed by a one-paragraph intuition
- every formal claim must be matched later by one explicit experiment row

### 4. Architecture and Cost

One compact subsection at the end of Method or at the start of Experiments.

Include only:

- `4 residual blocks`, `width 64`, `3 velocity heads`, `903K params`
- `5 epochs`, `batch 24`, `AdamW 2e-4`
- `42.9 s` latent generation
- `40.8 s` VAE decode
- `83.7 s` stylization total
- `97.3 s` full eval packet only as a secondary timing note

Delete the standalone “timing philosophy” tone. Keep the distinction factual.

### 5. Experiments

#### 5.1 Protocol

State only the essentials:

- Distinct5-WikiArt-512
- five styles with the lowest IDT CLIP-S from the broader WikiArt pool
- 18,000 train latents
- 750 evaluation pairs with 150 identity pairs
- HF CLIP ViT-B/32 and Alex LPIPS

#### 5.2 Main comparison

Main table should be the center of the section and already carry the cost story.

Recommended columns:

`Method | Interface | CLIP-S | LPIPS | Δ_IDT | Params | Train | Infer / 750`

Main-text interpretation should make four points, not re-read the table:

1. IDT changes the ranking.
2. WD-VF is the only trained domain-conditional row that is positive-IDT, sub-0.30 LPIPS, and minute-scale to train.
3. Diffusion and API systems still define the raw CLIP-S ceiling.
4. SaMam’s low LPIPS is not evidence of successful target-direction transfer.

#### 5.3 Page-1 figure

Keep it double-column and high-impact.

- Left: all-pairs CLIP-S vs `1 - LPIPS`, with full SaMam curve and IDT horizontal line.
- Right: target-pooled ArtFID bars with training-time labels inside bars.
- Caption should explain what the reader should conclude, not describe layout only.

#### 5.4 Ablations

Use only defensible, directly traceable rows.

Groups:

1. Wavelet basis:
   No DWT, Haar vs db2.
2. LL handling:
   lock LL, `w_LL` changes, HH head.
3. Endpoint alignment:
   per-step AdaIN, diagonal AdaIN vs WCT, inject LL too.
4. Routing:
   `p = 0.0`, `0.5`, `0.8`, `1.0`.
5. Solver:
   Euler, Heun, RK4.
6. Capacity:
   depth, width, gate.

Writing rule:

- each group gets one sentence of interpretation
- no “we explored many settings” language
- no mislabeled `w_LL` rows

#### 5.5 Controls and extensions

This section should do three jobs:

1. Show the result is not carried by identity pairs alone:
   transfer-only row `0.6908 / 0.2965`.
2. Show latent semantics matter:
   pixel256 `0.6960 / 0.5317` vs latent256 `0.7168 / 0.3125`.
3. Show the method is reusable:
   frozen-backbone / style-memory update `0.7218 / 0.3020`, and 8-style extension `0.7039 / 0.2555`.

Important framing:

- successful parameter-efficient extension belongs here
- old few-shot Pop_Art textual inversion failures do not belong in the main paper narrative

#### 5.6 Auxiliary quality diagnostics

Do not force MUSIQ into the main paper unless we have verified numbers.

Preferred policy:

1. Keep ArtFID in Figure 1 and as one short paragraph in the main text.
2. If verified MUSIQ / MANIQA / NonCLIPAcc numbers are found quickly, add one small auxiliary table.
3. If not verified, move all extra perceptual diagnostics to the supplement and do not mention MUSIQ in the body.

### 6. Discussion

Only three points:

1. Why the method is fast:
   the network stops spending capacity on low-frequency reconstruction conflict.
2. Why Haar is the default:
   exact orthogonality and clearer subband supervision; db2 does not improve the CLIP-S / LPIPS tradeoff enough.
3. What the paper does not claim:
   it is domain-conditional art transfer, not exemplar copying.

### 7. Conclusion

One short paragraph.

Close on:

- IDT as the evaluation correction
- WD-VF as the geometric fix
- the audited consumer-GPU operating point

Do not reopen related work or future work here.

## 5. Figure and table plan

### Keep in main paper

1. Figure 1:
   page-1 summary figure.
2. Figure 2:
   current architecture figure, unchanged in content.
3. Table 1:
   main comparison, double-column.
4. Table 2:
   ablation table.
5. Table 3:
   controls and extensions.

### Move to supplement if page pressure appears

1. standalone timing breakdown table
2. extra perceptual metrics table
3. pixel-vs-latent full matrix figure
4. additional few-shot logs or larger style-set details

## 6. Data sources to use

### Main paper

- Main baseline table:
  `docs/72/03_experiments.md`, `docs/72/07_related_works.md`, and the current `paper.tex`
- Pixel vs latent control:
  `docs/exp/pixel.md`
- Frozen-backbone / style-memory update:
  `docs/exp/local_experiments.md` row `630_phase4j3_fewshot_stylemem`
- Current selected WD-VF packet:
  `exp/FCSB/local_t/630_local_t11_stochastic_dwt_p08/full_eval/epoch_0005/summary.json`

### Use with caution

- old `paper_aaai2027.tex` artifact metrics language:
  can be mined for structure, not for unchecked numbers
- historical timing CSV:
  useful for provenance, but only copy rows that match the current packet and protocol

### Do not use

- any pre-correction SaMam `0.7175` or `0.7222` values
- old SDXL wording
- failed fewshot6 Pop_Art rows as if they were the main extension result

## 7. Specific edits to make in the next pass

1. Rewrite the abstract from scratch.
2. Compress the introduction to five short paragraphs plus contributions.
3. Cut related work to three compact paragraphs.
4. Recast the mathematics by role:
   - keep the Euclidean-collapse result as the only theorem-level explanatory claim;
   - downgrade the routed-objective decomposition to lemma/proposition level;
   - make every formal statement point to one explicit empirical consequence.
5. Rewrite the experiments prose so that it interprets the main table instead of repeating it.
6. Keep IDT framed as a benchmark calibration for this art-to-art setting, not as a universal threshold across datasets.
7. Present the style-memory update and 8-style result as successful parameter-efficient extension evidence.
8. Expand limitations into one serious paragraph: domain-conditional scope, narrow benchmark, palette-heavy styles, and single-seed reporting.
9. Keep MUSIQ out of the body until verified data is in hand.
