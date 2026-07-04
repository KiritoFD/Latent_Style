## AAAI 2027 Writing Cleanup Outline

### Goal

Turn the current draft into a submission-facing manuscript that reads like a standard AAAI paper:

- short sentences
- standard terminology
- no rebuttal tone
- no unsupported claims
- theory used to explain and predict, not to overclaim
- experiments interpreted rather than repeated

### One-sentence claim

IDT calibration shows when art-to-art style transfer is only reconstructing the source, and WD-VF fixes the underlying low-frequency conflict by moving rectified flow into wavelet coordinates, yielding real target-direction transfer at low cost.

### Core narrative

1. Art-to-art transfer is easy to mis-evaluate because the input is already an artwork.
2. IDT supplies the missing no-op calibration.
3. Under that calibration, some low-LPIPS systems are not doing real target-direction transfer.
4. The cause is geometric: Euclidean latent flow mixes layout and texture in the same coordinates.
5. WD-VF changes the coordinates and weakens the low-frequency conflict.
6. The result is a small local model that stays above the IDT floor with a strong CLIP-S/LPIPS tradeoff.

### Writing rules for the next pass

- Keep every paragraph on one claim.
- Prefer one clause over two.
- Do not restate raw table entries in prose unless the number itself is the point.
- Do not present training time as the scientific contribution by itself.
- Do not promise evidence that is not in the paper.
- Do not use defensive phrases such as "this theorem plays one role" or "the paper only claims."
- Keep IDT framed as a benchmark calibration for this setting.
- Keep the few-shot/style-memory result framed as a successful extension.

### Section plan

#### Abstract

Use four moves:

1. IDT fixes the evaluation bug.
2. Euclidean latent flow suppresses style motion.
3. WD-VF changes coordinates and routing.
4. Final quality and cost numbers.

#### Introduction

Use five short paragraphs:

1. the evaluation bug
2. the Distinct5 choice and the IDT floor
3. the failure of low-LPIPS alone
4. the geometric cause and the WD-VF fix
5. the final cost-quality operating point

#### Related Work

Keep only three compact blocks:

1. exemplar-conditioned stylization
2. domain-conditional and diffusion-based baselines
3. wavelets and evaluation

#### Method

Present theory by role:

1. setup
2. Euclidean collapse theorem
3. Haar coordinate change
4. implemented objective
5. routed-limit proposition
6. endpoint alignment proposition

Each formal statement should be followed by:

- one sentence of intuition
- one explicit empirical consequence

#### Experiments

Interpret results around four questions:

1. does IDT change the ranking?
2. is WD-VF the best efficient trained-local point?
3. what remains stronger if CLIP-S alone is optimized?
4. which module choices are actually necessary?

#### Discussion and Conclusion

Close on:

1. why the method is fast
2. why Haar is the practical default
3. what the method does not cover
4. the main takeaway: real style transfer depends on the right geometry

### Immediate edits to apply

1. Rewrite the abstract for tighter sentence length.
2. Compress the introduction and contribution bullets.
3. Replace meta-theory phrasing with direct predictive phrasing.
4. Tighten the main-results paragraphs so they interpret, not enumerate.
5. Rewrite the controls/extensions discussion to foreground what each control proves.
6. Make the discussion more factual and less slogan-like.
7. Shorten the conclusion to one compact closing paragraph.
