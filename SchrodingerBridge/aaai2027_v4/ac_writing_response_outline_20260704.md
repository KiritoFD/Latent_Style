# AAAI 2027 Writing Response Outline

## Goal

Respond to the current AC-style writing concerns without adding unsupported claims or touching the locked main architecture figure.

## Core story

1. Art-to-art transfer is easy to overclaim because the source is already an artwork.
2. IDT fixes the evaluation bug by defining a no-op floor on this benchmark.
3. Euclidean latent flow has a local style-suppressed regime because structure and texture share one basis.
4. WD-VF weakens that conflict by changing coordinates, routing style through high-frequency bands, and stylizing only at the endpoint.
5. The empirical result is not just speed. It is positive-IDT transfer at low damage with a small local model.

## Writing decisions

1. Keep IDT framed as a benchmark calibration for this art-to-art setting, not a universal threshold.
2. Keep the theory framed as explanatory and predictive, not as a proof of universal superiority.
3. Keep efficiency as a result of the geometry, not as the only contribution.
4. Present style-memory update and 8-style extension as successful reuse, not failure.
5. Keep Haar justified by exact orthogonality, local support, and the measured db2 tradeoff.

## Section targets

### Abstract

- State the evaluation bug first.
- State the Euclidean failure mechanism second.
- State WD-VF as the geometric fix third.
- End with the main quality and efficiency numbers, plus one sentence linking the ablations back to the theory.

### Introduction

- Paragraph 1: no-op failure in art-to-art transfer.
- Paragraph 2: why Distinct5 uses the lowest-IDT styles and why SaMam illustrates the problem.
- Paragraph 3: theorem gives a local style-suppressed regime and specific predictions.
- Paragraph 4: WD-VF changes the transport geometry.
- Paragraph 5: final operating point and why it matters.

### Method and Theory

- Rename the section so the math is visibly central.
- Theorem 1 explains the Euclidean failure mode only.
- The routed-limit proposition identifies the active supervised channels only.
- Every formal block must feed at least one direct experiment statement later.

### Experiments

- Main table should carry the cost story.
- Main text should interpret ranking, not reread cells.
- Ablations should be organized by prediction, not by component count.
- Controls should show transfer-only robustness, pixel-vs-latent evidence, and parameter-efficient extension.

### Discussion and Limitations

- Explain why the method is fast in optimization terms.
- Explain why Haar is the default in this setup.
- State the actual scope limits directly: domain-conditional setup, narrow benchmark, single seed, local theory.

## What not to do

1. Do not call few-shot or style-memory update a failure.
2. Do not introduce Hessian-spectrum claims unless new experiments are run.
3. Do not add open-world baseline claims that are not measured under the current protocol.
4. Do not turn the manuscript into a rebuttal.
5. Do not touch the locked main architecture figure content.
