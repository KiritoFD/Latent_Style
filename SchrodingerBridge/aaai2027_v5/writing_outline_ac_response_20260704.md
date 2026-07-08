## Writing Revision Outline

### One-sentence thesis

Art-to-art transfer needs a direction-aware evaluation rule and a transport geometry that separates structure from texture; IDT provides the first, and WD-VF provides the second.

### Core narrative

1. Evaluation bug.
The source image is already an artwork, so a method can look good while doing almost nothing. IDT makes that failure visible.

2. Mechanism.
In Euclidean latent flow, layout and texture share one basis. This can create a local style-suppressed regime.

3. Geometric fix.
WD-VF changes coordinates with Haar wavelets, weakens LL supervision, routes style through high-frequency bands, and stylizes only at the endpoint.

4. Predictive theory.
The theory is local. Its purpose is to predict ablation behavior, not to claim universal superiority.

5. Empirical claim.
WD-VF is the strongest trained positive-IDT operating point on the current Distinct5 benchmark, with explicit training and inference cost.

### Section plan

1. Abstract.
Problem -> mechanism -> method -> headline evidence. One result sentence for quality and one for cost.

2. Introduction.
Start from the no-op failure. Explain why Distinct5 uses the five lowest-IDT styles. Use SaMam as the clearest motivating example. Then move to the geometry claim.

3. Theory and method.
After every theorem or proposition, add two plain-language sentences:
- what it explains
- what experiment should verify it

4. Main results.
Read the main table by role, not by raw score order:
- IDT changes ranking
- WD-VF is the best trained positive-IDT tradeoff
- large priors still define the CLIP-S ceiling
- cost is supporting evidence, not the sole claim

5. Ablations.
Organize by prediction, not by search history or module count.

6. Controls.
Use transfer-only, pixel-vs-latent, style-memory update, and 8-style extension only to show scope and robustness.

7. Discussion and limitations.
State four limits directly:
- domain-conditional scope
- narrow benchmark scope
- palette-heavy targets remain harder
- theory is local and evidence is single-seed

### Style constraints

1. Prefer short declarative sentences.
2. Use standard ML/CV terms only.
3. Do not use rebuttal tone.
4. Do not oversell theory.
5. Do not restate numbers that are already obvious from a table unless the number makes a specific argumentative point.
